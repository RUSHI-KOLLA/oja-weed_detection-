"""
Energy Profiling Utility
========================
Standalone utility for measuring the GPU energy consumption of any
Ultralytics YOLO model (or custom PyTorch model) during inference.

How it works
------------
``nvidia-smi`` is polled in a background thread while inference passes run
on the main thread.  Power samples are averaged and multiplied by the mean
inference time to estimate energy per frame.

Usage (standalone)
------------------
    python scripts/energy_profiler.py --model yolov8n.pt --img-size 640

Usage (as a library)
--------------------
    from scripts.energy_profiler import profile_energy

    results = profile_energy(model, img_size=640)
    print(results)
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
import threading
from typing import Any, Dict, List, Optional

import torch

# ---------------------------------------------------------------------------
# ── nvidia-smi helpers ─────────────────────────────────────────────────────
# ---------------------------------------------------------------------------


def _read_gpu_power() -> Optional[float]:
    """Return instantaneous GPU power draw in Watts, or None on failure."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            timeout=5,
            text=True,
        ).strip()
        lines = [l.strip() for l in out.split("\n") if l.strip()]
        if lines:
            return float(lines[0])
    except Exception:
        pass
    return None


class _PowerSampler:
    """Background thread that continuously samples GPU power."""

    def __init__(self, interval_s: float = 0.05) -> None:
        self.interval_s = interval_s
        self.samples: List[float] = []
        self._stop_flag = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "_PowerSampler":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop_flag.set()
        self._thread.join(timeout=5)

    def average_w(self) -> Optional[float]:
        return (sum(self.samples) / len(self.samples)) if self.samples else None

    def _run(self) -> None:
        while not self._stop_flag.is_set():
            pwr = _read_gpu_power()
            if pwr is not None:
                self.samples.append(pwr)
            time.sleep(self.interval_s)


# ---------------------------------------------------------------------------
# ── Main profiling function ────────────────────────────────────────────────
# ---------------------------------------------------------------------------


def profile_energy(
    model: Any,
    img_size: int = 640,
    warmup_passes: int = 100,
    timed_passes: int = 300,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """
    Profile the inference energy consumption of *model*.

    Parameters
    ----------
    model:
        An Ultralytics YOLO instance or a raw ``torch.nn.Module``.
        The callable used is ``model.model`` if that attribute exists,
        otherwise ``model`` itself.
    img_size:
        Square input size in pixels.
    warmup_passes:
        Number of forward passes before timing begins.
    timed_passes:
        Number of timed forward passes.
    device:
        ``"cuda"`` or ``"cpu"``.  Auto-detected if None.

    Returns
    -------
    dict with keys:
        ``fps``              — frames per second
        ``inference_ms``    — mean inference time per frame (ms)
        ``avg_power_w``     — mean GPU power draw (W), NaN if unavailable
        ``energy_j_per_frame`` — energy per frame (J) = power × time
        ``total_energy_j``  — total energy for all timed passes (J)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Resolve the callable
    callable_model = getattr(model, "model", model)
    if device == "cuda":
        try:
            callable_model = callable_model.cuda()
        except Exception:
            device = "cpu"

    dummy = torch.randn(1, 3, img_size, img_size, device=device)

    results: Dict[str, float] = {
        "fps": float("nan"),
        "inference_ms": float("nan"),
        "avg_power_w": float("nan"),
        "energy_j_per_frame": float("nan"),
        "total_energy_j": float("nan"),
    }

    try:
        # Warm-up
        for _ in range(warmup_passes):
            callable_model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()

        # Start power sampler (background thread)
        sampler = _PowerSampler(interval_s=0.05).start()

        # Timed passes
        if device == "cuda":
            torch.cuda.synchronize()
        t_start = time.perf_counter()
        for _ in range(timed_passes):
            callable_model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()
        t_end = time.perf_counter()

        sampler.stop()
        elapsed = t_end - t_start

        fps = timed_passes / elapsed
        inf_ms = (elapsed / timed_passes) * 1000.0
        avg_pwr = sampler.average_w()
        avg_pwr_float = avg_pwr if avg_pwr is not None else float("nan")
        energy_j = (
            avg_pwr_float * (inf_ms / 1000.0)
            if not math.isnan(avg_pwr_float)
            else float("nan")
        )
        total_energy = (
            avg_pwr_float * elapsed
            if not math.isnan(avg_pwr_float)
            else float("nan")
        )

        results.update(
            fps=fps,
            inference_ms=inf_ms,
            avg_power_w=avg_pwr_float,
            energy_j_per_frame=energy_j,
            total_energy_j=total_energy,
        )

    except Exception as exc:
        print(f"⚠️  Energy profiling failed: {exc}")

    return results


# ---------------------------------------------------------------------------
# ── CLI entry-point ────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure GPU energy consumption for an Ultralytics YOLO model."
    )
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics model path or name.")
    parser.add_argument("--img-size", type=int, default=640, help="Input image size (pixels).")
    parser.add_argument("--warmup", type=int, default=100, help="Number of warm-up passes.")
    parser.add_argument("--passes", type=int, default=300, help="Number of timed passes.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        print("❌ ultralytics not installed.  Run: pip install ultralytics")
        sys.exit(1)

    print(f"🔍 Loading model: {args.model}")
    try:
        model = YOLO(args.model)
    except Exception as exc:
        print(f"❌ Could not load model: {exc}")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Device: {device}")
    print(f"⚡ Running {args.warmup} warm-up + {args.passes} timed passes …\n")

    results = profile_energy(
        model,
        img_size=args.img_size,
        warmup_passes=args.warmup,
        timed_passes=args.passes,
    )

    def _fmt(v: float) -> str:
        return f"{v:.4f}" if not math.isnan(v) else "N/A"

    print("=" * 45)
    print("⚡  ENERGY PROFILE RESULTS")
    print("=" * 45)
    print(f"  FPS                  : {_fmt(results['fps'])}")
    print(f"  Inference time (ms)  : {_fmt(results['inference_ms'])}")
    print(f"  Avg GPU power (W)    : {_fmt(results['avg_power_w'])}")
    print(f"  Energy / frame (J)   : {_fmt(results['energy_j_per_frame'])}")
    print(f"  Total energy (J)     : {_fmt(results['total_energy_j'])}")
    print("=" * 45)


if __name__ == "__main__":
    main()
