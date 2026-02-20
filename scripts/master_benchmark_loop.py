"""
Master Benchmark Loop
=====================
Runs the full 10 model × 3 dataset benchmark grid for the AgroKD-Net
research sprint.

For every model/dataset pair the script:
  1. Loads the Ultralytics model.
  2. Counts parameters and estimates GFLOPs.
  3. Trains with standardised settings (from configs/experiment_config.yaml).
  4. Extracts mAP50 and mAP50-95 from training results.
  5. Measures inference FPS (100 warm-up + 300 timed passes).
  6. Reads GPU power via nvidia-smi to estimate energy per frame.
  7. Computes Efficiency Score = mAP50 / (GFLOPs / 1e9).
  8. Appends a row to a running CSV on Google Drive.

Usage
-----
    python scripts/master_benchmark_loop.py

Google Colab
------------
This script auto-mounts Google Drive when running inside Colab and saves
all results to ``/content/drive/MyDrive/agrokd_project/``.
"""

from __future__ import annotations

import csv
import math
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import torch
import yaml

# ---------------------------------------------------------------------------
# ── Paths ──────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CFG_PATH = os.path.join(_REPO_ROOT, "configs", "experiment_config.yaml")

# ---------------------------------------------------------------------------
# ── Load experiment config ─────────────────────────────────────────────────
# ---------------------------------------------------------------------------


def _load_config() -> Dict[str, Any]:
    """Load ``configs/experiment_config.yaml`` and return as dict."""
    with open(_CFG_PATH, "r") as fh:
        return yaml.safe_load(fh)


_CFG = _load_config()
_TRAIN = _CFG["training"]
_INF = _CFG["inference"]
_OUT = _CFG["output"]

# Training hyper-parameters (all sourced from the YAML)
EPOCHS: int = int(_TRAIN["epochs"])
IMG_SIZE: int = int(_TRAIN["imgsz"])
BATCH: int = int(_TRAIN["batch"])
OPTIMIZER: str = str(_TRAIN["optimizer"])
LR0: float = float(_TRAIN["lr0"])
WEIGHT_DECAY: float = float(_TRAIN["weight_decay"])
COS_LR: bool = bool(_TRAIN["cos_lr"])
SEED: int = int(_TRAIN["seed"])
SAVE_PERIOD: int = int(_TRAIN["save_period"])

# Inference settings
WARMUP_PASSES: int = int(_INF["warmup_passes"])
TIMED_PASSES: int = int(_INF["timed_passes"])

# ---------------------------------------------------------------------------
# ── Model / dataset registry ───────────────────────────────────────────────
# ---------------------------------------------------------------------------

BENCHMARK_MODELS: Dict[str, str] = {
    name: path for name, path in _CFG["benchmark_models"].items()
}

DATASETS: List[Dict[str, str]] = [
    {
        "name": ds["name"],
        "config": os.path.join(_REPO_ROOT, ds["config"]),
        "task": ds["task"],
    }
    for ds in _CFG["datasets"]
]

# ---------------------------------------------------------------------------
# ── Google Drive / output setup ────────────────────────────────────────────
# ---------------------------------------------------------------------------


def _setup_drive() -> str:
    """
    Mount Google Drive inside Colab (no-op outside Colab) and return the
    project root path on Drive (or a local fallback).
    """
    try:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive")
        drive_root = _OUT["drive_path"]
        print(f"✅ Google Drive mounted — project root: {drive_root}")
    except ImportError:
        drive_root = os.path.join(_REPO_ROOT, _OUT["results_dir"])
        print(f"ℹ️  Not in Colab — saving results to {drive_root}")

    os.makedirs(drive_root, exist_ok=True)
    return drive_root


# ---------------------------------------------------------------------------
# ── Parameter / FLOPs helpers ─────────────────────────────────────────────
# ---------------------------------------------------------------------------


def count_parameters(model: Any) -> int:
    """Return the total number of parameters in an Ultralytics model."""
    return sum(p.numel() for p in model.model.parameters())


def compute_gflops(model: Any, img_size: int = 640) -> float:
    """Return GFLOPs for a single (1, 3, img_size, img_size) forward pass."""
    try:
        from thop import profile  # type: ignore

        dummy = torch.randn(1, 3, img_size, img_size)
        flops, _ = profile(model.model, inputs=(dummy,), verbose=False)
        return flops / 1e9
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# ── Training ───────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------


def train_model(
    model: Any,
    run_name: str,
    dataset_yaml: str,
    save_dir: str,
) -> Dict[str, float]:
    """
    Train *model* on *dataset_yaml* and return mAP metrics.

    Returns
    -------
    dict with keys ``map50`` and ``map50_95`` (floats, NaN on failure).
    """
    metrics: Dict[str, float] = {"map50": float("nan"), "map50_95": float("nan")}
    try:
        results = model.train(
            data=dataset_yaml,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH,
            optimizer=OPTIMIZER,
            lr0=LR0,
            weight_decay=WEIGHT_DECAY,
            cos_lr=COS_LR,
            seed=SEED,
            save_period=SAVE_PERIOD,
            project=save_dir,
            name=run_name,
            exist_ok=True,
            verbose=False,
        )
        rd = getattr(results, "results_dict", {}) or {}
        metrics["map50"] = float(rd.get("metrics/mAP50(B)", float("nan")))
        metrics["map50_95"] = float(rd.get("metrics/mAP50-95(B)", float("nan")))
    except Exception as exc:
        print(f"  ❌ Training error for {run_name}: {exc}")
    return metrics


# ---------------------------------------------------------------------------
# ── Inference speed ────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------


def measure_fps(model: Any, img_size: int = 640) -> float:
    """
    Return inference FPS using WARMUP_PASSES warm-up and TIMED_PASSES
    timed forward passes (batch size 1).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dummy = torch.randn(1, 3, img_size, img_size, device=device)

    try:
        for _ in range(WARMUP_PASSES):
            model.model(dummy)

        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(TIMED_PASSES):
            model.model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        return TIMED_PASSES / elapsed
    except Exception as exc:
        print(f"  ⚠️  FPS measurement failed: {exc}")
        return float("nan")


# ---------------------------------------------------------------------------
# ── Energy estimation ──────────────────────────────────────────────────────
# ---------------------------------------------------------------------------


def read_gpu_power_w() -> Optional[float]:
    """Return current GPU power draw in Watts via nvidia-smi, or None."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            timeout=5,
            text=True,
        ).strip()
        return float(out.split("\n")[0].strip())
    except Exception:
        return None


def measure_energy(model: Any, img_size: int = 640) -> Dict[str, float]:
    """
    Measure average GPU power and inference time, returning energy per frame.

    Returns
    -------
    dict with keys ``fps``, ``avg_power_w``, ``inference_ms``,
    ``energy_j_per_frame``.
    """
    result: Dict[str, float] = {
        "fps": float("nan"),
        "avg_power_w": float("nan"),
        "inference_ms": float("nan"),
        "energy_j_per_frame": float("nan"),
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dummy = torch.randn(1, 3, img_size, img_size, device=device)

    try:
        # Warm-up
        for _ in range(WARMUP_PASSES):
            model.model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()

        # Timed passes with power sampling
        power_readings: List[float] = []
        start = time.perf_counter()
        for i in range(TIMED_PASSES):
            model.model(dummy)
            if device == "cuda":
                torch.cuda.synchronize()
            # Sample power every 10 frames to reduce overhead
            if i % 10 == 0:
                pwr = read_gpu_power_w()
                if pwr is not None:
                    power_readings.append(pwr)
        elapsed = time.perf_counter() - start

        fps = TIMED_PASSES / elapsed
        inference_ms = (elapsed / TIMED_PASSES) * 1000
        avg_power = float("nan") if not power_readings else sum(power_readings) / len(power_readings)
        energy_j = (avg_power * (inference_ms / 1000)) if not math.isnan(avg_power) else float("nan")

        result.update(
            fps=fps,
            avg_power_w=avg_power,
            inference_ms=inference_ms,
            energy_j_per_frame=energy_j,
        )
    except Exception as exc:
        print(f"  ⚠️  Energy measurement failed: {exc}")

    return result


# ---------------------------------------------------------------------------
# ── Efficiency score ────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------


def compute_efficiency_score(map50: float, energy_j: float) -> float:
    """
    η = mAP@0.5 / energy_per_frame_J.

    Returns NaN if either value is NaN or energy is zero.
    """
    if math.isnan(map50) or math.isnan(energy_j) or energy_j == 0:
        return float("nan")
    return map50 / energy_j


# ---------------------------------------------------------------------------
# ── CSV persistence ────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

_CSV_COLUMNS = [
    "Model",
    "Dataset",
    "Params(M)",
    "GFLOPs",
    "mAP50",
    "mAP50-95",
    "FPS",
    "Energy(J/frame)",
    "Efficiency_Score",
]


def _csv_path(drive_root: str) -> str:
    path = os.path.join(drive_root, "results", "benchmark_results.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def _append_csv_row(path: str, row: Dict[str, Any]) -> None:
    """Append one result row to the CSV, creating headers if the file is new."""
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in _CSV_COLUMNS})


# ---------------------------------------------------------------------------
# ── Results table ──────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------


def _print_table(rows: List[Dict[str, Any]]) -> None:
    """Print a formatted summary table of all benchmark results."""
    try:
        from tabulate import tabulate  # type: ignore

        table_data = [
            [
                r.get("Model", ""),
                r.get("Dataset", ""),
                r.get("Params(M)", ""),
                r.get("GFLOPs", ""),
                r.get("mAP50", ""),
                r.get("mAP50-95", ""),
                r.get("FPS", ""),
                r.get("Energy(J/frame)", ""),
                r.get("Efficiency_Score", ""),
            ]
            for r in rows
        ]
        print(tabulate(table_data, headers=_CSV_COLUMNS, tablefmt="github", floatfmt=".4f"))
    except ImportError:
        # Fallback manual formatting
        col_w = [max(len(c), 12) for c in _CSV_COLUMNS]
        sep = "+" + "+".join("-" * (w + 2) for w in col_w) + "+"
        header = "| " + " | ".join(c.center(col_w[i]) for i, c in enumerate(_CSV_COLUMNS)) + " |"
        print(sep)
        print(header)
        print(sep)
        for row in rows:
            vals = [str(row.get(col, ""))[:col_w[i]] for i, col in enumerate(_CSV_COLUMNS)]
            print("| " + " | ".join(v.ljust(col_w[i]) for i, v in enumerate(vals)) + " |")
        print(sep)


# ---------------------------------------------------------------------------
# ── Main benchmark loop ────────────────────────────────────────────────────
# ---------------------------------------------------------------------------


def main() -> None:
    # ── imports ──────────────────────────────────────────────────────────────
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        print("❌ ultralytics not installed.  Run: pip install ultralytics")
        sys.exit(1)

    drive_root = _setup_drive()
    csv_path = _csv_path(drive_root)
    runs_dir = os.path.join(drive_root, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    print(f"\n📋 Experiment config: {_CFG_PATH}")
    print(f"📊 Results CSV: {csv_path}")
    print(f"🗓  {len(BENCHMARK_MODELS)} models × {len(DATASETS)} datasets = "
          f"{len(BENCHMARK_MODELS) * len(DATASETS)} runs\n")

    all_rows: List[Dict[str, Any]] = []

    for model_name, model_path in BENCHMARK_MODELS.items():
        for ds in DATASETS:
            ds_name = ds["name"]
            ds_yaml = ds["config"]
            run_name = f"{model_name}_{ds_name}"

            print(f"\n{'=' * 65}")
            print(f"🚀  {run_name}  ({model_path}  ×  {ds_name})")
            print(f"{'=' * 65}")

            row: Dict[str, Any] = {
                "Model": model_name,
                "Dataset": ds_name,
                "Params(M)": "ERR",
                "GFLOPs": "ERR",
                "mAP50": "ERR",
                "mAP50-95": "ERR",
                "FPS": "ERR",
                "Energy(J/frame)": "ERR",
                "Efficiency_Score": "ERR",
            }

            try:
                # 1. Load model
                model = YOLO(model_path)

                # 2. Parameters & GFLOPs
                params = count_parameters(model)
                gflops = compute_gflops(model, IMG_SIZE)
                row["Params(M)"] = f"{params / 1e6:.3f}"
                row["GFLOPs"] = f"{gflops:.2f}" if not math.isnan(gflops) else "N/A"
                print(f"  📊 Params: {params / 1e6:.2f} M  |  GFLOPs: {row['GFLOPs']}")

                # 3. Train
                print(f"  🔥 Training for {EPOCHS} epochs …")
                train_metrics = train_model(model, run_name, ds_yaml, runs_dir)
                row["mAP50"] = f"{train_metrics['map50']:.4f}"
                row["mAP50-95"] = f"{train_metrics['map50_95']:.4f}"
                print(f"  ✅ mAP50={row['mAP50']}  mAP50-95={row['mAP50-95']}")

                # 4. Energy + FPS
                print("  ⚡ Measuring FPS and energy …")
                energy_info = measure_energy(model, IMG_SIZE)
                fps = energy_info["fps"]
                energy_j = energy_info["energy_j_per_frame"]
                row["FPS"] = f"{fps:.1f}" if not math.isnan(fps) else "N/A"
                row["Energy(J/frame)"] = (
                    f"{energy_j:.6f}" if not math.isnan(energy_j) else "N/A"
                )
                print(f"  ⚡ FPS={row['FPS']}  Energy={row['Energy(J/frame)']} J/frame")

                # 5. Efficiency score
                eff = compute_efficiency_score(train_metrics["map50"], energy_j)
                row["Efficiency_Score"] = f"{eff:.4f}" if not math.isnan(eff) else "N/A"

            except Exception as exc:
                print(f"  ❌ {run_name} failed: {exc}")

            # Persist immediately so a crash doesn't lose prior work
            _append_csv_row(csv_path, row)
            all_rows.append(row)
            print(f"  💾 Row saved to {csv_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("📊  FULL BENCHMARK RESULTS")
    print(f"{'=' * 65}\n")
    _print_table(all_rows)
    print(f"\n✅  All results saved to {csv_path}")


if __name__ == "__main__":
    main()
