"""
Benchmark script: compare 10 object-detection / classification models on a
crop–weed dataset.

For each model the script:
1. Loads the Ultralytics model and counts parameters.
2. Computes FLOPs (via thop).
3. Trains for EPOCHS on the configured dataset.
4. Measures inference speed (FPS).
5. Records mAP50 and mAP50-95 from training metrics.
6. Saves a results CSV and prints a summary table.

Usage
-----
    python scripts/benchmark_models.py

Edit the ``# ── Configuration ──`` block below to point to your dataset YAML
and to adjust training hyper-parameters.
"""

from __future__ import annotations

import os
import sys
import time
import csv
from typing import Any, Dict, List

import torch

# ---------------------------------------------------------------------------
# ── Configuration ──────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

DATASET_YAML: str = "configs/dataset_configs/cottonweed.yaml"
EPOCHS: int = 50
IMG_SIZE: int = 640
BATCH_SIZE: int = 16
SEED: int = 42
SAVE_DIR: str = "results/benchmarks"

# Models to benchmark: display name → Ultralytics model identifier
BENCHMARK_MODELS: Dict[str, str] = {
    "YOLOv8n":       "yolov8n.pt",
    "YOLOv8s":       "yolov8s.pt",
    "YOLOv8m":       "yolov8m.pt",
    "YOLOv9t":       "yolov9t.pt",
    "YOLOv9s":       "yolov9s.pt",
    "YOLOv10n":      "yolov10n.pt",
    "YOLOv10s":      "yolov10s.pt",
    "YOLOv11n":      "yolo11n.pt",
    "RT-DETR-L":     "rtdetr-l.pt",
    "YOLOv8n-World": "yolov8s-worldv2.pt",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def count_parameters(model: Any) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.model.parameters() if p.requires_grad)


def compute_flops(model: Any, img_size: int = 640) -> float:
    """Return GFLOPs for a single (1, 3, img_size, img_size) forward pass."""
    try:
        from thop import profile  # type: ignore

        dummy = torch.randn(1, 3, img_size, img_size)
        flops, _ = profile(model.model, inputs=(dummy,), verbose=False)
        return flops / 1e9
    except Exception:
        return float("nan")


def measure_fps(model: Any, img_size: int = 640, n_iters: int = 100) -> float:
    """Warm-up then time *n_iters* inference calls; return FPS."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dummy = torch.randn(1, 3, img_size, img_size, device=device)

    # Warm-up
    for _ in range(10):
        _ = model.model(dummy)

    start = time.perf_counter()
    for _ in range(n_iters):
        _ = model.model(dummy)
    elapsed = time.perf_counter() - start
    return n_iters / elapsed


def train_model(
    model: Any,
    name: str,
    run_dir: str,
) -> Dict[str, float]:
    """
    Train *model* and return a dict of metrics extracted from results.csv.

    Returns
    -------
    dict with keys: map50, map50_95 (floats, NaN on failure)
    """
    metrics: Dict[str, float] = {"map50": float("nan"), "map50_95": float("nan")}
    try:
        results = model.train(
            data=DATASET_YAML,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            seed=SEED,
            project=run_dir,
            name=name,
            exist_ok=True,
            verbose=False,
        )
        # Ultralytics stores metrics in results.results_dict
        rd = getattr(results, "results_dict", {})
        metrics["map50"] = float(rd.get("metrics/mAP50(B)", float("nan")))
        metrics["map50_95"] = float(rd.get("metrics/mAP50-95(B)", float("nan")))
    except Exception as exc:
        print(f"  ❌ Training failed for {name}: {exc}")
    return metrics


def print_table(rows: List[Dict[str, Any]]) -> None:
    """Print a formatted comparison table to stdout."""
    headers = ["Model", "Params (M)", "GFLOPs", "mAP50", "mAP50-95", "FPS"]
    col_w = [max(len(h), 14) for h in headers]

    sep = "+" + "+".join("-" * (w + 2) for w in col_w) + "+"
    print(sep)
    header_row = "| " + " | ".join(h.center(col_w[i]) for i, h in enumerate(headers)) + " |"
    print(header_row)
    print(sep)

    for row in rows:
        vals = [
            row["name"],
            f"{row['params'] / 1e6:.2f}",
            f"{row['gflops']:.2f}" if row["gflops"] == row["gflops"] else "N/A",
            f"{row['map50']:.4f}" if row["map50"] == row["map50"] else "N/A",
            f"{row['map50_95']:.4f}" if row["map50_95"] == row["map50_95"] else "N/A",
            f"{row['fps']:.1f}" if row["fps"] == row["fps"] else "N/A",
        ]
        print("| " + " | ".join(v.ljust(col_w[i]) for i, v in enumerate(vals)) + " |")

    print(sep)


def save_csv(rows: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["name", "params", "gflops", "map50", "map50_95", "fps"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"📊 Results saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        print("❌ ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    # Import AutoCheckpointSaver for safety (not used directly here because
    # Ultralytics manages its own checkpoints, but we keep the import so that
    # the dependency is verified).
    try:
        from utils.auto_checkpoint_saver import AutoCheckpointSaver  # noqa: F401
        print("✅ AutoCheckpointSaver available")
    except ImportError:
        print("⚠️  utils.auto_checkpoint_saver not found — continuing without it")

    os.makedirs(SAVE_DIR, exist_ok=True)
    run_dir = os.path.join(SAVE_DIR, "runs")

    rows: List[Dict[str, Any]] = []

    for display_name, model_id in BENCHMARK_MODELS.items():
        print(f"\n{'=' * 60}")
        print(f"🚀 Benchmarking: {display_name} ({model_id})")
        print(f"{'=' * 60}")

        try:
            model = YOLO(model_id)
        except Exception as exc:
            print(f"  ❌ Could not load {model_id}: {exc}")
            rows.append(
                dict(name=display_name, params=0, gflops=float("nan"),
                     map50=float("nan"), map50_95=float("nan"), fps=float("nan"))
            )
            continue

        params = count_parameters(model)
        gflops = compute_flops(model, IMG_SIZE)
        print(f"  📊 Parameters: {params / 1e6:.2f} M | GFLOPs: {gflops:.2f}")

        metrics = train_model(model, display_name, run_dir)
        fps = measure_fps(model, IMG_SIZE)

        print(
            f"  ✅ mAP50={metrics['map50']:.4f} | "
            f"mAP50-95={metrics['map50_95']:.4f} | FPS={fps:.1f}"
        )

        rows.append(
            dict(
                name=display_name,
                params=params,
                gflops=gflops,
                map50=metrics["map50"],
                map50_95=metrics["map50_95"],
                fps=fps,
            )
        )

    csv_path = os.path.join(SAVE_DIR, "benchmark_results.csv")
    save_csv(rows, csv_path)

    print(f"\n{'=' * 60}")
    print("📊 BENCHMARK SUMMARY")
    print(f"{'=' * 60}")
    print_table(rows)


if __name__ == "__main__":
    main()
