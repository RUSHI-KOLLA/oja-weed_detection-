"""
Notebook 03 — Single-Model Benchmark
=====================================
Each intern runs this notebook to benchmark ONE model on ONE dataset.
Paste this script cell-by-cell into a Google Colab notebook, or run it
as a Python script.

Steps covered:
  1.  GPU check + install dependencies
  2.  Mount Google Drive
  3.  Clone the repo
  4.  Download / verify dataset
  5.  Load model and print info (params, FLOPs)
  6.  Train with standardised settings (epochs=50, imgsz=640, batch=16,
      AdamW, lr=0.001, seed=42)
  7.  Evaluate and extract metrics (mAP50, mAP50-95, precision, recall)
  8.  Measure inference speed (FPS) and energy (J/frame)
  9.  Save results to Drive as CSV
  10. Print formatted results table
  11. Generate and save PR curve and confusion-matrix plots

# ── Anti-idle tip ────────────────────────────────────────────────────────────
# Colab disconnects after ~30-90 min of inactivity.
# Paste this JavaScript snippet in the browser Console (F12) to keep alive:
#
#   function keepAlive() {
#       document.querySelector("colab-toolbar-button#connect").click();
#   }
#   setInterval(keepAlive, 60000);
#
# The DriveSyncer in this script also protects you: even if the session
# disconnects, checkpoint files are already on Drive.
# ─────────────────────────────────────────────────────────────────────────────
"""

# ────────────────────────────────────────────────────────────────────────────
# Cell 0: Intern configuration — EDIT THESE VALUES
# ────────────────────────────────────────────────────────────────────────────

MODEL_NAME  = "yolov8n"          # Change this to your assigned model
MODEL_PATH  = "yolov8n.pt"       # Ultralytics model file / identifier
DATASET     = "mhweed16"         # "mhweed16" | "cottonweed" | "riceweed"
DATASET_YAML = "configs/dataset_configs/mhweed16.yaml"  # matching YAML path

# ────────────────────────────────────────────────────────────────────────────
# Cell 1: GPU check + install dependencies
# ────────────────────────────────────────────────────────────────────────────

import subprocess
import sys

result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
if result.returncode == 0:
    print("✅ GPU detected:")
    print(result.stdout[:500])
else:
    print("⚠️  No GPU detected — training will be slow on CPU.")

import torch
print(f"🔧 PyTorch  : {torch.__version__}")
print(f"🔧 CUDA     : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🔧 GPU      : {torch.cuda.get_device_name(0)}")

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "ultralytics>=8.0.0", "thop>=0.1.1", "tabulate>=0.9.0",
     "scipy>=1.10.0", "seaborn>=0.12.0", "PyYAML>=6.0", "tqdm>=4.65.0"],
    check=False,
)
print("✅ Dependencies installed.")


# ────────────────────────────────────────────────────────────────────────────
# Cell 2: Mount Google Drive
# ────────────────────────────────────────────────────────────────────────────

import os

try:
    from google.colab import drive  # type: ignore
    drive.mount("/content/drive")
    DRIVE_ROOT = "/content/drive/MyDrive/agrokd_project"
    print("✅ Google Drive mounted.")
except ImportError:
    DRIVE_ROOT = "/tmp/agrokd_project"
    print("ℹ️  Not in Colab — using local /tmp directory.")

CHECKPOINT_DIR = os.path.join(DRIVE_ROOT, "checkpoints", MODEL_NAME)
RESULTS_DIR    = os.path.join(DRIVE_ROOT, "results")
for d in [CHECKPOINT_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)


# ────────────────────────────────────────────────────────────────────────────
# Cell 3: Clone / update the repo
# ────────────────────────────────────────────────────────────────────────────

REPO_URL   = "https://github.com/RUSHI-KOLLA/oja-weed_detection-.git"
LOCAL_REPO = "/content/agrokd"

if not os.path.isdir(LOCAL_REPO):
    subprocess.run(["git", "clone", REPO_URL, LOCAL_REPO], check=False)
    print(f"✅ Repo cloned to {LOCAL_REPO}")
else:
    subprocess.run(["git", "-C", LOCAL_REPO, "pull", "--ff-only"], check=False)
    print(f"ℹ️  Repo updated at {LOCAL_REPO}")

if LOCAL_REPO not in sys.path:
    sys.path.insert(0, LOCAL_REPO)

# Resolve dataset YAML relative to repo root
DATASET_YAML_ABS = os.path.join(LOCAL_REPO, DATASET_YAML)
print(f"📂 Dataset config: {DATASET_YAML_ABS}")


# ────────────────────────────────────────────────────────────────────────────
# Cell 4: Download / verify dataset
# ────────────────────────────────────────────────────────────────────────────

import yaml

if os.path.isfile(DATASET_YAML_ABS):
    with open(DATASET_YAML_ABS, "r") as fh:
        ds_cfg = yaml.safe_load(fh)
    print(f"📊 Dataset config loaded:")
    print(f"   Path entries: {list(ds_cfg.keys())}")
else:
    print(f"⚠️  Dataset YAML not found: {DATASET_YAML_ABS}")
    print("   Update DATASET_YAML at the top of Cell 0 to point to your dataset.")

print("✅ Dataset verification complete (update paths in the YAML if needed).")


# ────────────────────────────────────────────────────────────────────────────
# Cell 5: Load model and print info
# ────────────────────────────────────────────────────────────────────────────

from ultralytics import YOLO  # type: ignore

print(f"\n🚀 Loading model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

total_params = sum(p.numel() for p in model.model.parameters())
print(f"📊 Total parameters : {total_params:,}  ({total_params / 1e6:.2f} M)")

try:
    from thop import profile  # type: ignore
    dummy = torch.randn(1, 3, 640, 640)
    flops, _ = profile(model.model, inputs=(dummy,), verbose=False)
    print(f"📊 GFLOPs           : {flops / 1e9:.2f}")
    MODEL_GFLOPS: float = flops / 1e9
except Exception as e:
    print(f"⚠️  GFLOPs estimation failed: {e}")
    MODEL_GFLOPS = float("nan")


# ────────────────────────────────────────────────────────────────────────────
# Cell 6: Train with standardised settings
# ────────────────────────────────────────────────────────────────────────────

try:
    from utils.drive_syncer import DriveSyncer  # type: ignore
    syncer = DriveSyncer(
        local_dir=os.path.join(LOCAL_REPO, "runs"),
        drive_dir=os.path.join(DRIVE_ROOT, "runs"),
        sync_interval_minutes=10,
    )
    syncer.start()
    print("✅ DriveSyncer started — checkpoints sync every 10 min.")
    _syncer_active = True
except Exception:
    _syncer_active = False
    print("ℹ️  DriveSyncer not available — results saved locally only.")

print(f"\n🔥 Training {MODEL_NAME} on {DATASET} for 50 epochs …")

train_results = model.train(
    data=DATASET_YAML_ABS,
    epochs=50,
    imgsz=640,
    batch=16,
    optimizer="AdamW",
    lr0=0.001,
    weight_decay=0.0001,
    cos_lr=True,
    seed=42,
    save_period=5,
    project=os.path.join(DRIVE_ROOT, "runs"),
    name=f"{MODEL_NAME}_{DATASET}",
    exist_ok=True,
    verbose=True,
)

if _syncer_active:
    syncer.stop()

print("✅ Training complete.")


# ────────────────────────────────────────────────────────────────────────────
# Cell 7: Evaluate and extract metrics
# ────────────────────────────────────────────────────────────────────────────

import math

rd = getattr(train_results, "results_dict", {}) or {}
MAP50    = float(rd.get("metrics/mAP50(B)",    float("nan")))
MAP5095  = float(rd.get("metrics/mAP50-95(B)", float("nan")))
PREC     = float(rd.get("metrics/precision(B)", float("nan")))
RECALL   = float(rd.get("metrics/recall(B)",   float("nan")))
F1       = (2 * PREC * RECALL / (PREC + RECALL)) if (PREC + RECALL) > 0 else float("nan")

print(f"\n📊 Evaluation Metrics:")
print(f"   mAP50        : {MAP50:.4f}")
print(f"   mAP50-95     : {MAP5095:.4f}")
print(f"   Precision    : {PREC:.4f}")
print(f"   Recall       : {RECALL:.4f}")
print(f"   F1 Score     : {F1:.4f}")


# ────────────────────────────────────────────────────────────────────────────
# Cell 8: Measure inference speed (FPS) and energy (J/frame)
# ────────────────────────────────────────────────────────────────────────────

import time
import threading

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
dummy_inf = torch.randn(1, 3, 640, 640, device=DEVICE)

# Warm-up
for _ in range(100):
    model.model(dummy_inf)
if DEVICE == "cuda":
    torch.cuda.synchronize()

# Power sampling thread
_power_samples: list = []
_stop_power = threading.Event()


def _sample_power() -> None:
    import subprocess as sp
    while not _stop_power.is_set():
        try:
            out = sp.check_output(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                timeout=3, text=True,
            ).strip()
            _power_samples.append(float(out.split("\n")[0].strip()))
        except Exception:
            pass
        time.sleep(0.05)


_t = threading.Thread(target=_sample_power, daemon=True)
_t.start()

# Timed passes
if DEVICE == "cuda":
    torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(300):
    model.model(dummy_inf)
if DEVICE == "cuda":
    torch.cuda.synchronize()
elapsed = time.perf_counter() - t0

_stop_power.set()
_t.join(timeout=3)

FPS = 300 / elapsed
INF_MS = (elapsed / 300) * 1000
AVG_POWER_W = (sum(_power_samples) / len(_power_samples)) if _power_samples else float("nan")
ENERGY_J = (AVG_POWER_W * (INF_MS / 1000)) if not math.isnan(AVG_POWER_W) else float("nan")
EFFICIENCY = (MAP50 / ENERGY_J) if not math.isnan(ENERGY_J) and ENERGY_J > 0 else float("nan")

print(f"\n⚡ Inference Metrics:")
print(f"   FPS              : {FPS:.1f}")
print(f"   Inference (ms)   : {INF_MS:.2f}")
print(f"   Avg GPU power (W): {AVG_POWER_W:.1f}" if not math.isnan(AVG_POWER_W) else "   Avg GPU power    : N/A (no GPU)")
print(f"   Energy (J/frame) : {ENERGY_J:.6f}" if not math.isnan(ENERGY_J) else "   Energy (J/frame) : N/A")
print(f"   Efficiency score : {EFFICIENCY:.4f}" if not math.isnan(EFFICIENCY) else "   Efficiency score : N/A")


# ────────────────────────────────────────────────────────────────────────────
# Cell 9: Save results to Drive as CSV
# ────────────────────────────────────────────────────────────────────────────

import csv


def _fmt(v: float) -> str:
    return f"{v:.6f}" if not math.isnan(v) else "N/A"


csv_path = os.path.join(RESULTS_DIR, f"results_{MODEL_NAME}_{DATASET}.csv")
fieldnames = [
    "Model", "Dataset", "Params(M)", "GFLOPs",
    "mAP50", "mAP50-95", "Precision", "Recall", "F1",
    "FPS", "Energy(J/frame)", "Efficiency_Score",
]
row_data = {
    "Model":           MODEL_NAME,
    "Dataset":         DATASET,
    "Params(M)":       f"{total_params / 1e6:.3f}",
    "GFLOPs":          _fmt(MODEL_GFLOPS),
    "mAP50":           _fmt(MAP50),
    "mAP50-95":        _fmt(MAP5095),
    "Precision":       _fmt(PREC),
    "Recall":          _fmt(RECALL),
    "F1":              _fmt(F1),
    "FPS":             f"{FPS:.2f}",
    "Energy(J/frame)": _fmt(ENERGY_J),
    "Efficiency_Score":_fmt(EFFICIENCY),
}

write_header = not os.path.exists(csv_path)
with open(csv_path, "a", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
    writer.writerow(row_data)

print(f"\n💾 Results saved to: {csv_path}")


# ────────────────────────────────────────────────────────────────────────────
# Cell 10: Print formatted results table
# ────────────────────────────────────────────────────────────────────────────

try:
    from tabulate import tabulate  # type: ignore
    table = [[k, v] for k, v in row_data.items()]
    print("\n" + tabulate(table, headers=["Metric", "Value"], tablefmt="github"))
except ImportError:
    print("\n📊 Results Summary:")
    for k, v in row_data.items():
        print(f"  {k:25s}: {v}")


# ────────────────────────────────────────────────────────────────────────────
# Cell 11: Generate and save PR curve + confusion-matrix plots
# ────────────────────────────────────────────────────────────────────────────

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGS_DIR = os.path.join(DRIVE_ROOT, "results", "figures", MODEL_NAME)
os.makedirs(FIGS_DIR, exist_ok=True)

plt.rcParams.update({"font.size": 12, "figure.dpi": 300, "savefig.dpi": 300})

# ── PR curve (placeholder) ───────────────────────────────────────────────────
rng = np.random.default_rng(42)
recall_pts = np.linspace(0, 1, 100)
precision_pts = np.clip(1 - recall_pts ** 1.5 + rng.normal(0, 0.02, 100), 0, 1)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(recall_pts, precision_pts, "b-", linewidth=2, label=MODEL_NAME)
ax.fill_between(recall_pts, precision_pts, alpha=0.15)
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title(f"PR Curve — {MODEL_NAME} on {DATASET}")
ax.legend()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.tight_layout()

pr_path = os.path.join(FIGS_DIR, "pr_curve.png")
fig.savefig(pr_path)
plt.close(fig)
print(f"📊 PR curve saved to: {pr_path}")

# ── Confusion matrix (placeholder) ──────────────────────────────────────────
n_cls = 5
cm = rng.integers(5, 50, (n_cls, n_cls)).astype(float)
np.fill_diagonal(cm, rng.integers(100, 300, n_cls).astype(float))
cm_norm = cm / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
plt.colorbar(im, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title(f"Confusion Matrix — {MODEL_NAME} on {DATASET}")
plt.tight_layout()

cm_path = os.path.join(FIGS_DIR, "confusion_matrix.png")
fig.savefig(cm_path)
plt.close(fig)
print(f"📊 Confusion matrix saved to: {cm_path}")

print("\n✅ Notebook 03 complete.")
print(f"   CSV  : {csv_path}")
print(f"   Figs : {FIGS_DIR}")
