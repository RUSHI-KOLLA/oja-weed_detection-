"""
Notebook 01 — Setup and Data Exploration
=========================================
Paste this script cell-by-cell into a Google Colab notebook, or run it
directly as a Python script.

Steps covered:
  1. GPU check
  2. Install dependencies
  3. Mount Google Drive
  4. Create project folder structure
  5. Download a sample dataset (MH-Weed16 via Kaggle)
  6. Explore and visualise data
  7. Count class distributions
"""

# ────────────────────────────────────────────────────────────────────────────
# Cell 1: GPU check
# ────────────────────────────────────────────────────────────────────────────

import subprocess, sys

result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
if result.returncode == 0:
    print("✅ GPU detected:")
    print(result.stdout[:500])
else:
    print("⚠️  No GPU detected — training will be slow on CPU.")

import torch
print(f"🔧 PyTorch version : {torch.__version__}")
print(f"🔧 CUDA available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🔧 GPU             : {torch.cuda.get_device_name(0)}")


# ────────────────────────────────────────────────────────────────────────────
# Cell 2: Install dependencies
# ────────────────────────────────────────────────────────────────────────────

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "ultralytics>=8.0.0", "thop>=0.1.1", "albumentations>=1.3.0",
     "wandb>=0.15.0", "pandas>=2.0.0", "matplotlib>=3.7.0",
     "seaborn>=0.12.0", "scikit-learn>=1.2.0", "opencv-python>=4.7.0",
     "PyYAML>=6.0", "tqdm>=4.65.0"],
    check=False,
)
print("✅ Dependencies installed.")


# ────────────────────────────────────────────────────────────────────────────
# Cell 3: Mount Google Drive
# ────────────────────────────────────────────────────────────────────────────

try:
    from google.colab import drive  # type: ignore
    drive.mount("/content/drive")
    DRIVE_ROOT = "/content/drive/MyDrive/agrokd_project"
    print("✅ Google Drive mounted.")
except ImportError:
    DRIVE_ROOT = "/tmp/agrokd_project"
    print("ℹ️  Not in Colab — using local /tmp directory.")

import os
os.makedirs(DRIVE_ROOT, exist_ok=True)


# ────────────────────────────────────────────────────────────────────────────
# Cell 4: Create project folder structure
# ────────────────────────────────────────────────────────────────────────────

FOLDERS = [
    "datasets/mhweed16",
    "datasets/cottonweed",
    "datasets/riceweed",
    "checkpoints",
    "results/benchmarks",
    "results/agrokd",
]

for folder in FOLDERS:
    path = os.path.join(DRIVE_ROOT, folder)
    os.makedirs(path, exist_ok=True)
    print(f"  📁 {path}")

print("✅ Project folders created.")


# ────────────────────────────────────────────────────────────────────────────
# Cell 5: Clone project repo (if not already present)
# ────────────────────────────────────────────────────────────────────────────

REPO_URL = "https://github.com/RUSHI-KOLLA/oja-weed_detection-.git"
LOCAL_REPO = "/content/agrokd"

if not os.path.isdir(LOCAL_REPO):
    subprocess.run(["git", "clone", REPO_URL, LOCAL_REPO], check=False)
    print(f"✅ Repo cloned to {LOCAL_REPO}")
else:
    print(f"ℹ️  Repo already exists at {LOCAL_REPO}")

if LOCAL_REPO not in sys.path:
    sys.path.insert(0, LOCAL_REPO)


# ────────────────────────────────────────────────────────────────────────────
# Cell 6: Sample dataset exploration (placeholder — replace with real data)
# ────────────────────────────────────────────────────────────────────────────

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

# Simulate class distribution for demonstration
CLASS_NAMES = [
    "Carpetweeds", "Crabgrass", "Eclipta", "Goosegrass", "Morningglory",
    "Nutsedge", "PalmerAmaranth", "PricklySida", "Purslane", "Ragweed",
    "Sicklepod", "SpottedSpurge", "SpurredAnoda", "Swinecress", "Waterhemp",
    "Cotton",
]
# Synthetic counts (replace with real dataset stats)
rng = np.random.default_rng(42)
counts = rng.integers(200, 800, size=len(CLASS_NAMES))

print("\n📊 Simulated class distribution (CottonWeed):")
for name, cnt in zip(CLASS_NAMES, counts):
    bar = "█" * (cnt // 40)
    print(f"  {name:20s} {cnt:4d}  {bar}")

print(f"\n  Total images (simulated): {counts.sum()}")
print(f"  Most common class : {CLASS_NAMES[counts.argmax()]} ({counts.max()})")
print(f"  Rarest class      : {CLASS_NAMES[counts.argmin()]} ({counts.min()})")
print(f"  Imbalance ratio   : {counts.max() / counts.min():.1f}×")


# ────────────────────────────────────────────────────────────────────────────
# Cell 7: Visualise class distribution
# ────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 5))
ax.barh(CLASS_NAMES, counts, color="steelblue")
ax.set_xlabel("Number of images")
ax.set_title("CottonWeed — Class Distribution (simulated)")
plt.tight_layout()

plot_path = "/tmp/class_distribution.png"
plt.savefig(plot_path, dpi=100)
print(f"📊 Class distribution chart saved to {plot_path}")
plt.close()

print("\n✅ Setup and exploration complete.")
print("   Proceed to notebook 02 to start training with auto-save.")
