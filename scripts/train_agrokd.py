"""
Training script for AgroKD-Net with automatic checkpoint saving.

Designed for Google Colab: mounts Google Drive, resumes from the last
checkpoint automatically, and saves progress every SAVE_EVERY_MINUTES to
prevent data loss on disconnects.

Usage
-----
    python scripts/train_agrokd.py

Edit the ``# ── Configuration ──`` block to customise paths and hyper-params.
"""

from __future__ import annotations

import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.optim as optim

# Ensure project root is on sys.path when running from scripts/
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.agrokd_net import AgroKDNet
from utils.auto_checkpoint_saver import AutoCheckpointSaver, resume_from_checkpoint

# ---------------------------------------------------------------------------
# ── Configuration ──────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

NUM_CLASSES: int = 15          # Adjust to match your dataset
NUM_EPOCHS: int = 100
LR: float = 1e-3
WEIGHT_DECAY: float = 1e-4
BATCH_SIZE: int = 16
IMG_SIZE: int = 640
SAVE_EVERY_MINUTES: float = 10.0
MAX_CHECKPOINTS: int = 3
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# Google Drive checkpoint directory (update to your Drive path in Colab)
CHECKPOINT_DIR: str = "/content/drive/MyDrive/agrokd_checkpoints"
# Local fallback when Drive is not mounted (e.g. running locally)
if not os.path.isdir("/content/drive"):
    CHECKPOINT_DIR = os.path.join(_PROJECT_ROOT, "checkpoints")

BEST_MODEL_PATH: str = os.path.join(CHECKPOINT_DIR, "best_model.pt")

# ---------------------------------------------------------------------------
# Google Drive mount (Colab only)
# ---------------------------------------------------------------------------


def _try_mount_drive() -> None:
    """Attempt to mount Google Drive.  Silently skips if not in Colab."""
    try:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive")
        print("✅ Google Drive mounted.")
    except ImportError:
        print("ℹ️  Not running in Colab — skipping Drive mount.")
    except Exception as exc:
        print(f"⚠️  Could not mount Drive: {exc}")


# ---------------------------------------------------------------------------
# Placeholder DataLoader (replace with your real dataset)
# ---------------------------------------------------------------------------


def _make_dataloaders(batch_size: int, img_size: int):
    """
    Return (train_loader, val_loader) using random data as a placeholder.

    Replace with your actual dataset loading logic.
    """
    from torch.utils.data import DataLoader, TensorDataset

    print("⚠️  Using random placeholder data — replace with real dataset loaders.")
    n_train, n_val = 256, 64
    X_train = torch.randn(n_train, 3, img_size, img_size)
    y_train = torch.randint(0, NUM_CLASSES, (n_train,))
    X_val = torch.randn(n_val, 3, img_size, img_size)
    y_val = torch.randint(0, NUM_CLASSES, (n_val,))

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Simple per-epoch train / eval
# ---------------------------------------------------------------------------


def _train_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        # AgroKDNet returns 3 detection outputs; use the medium scale for
        # the placeholder classification loss (replace with real loss).
        _, out_m, _ = model(X)
        # Global-average-pool + flatten → (B, num_classes)
        pooled = out_m.mean(dim=[2, 3])[:, : NUM_CLASSES]
        loss = criterion(pooled, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def _eval_epoch(
    model: nn.Module,
    loader,
    device: str,
) -> float:
    """Return a mock mAP (accuracy) as proxy for real mAP."""
    model.eval()
    correct = 0
    total = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        _, out_m, _ = model(X)
        pooled = out_m.mean(dim=[2, 3])[:, :NUM_CLASSES]
        preds = pooled.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += len(y)
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def main() -> None:
    _try_mount_drive()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"🔧 Device       : {DEVICE}")
    print(f"🔧 Checkpoint dir: {CHECKPOINT_DIR}")

    # ── Model & optimiser ──────────────────────────────────────────────────
    model = AgroKDNet(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion = nn.CrossEntropyLoss()

    # ── Resume ─────────────────────────────────────────────────────────────
    start_epoch, best_map = resume_from_checkpoint(
        checkpoint_dir=CHECKPOINT_DIR,
        model=model,
        optimizer=optimizer,
        device=DEVICE,
    )

    # ── Data ───────────────────────────────────────────────────────────────
    train_loader, val_loader = _make_dataloaders(BATCH_SIZE, IMG_SIZE)

    # ── Auto-checkpoint saver ──────────────────────────────────────────────
    saver = AutoCheckpointSaver(
        model=model,
        optimizer=optimizer,
        checkpoint_dir=CHECKPOINT_DIR,
        save_interval_minutes=SAVE_EVERY_MINUTES,
        max_checkpoints=MAX_CHECKPOINTS,
    )

    # ── Training loop ──────────────────────────────────────────────────────
    print(f"\n🚀 Starting training from epoch {start_epoch} / {NUM_EPOCHS}")
    try:
        for epoch in range(start_epoch, NUM_EPOCHS):
            t0 = time.time()

            train_loss = _train_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_map = _eval_epoch(model, val_loader, DEVICE)
            scheduler.step()

            current_lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - t0

            print(
                f"Epoch {epoch + 1:4d}/{NUM_EPOCHS} | "
                f"loss={train_loss:.4f} | "
                f"mAP={val_map:.4f} | "
                f"lr={current_lr:.2e} | "
                f"time={elapsed:.1f}s"
            )

            # Update saver state at end of each epoch
            saver.update(
                epoch=epoch,
                best_map=best_map,
                loss=train_loss,
                lr=current_lr,
                val_map=val_map,
            )

            # Save best model separately when mAP improves
            if val_map > best_map:
                best_map = val_map
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "best_map": best_map,
                    },
                    BEST_MODEL_PATH,
                )
                print(f"  🏆 New best mAP={best_map:.4f} — saved to {BEST_MODEL_PATH}")

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user.")
    finally:
        saver.stop()

    print(f"\n🎉 Training complete. Best mAP: {best_map:.4f}")
    print(f"📦 Best model: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
