"""
Notebook 02 — Train AgroKD-Net with Auto-Save
==============================================
Paste this script cell-by-cell into a Google Colab notebook, or run it as a
Python script.

Steps covered:
  1. Mount Google Drive
  2. Import AutoCheckpointSaver
  3. Set up model and optimizer
  4. Resume from checkpoint (if available)
  5. Training loop with auto-save
  6. Best model saving

# ── Anti-idle tip ─────────────────────────────────────────────────────────────
# Colab disconnects after ~30-90 min of inactivity in the browser.
# To prevent this, paste the following JavaScript snippet into the browser
# Console (F12 → Console tab) while the Colab tab is open:
#
#   function keepAlive() {
#       document.querySelector("colab-toolbar-button#connect").click();
#   }
#   setInterval(keepAlive, 60000);
#
# The AutoCheckpointSaver in this script also protects you: even if the
# session disconnects, the latest weights are already on Drive.
# ─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim

# ────────────────────────────────────────────────────────────────────────────
# Cell 1: Mount Google Drive
# ────────────────────────────────────────────────────────────────────────────

try:
    from google.colab import drive  # type: ignore
    drive.mount("/content/drive")
    DRIVE_ROOT = "/content/drive/MyDrive/agrokd_project"
    CHECKPOINT_DIR = os.path.join(DRIVE_ROOT, "checkpoints")
    print("✅ Google Drive mounted.")
except ImportError:
    CHECKPOINT_DIR = "/tmp/agrokd_checkpoints"
    print("ℹ️  Not in Colab — saving checkpoints to /tmp")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ────────────────────────────────────────────────────────────────────────────
# Cell 2: Add project root to path and import helpers
# ────────────────────────────────────────────────────────────────────────────

LOCAL_REPO = "/content/agrokd"
if not os.path.isdir(LOCAL_REPO):
    import subprocess
    subprocess.run(
        ["git", "clone",
         "https://github.com/RUSHI-KOLLA/oja-weed_detection-.git",
         LOCAL_REPO],
        check=False,
    )

if LOCAL_REPO not in sys.path:
    sys.path.insert(0, LOCAL_REPO)

from models.agrokd_net import AgroKDNet                                    # noqa: E402
from utils.auto_checkpoint_saver import AutoCheckpointSaver, resume_from_checkpoint  # noqa: E402

print("✅ Imports successful.")


# ────────────────────────────────────────────────────────────────────────────
# Cell 3: Configuration
# ────────────────────────────────────────────────────────────────────────────

NUM_CLASSES         = 15
NUM_EPOCHS          = 100
LR                  = 1e-3
WEIGHT_DECAY        = 1e-4
BATCH_SIZE          = 16
IMG_SIZE            = 640
SAVE_EVERY_MINUTES  = 10.0
MAX_CHECKPOINTS     = 3
DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"
BEST_MODEL_PATH     = os.path.join(CHECKPOINT_DIR, "best_model.pt")

print(f"🔧 Device         : {DEVICE}")
print(f"🔧 Checkpoint dir : {CHECKPOINT_DIR}")
print(f"🔧 Save every     : {SAVE_EVERY_MINUTES} min")


# ────────────────────────────────────────────────────────────────────────────
# Cell 4: Build model and optimizer
# ────────────────────────────────────────────────────────────────────────────

model = AgroKDNet(num_classes=NUM_CLASSES).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
criterion = nn.CrossEntropyLoss()

total_params = sum(p.numel() for p in model.parameters())
print(f"📊 Total parameters: {total_params:,}")


# ────────────────────────────────────────────────────────────────────────────
# Cell 5: Resume from checkpoint
# ────────────────────────────────────────────────────────────────────────────

start_epoch, best_map = resume_from_checkpoint(
    checkpoint_dir=CHECKPOINT_DIR,
    model=model,
    optimizer=optimizer,
    device=DEVICE,
)

# Fast-forward scheduler to match resumed epoch
for _ in range(start_epoch):
    scheduler.step()


# ────────────────────────────────────────────────────────────────────────────
# Cell 6: Placeholder DataLoader (replace with real dataset)
# ────────────────────────────────────────────────────────────────────────────

from torch.utils.data import DataLoader, TensorDataset

print("⚠️  Using random placeholder data — replace with your real DataLoaders.")
X_train = torch.randn(256, 3, IMG_SIZE, IMG_SIZE)
y_train = torch.randint(0, NUM_CLASSES, (256,))
X_val   = torch.randn(64, 3, IMG_SIZE, IMG_SIZE)
y_val   = torch.randint(0, NUM_CLASSES, (64,))

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val, y_val),   batch_size=BATCH_SIZE)


# ────────────────────────────────────────────────────────────────────────────
# Cell 7: Start AutoCheckpointSaver
# ────────────────────────────────────────────────────────────────────────────

saver = AutoCheckpointSaver(
    model=model,
    optimizer=optimizer,
    checkpoint_dir=CHECKPOINT_DIR,
    save_interval_minutes=SAVE_EVERY_MINUTES,
    max_checkpoints=MAX_CHECKPOINTS,
)


# ────────────────────────────────────────────────────────────────────────────
# Cell 8: Training loop
# ────────────────────────────────────────────────────────────────────────────

print(f"\n🚀 Training from epoch {start_epoch} to {NUM_EPOCHS}")

try:
    for epoch in range(start_epoch, NUM_EPOCHS):
        t0 = time.time()

        # ── Train one epoch ────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            _, out_m, _ = model(X)
            pooled = out_m.mean(dim=[2, 3])[:, :NUM_CLASSES]
            loss = criterion(pooled, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)
        train_loss /= len(train_loader.dataset)

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        correct = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                _, out_m, _ = model(X)
                pooled = out_m.mean(dim=[2, 3])[:, :NUM_CLASSES]
                correct += (pooled.argmax(1) == y).sum().item()
        val_map = correct / len(val_loader.dataset)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch + 1:4d}/{NUM_EPOCHS} | "
            f"loss={train_loss:.4f} | mAP={val_map:.4f} | "
            f"lr={current_lr:.2e} | time={elapsed:.1f}s"
        )

        # Update checkpoint state
        saver.update(
            epoch=epoch,
            best_map=best_map,
            loss=train_loss,
            lr=current_lr,
            val_map=val_map,
        )

        # Save best model
        if val_map > best_map:
            best_map = val_map
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "best_map": best_map},
                BEST_MODEL_PATH,
            )
            print(f"  🏆 New best mAP={best_map:.4f} — saved to {BEST_MODEL_PATH}")

except KeyboardInterrupt:
    print("\n⚠️  Interrupted — performing emergency checkpoint save …")
finally:
    saver.stop()

print(f"\n🎉 Done! Best mAP: {best_map:.4f}")
print(f"📦 Best model saved to: {BEST_MODEL_PATH}")
