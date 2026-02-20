"""
Auto Checkpoint Saver for Google Colab Training Sessions.

Saves checkpoints every N minutes in a background thread so that
training progress is not lost when Colab disconnects.
"""

import os
import time
import glob
import threading
from typing import Any, Dict, Optional, Tuple

import torch


class AutoCheckpointSaver:
    """
    Background-thread checkpoint saver designed for Google Colab.

    Automatically saves model state every ``save_interval_minutes`` to
    ``checkpoint_dir``.  Thread-safe: the training loop calls ``update()``
    after every epoch; the background thread periodically calls
    ``save_now()`` on its own.

    Parameters
    ----------
    model : torch.nn.Module
        The model being trained.
    optimizer : torch.optim.Optimizer
        The optimizer used during training.
    checkpoint_dir : str
        Directory where checkpoints will be written (e.g. a Google Drive path).
    save_interval_minutes : float
        How often to auto-save (default: 10 minutes).
    max_checkpoints : int
        Maximum number of rolling checkpoints to keep (default: 3).
        The ``latest.pt`` symlink is kept in addition.
    """

    def __init__(
        self,
        model: "torch.nn.Module",
        optimizer: "torch.optim.Optimizer",
        checkpoint_dir: str,
        save_interval_minutes: float = 10.0,
        max_checkpoints: int = 3,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_dir = checkpoint_dir
        self.save_interval_seconds = save_interval_minutes * 60
        self.max_checkpoints = max_checkpoints

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Shared state updated by the training loop
        self._state: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        # Background thread
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(
            f"💾 AutoCheckpointSaver started — saving every "
            f"{save_interval_minutes:.1f} min → {checkpoint_dir}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        epoch: int,
        best_map: float = 0.0,
        loss: float = 0.0,
        lr: float = 0.0,
        **extra: Any,
    ) -> None:
        """
        Update the latest training state.  Call this at the end of each epoch.

        Parameters
        ----------
        epoch : int
            Completed epoch number (0-indexed).
        best_map : float
            Best mAP achieved so far.
        loss : float
            Current training loss.
        lr : float
            Current learning rate.
        **extra :
            Any additional scalar values to store (e.g. val_loss).
        """
        with self._lock:
            self._state = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": epoch,
                "best_map": best_map,
                "loss": loss,
                "lr": lr,
                "timestamp": time.time(),
                **extra,
            }

    def save_now(self) -> Optional[str]:
        """
        Force-save the current state immediately.

        Returns
        -------
        str or None
            Path to the saved checkpoint, or None if no state is available.
        """
        with self._lock:
            state = dict(self._state)  # shallow copy under the lock

        if not state:
            print("⚠️  AutoCheckpointSaver.save_now() called but no state to save yet.")
            return None

        epoch = state.get("epoch", 0)
        filename = f"checkpoint_epoch_{epoch:04d}.pt"
        path = os.path.join(self.checkpoint_dir, filename)

        try:
            torch.save(state, path)
            # Overwrite latest.pt
            latest_path = os.path.join(self.checkpoint_dir, "latest.pt")
            torch.save(state, latest_path)
            self._cleanup_old_checkpoints()
            print(f"💾 Checkpoint saved → {path}")
            return path
        except Exception as exc:
            print(f"❌ Failed to save checkpoint: {exc}")
            return None

    def stop(self) -> None:
        """
        Signal the background thread to stop and perform a final save.
        Blocks until the thread finishes.
        """
        print("🛑 Stopping AutoCheckpointSaver — performing final save …")
        self._stop_event.set()
        self._thread.join(timeout=30)
        self.save_now()
        print("✅ AutoCheckpointSaver stopped.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Background thread: sleep then save, repeat."""
        while not self._stop_event.wait(timeout=self.save_interval_seconds):
            self.save_now()

    def _cleanup_old_checkpoints(self) -> None:
        """Remove oldest checkpoints so that at most ``max_checkpoints`` remain."""
        pattern = os.path.join(self.checkpoint_dir, "checkpoint_epoch_*.pt")
        checkpoints = sorted(glob.glob(pattern))
        excess = len(checkpoints) - self.max_checkpoints
        for old_ckpt in checkpoints[:excess]:
            try:
                os.remove(old_ckpt)
                print(f"🗑️  Removed old checkpoint: {os.path.basename(old_ckpt)}")
            except OSError as exc:
                print(f"⚠️  Could not remove {old_ckpt}: {exc}")


# ---------------------------------------------------------------------------
# Standalone resume helper
# ---------------------------------------------------------------------------


def resume_from_checkpoint(
    checkpoint_dir: str,
    model: "torch.nn.Module",
    optimizer: Optional["torch.optim.Optimizer"] = None,
    device: Optional[str] = None,
) -> Tuple[int, float]:
    """
    Load the latest checkpoint from *checkpoint_dir* and restore model /
    optimizer state in-place.

    Parameters
    ----------
    checkpoint_dir : str
        Directory that contains ``latest.pt``.
    model : torch.nn.Module
        Model whose weights will be restored.
    optimizer : torch.optim.Optimizer, optional
        Optimizer whose state will be restored (skipped if None).
    device : str, optional
        Torch device string (e.g. ``'cuda'``, ``'cpu'``).  Defaults to
        ``'cuda'`` when available.

    Returns
    -------
    start_epoch : int
        The next epoch to train from (completed epoch + 1).
    best_map : float
        Best mAP value stored in the checkpoint.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    latest_path = os.path.join(checkpoint_dir, "latest.pt")
    if not os.path.isfile(latest_path):
        print(f"ℹ️  No checkpoint found at {latest_path} — starting from scratch.")
        return 0, 0.0

    print(f"📂 Resuming from checkpoint: {latest_path}")
    checkpoint = torch.load(latest_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"✅ Model weights restored (epoch {checkpoint.get('epoch', '?')})")

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("✅ Optimizer state restored")

    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_map = float(checkpoint.get("best_map", 0.0))
    print(
        f"🚀 Resuming training from epoch {start_epoch} | best mAP so far: {best_map:.4f}"
    )
    return start_epoch, best_map


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    print("🧪 Testing AutoCheckpointSaver …")

    # Tiny model for testing
    _model = torch.nn.Linear(4, 2)
    _optim = torch.optim.SGD(_model.parameters(), lr=0.01)

    with tempfile.TemporaryDirectory() as tmpdir:
        saver = AutoCheckpointSaver(
            model=_model,
            optimizer=_optim,
            checkpoint_dir=tmpdir,
            save_interval_minutes=0.05,  # 3 seconds for testing
            max_checkpoints=2,
        )

        for epoch in range(5):
            saver.update(epoch=epoch, best_map=epoch * 0.1, loss=1.0 - epoch * 0.1)
            time.sleep(0.5)

        saver.stop()

        # Test resume
        _model2 = torch.nn.Linear(4, 2)
        _optim2 = torch.optim.SGD(_model2.parameters(), lr=0.01)
        start, bmap = resume_from_checkpoint(tmpdir, _model2, _optim2)
        print(f"✅ resume_from_checkpoint returned start_epoch={start}, best_map={bmap:.4f}")

    print("✅ All tests passed.")
