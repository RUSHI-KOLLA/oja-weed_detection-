"""
Google Drive Syncer for Ultralytics YOLO training on Google Colab.

Periodically copies YOLO training outputs (weights, results CSV) from the
local Colab filesystem to a Google Drive folder so that progress is preserved
across disconnects.
"""

import os
import shutil
import time
import threading
from typing import Optional


class DriveSyncer:
    """
    Background-thread syncer that copies YOLO run outputs to Google Drive.

    Parameters
    ----------
    local_run_dir : str
        Path to the YOLO run directory on local Colab disk
        (e.g. ``runs/detect/train``).
    drive_save_dir : str
        Destination directory on Google Drive
        (e.g. ``/content/drive/MyDrive/yolo_runs/train``).
    sync_interval_minutes : float
        How often to sync (default: 10 minutes).
    """

    def __init__(
        self,
        local_run_dir: str,
        drive_save_dir: str,
        sync_interval_minutes: float = 10.0,
    ) -> None:
        self.local_run_dir = local_run_dir
        self.drive_save_dir = drive_save_dir
        self.sync_interval_seconds = sync_interval_minutes * 60

        os.makedirs(drive_save_dir, exist_ok=True)

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(
            f"☁️  DriveSyncer started — syncing every "
            f"{sync_interval_minutes:.1f} min\n"
            f"   {local_run_dir} → {drive_save_dir}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Stop the background sync thread and perform a final sync."""
        print("🛑 Stopping DriveSyncer — performing final sync …")
        self._stop_event.set()
        self._thread.join(timeout=60)
        self._sync()
        print("✅ DriveSyncer stopped.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Background thread loop."""
        while not self._stop_event.wait(timeout=self.sync_interval_seconds):
            self._sync()

    def _sync(self) -> None:
        """Copy weight files and results.csv to Google Drive."""
        if not os.path.isdir(self.local_run_dir):
            print(f"⚠️  DriveSyncer: local run dir not found yet: {self.local_run_dir}")
            return

        copied = 0
        errors = 0

        # Sync weights/ sub-directory
        weights_src = os.path.join(self.local_run_dir, "weights")
        weights_dst = os.path.join(self.drive_save_dir, "weights")
        if os.path.isdir(weights_src):
            os.makedirs(weights_dst, exist_ok=True)
            for wf in os.listdir(weights_src):
                if wf.endswith(".pt"):
                    src = os.path.join(weights_src, wf)
                    dst = os.path.join(weights_dst, wf)
                    try:
                        shutil.copy2(src, dst)
                        copied += 1
                    except Exception as exc:
                        print(f"❌ Failed to copy {wf}: {exc}")
                        errors += 1

        # Sync results.csv
        results_src = os.path.join(self.local_run_dir, "results.csv")
        if os.path.isfile(results_src):
            try:
                shutil.copy2(results_src, os.path.join(self.drive_save_dir, "results.csv"))
                copied += 1
            except Exception as exc:
                print(f"❌ Failed to copy results.csv: {exc}")
                errors += 1

        status = "✅" if errors == 0 else "⚠️"
        print(
            f"{status} DriveSyncer synced {copied} file(s) to Drive"
            + (f" ({errors} error(s))" if errors else "")
        )


# ---------------------------------------------------------------------------
# High-level helper: train YOLO with Drive auto-save
# ---------------------------------------------------------------------------


def train_yolo_with_drive_save(
    model_name: str,
    data_yaml: str,
    drive_save_dir: str,
    epochs: int = 100,
    img_size: int = 640,
    batch_size: int = 16,
    save_period: int = 5,
    local_project: str = "runs/detect",
    run_name: str = "train",
    sync_interval_minutes: float = 10.0,
    extra_train_kwargs: Optional[dict] = None,
) -> None:
    """
    Train a YOLO model and continuously sync outputs to Google Drive.

    Checks Google Drive for a previous ``best.pt`` or ``last.pt`` weight and
    resumes from it when available.

    Parameters
    ----------
    model_name : str
        Ultralytics model identifier, e.g. ``'yolov8n.pt'``.
    data_yaml : str
        Path to the dataset YAML file.
    drive_save_dir : str
        Google Drive directory where weights/results are mirrored.
    epochs : int
        Total number of training epochs.
    img_size : int
        Input image size.
    batch_size : int
        Training batch size.
    save_period : int
        Save a YOLO checkpoint every N epochs (Ultralytics built-in).
    local_project : str
        Root of the local YOLO runs directory.
    run_name : str
        Name of this run.
    sync_interval_minutes : float
        Background sync frequency (minutes).
    extra_train_kwargs : dict, optional
        Additional keyword arguments forwarded to ``model.train()``.
    """
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "ultralytics is required for train_yolo_with_drive_save. "
            "Install it with: pip install ultralytics"
        ) from exc

    if extra_train_kwargs is None:
        extra_train_kwargs = {}

    # Check for existing weights on Drive to resume from
    resume_weights: Optional[str] = None
    for candidate in ("best.pt", "last.pt"):
        drive_weights_path = os.path.join(drive_save_dir, "weights", candidate)
        if os.path.isfile(drive_weights_path):
            resume_weights = drive_weights_path
            print(f"📂 Found previous weights on Drive — resuming from {resume_weights}")
            break

    # Load model (resume from Drive weights if found)
    if resume_weights:
        model = YOLO(resume_weights)
    else:
        model = YOLO(model_name)
        print(f"🚀 Starting fresh training with {model_name}")

    # Local run output directory
    local_run_dir = os.path.join(local_project, run_name)

    # Start background Drive syncer
    syncer = DriveSyncer(
        local_run_dir=local_run_dir,
        drive_save_dir=drive_save_dir,
        sync_interval_minutes=sync_interval_minutes,
    )

    try:
        model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            save_period=save_period,
            project=local_project,
            name=run_name,
            **extra_train_kwargs,
        )
        print("🎉 Training complete!")
    finally:
        syncer.stop()

    # Final: copy all output to Drive
    if os.path.isdir(local_run_dir):
        print(f"📦 Copying full run directory to Drive: {drive_save_dir}")
        try:
            if os.path.exists(drive_save_dir):
                shutil.rmtree(drive_save_dir)
            shutil.copytree(local_run_dir, drive_save_dir)
            print("✅ All results saved to Google Drive.")
        except Exception as exc:
            print(f"❌ Could not copy run dir to Drive: {exc}")


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    print("🧪 Testing DriveSyncer …")

    with tempfile.TemporaryDirectory() as tmpdir:
        local_run = os.path.join(tmpdir, "runs", "detect", "train")
        drive_dir = os.path.join(tmpdir, "drive_backup")

        # Simulate YOLO creating output files
        os.makedirs(os.path.join(local_run, "weights"), exist_ok=True)
        for name in ("best.pt", "last.pt"):
            open(os.path.join(local_run, "weights", name), "wb").close()
        open(os.path.join(local_run, "results.csv"), "w").close()

        syncer = DriveSyncer(
            local_run_dir=local_run,
            drive_save_dir=drive_dir,
            sync_interval_minutes=0.05,  # 3 seconds for testing
        )
        time.sleep(5)
        syncer.stop()

        synced_files = os.listdir(os.path.join(drive_dir, "weights"))
        assert "best.pt" in synced_files, "best.pt not synced!"
        assert "last.pt" in synced_files, "last.pt not synced!"
        print("✅ DriveSyncer test passed.")
