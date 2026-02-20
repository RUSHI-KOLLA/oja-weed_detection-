# 🌾 AgroKD-Net: Energy-Aware Lightweight Crop–Weed Detection

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

> **AgroKD-Net** is a lightweight, energy-aware object detection network for
> crop–weed identification in precision agriculture.  It uses **Knowledge
> Distillation (KD)** from larger teacher models and **Organic Judgment
> Surveillance (OJS)** loss functions to achieve high accuracy at low
> computational cost — ideal for edge-device deployment.

---

## 🌟 Project Vision

Weeds cost farmers billions of dollars per year in lost yield and herbicide
spend.  Autonomous weed identification on low-power field robots requires
models that are *small*, *fast*, and *accurate*.  AgroKD-Net distils
knowledge from heavy teacher detectors into a depthwise-separable backbone
with SE-attention, making it suitable for inference on NVIDIA Jetson-class
hardware or even microcontrollers.

---

## ⚡ Quick Start (3 steps)

```bash
# 1. Clone the repo
git clone https://github.com/RUSHI-KOLLA/oja-weed_detection-.git
cd oja-weed_detection-

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train AgroKD-Net (uses placeholder data; swap in your dataset)
python scripts/train_agrokd.py
```

---

## 📁 Project Structure

```
oja-weed_detection-/
├── models/
│   ├── agrokd_net.py          # AgroKD-Net architecture
│   ├── ojs_loss.py            # OJS loss functions (GBPR, AIPL, EAKD, SCD)
│   └── __init__.py
├── utils/
│   ├── auto_checkpoint_saver.py  # ⭐ Auto-save system for Colab
│   ├── drive_syncer.py           # Google Drive sync for YOLO runs
│   └── __init__.py
├── scripts/
│   ├── train_agrokd.py        # Full AgroKD-Net training script
│   ├── benchmark_models.py    # Compare 10 detection models
│   └── __init__.py
├── configs/
│   ├── default_config.yaml    # Default hyper-parameters
│   ├── dataset_configs/
│   │   ├── deepweeds.yaml
│   │   ├── cottonweed.yaml
│   │   └── riceweed.yaml
│   └── __init__.py
├── notebooks/
│   ├── 01_setup_and_explore.py   # Colab setup & data exploration
│   └── 02_train_with_autosave.py # Colab training with auto-save
├── results/                   # Training outputs (gitignored)
├── requirements.txt
└── README.md
```

---

## 💾 Auto-Save Checkpoint System

### Why it matters

Google Colab disconnects after **30–90 minutes** of browser inactivity.
Without checkpointing, all training progress is lost.

### How it works

`utils/auto_checkpoint_saver.py` runs a **background thread** that
saves the current model state every 10 minutes to Google Drive while
training continues uninterrupted in the main thread.

```python
from utils.auto_checkpoint_saver import AutoCheckpointSaver, resume_from_checkpoint

# 1. Resume from Drive if a previous session was interrupted
start_epoch, best_map = resume_from_checkpoint(
    checkpoint_dir="/content/drive/MyDrive/checkpoints",
    model=model,
    optimizer=optimizer,
)

# 2. Start auto-saver (background thread, saves every 10 min)
saver = AutoCheckpointSaver(
    model=model,
    optimizer=optimizer,
    checkpoint_dir="/content/drive/MyDrive/checkpoints",
    save_interval_minutes=10,
    max_checkpoints=3,   # keeps Drive tidy
)

# 3. Inside training loop — update state after each epoch
saver.update(epoch=epoch, best_map=best_map, loss=loss, lr=lr)

# 4. Clean shutdown with final save
saver.stop()
```

### What gets saved
| Field | Description |
|---|---|
| `model_state_dict` | Full model weights |
| `optimizer_state_dict` | Optimizer momentum/state |
| `epoch` | Last completed epoch |
| `best_map` | Best mAP achieved |
| `loss` | Latest training loss |
| `lr` | Current learning rate |
| `timestamp` | Unix timestamp |

---

## 🏋️ Training AgroKD-Net

```bash
python scripts/train_agrokd.py
```

Edit the `# ── Configuration ──` block at the top of the script to set:
- `NUM_CLASSES`, `NUM_EPOCHS`, `LR`, `BATCH_SIZE`
- `CHECKPOINT_DIR` (your Google Drive path)
- Replace the placeholder DataLoaders with your real dataset

---

## 📊 Benchmarking 10 Models

```bash
python scripts/benchmark_models.py
```

Models compared:

| # | Model | Type |
|---|---|---|
| 1 | YOLOv8n | Nano |
| 2 | YOLOv8s | Small |
| 3 | YOLOv8m | Medium |
| 4 | YOLOv9t | Tiny |
| 5 | YOLOv9s | Small |
| 6 | YOLOv10n | Nano |
| 7 | YOLOv10s | Small |
| 8 | YOLOv11n | Nano |
| 9 | RT-DETR-L | Transformer |
| 10 | YOLOv8n-World | Open-vocab |

Results are saved to `results/benchmarks/benchmark_results.csv`.

---

## 🌿 Datasets

| Dataset | Task | Classes | Images |
|---|---|---|---|
| [DeepWeeds](https://github.com/AlexOlsen/DeepWeeds) | Classification | 8 weeds + negative | ~17 509 |
| [CottonWeed](https://zenodo.org/record/7535814) | Detection | 15 weeds + cotton | ~7 578 |
| [RiceWeed](https://universe.roboflow.com) | Detection | rice + weeds | ~3,000 |

Update `configs/dataset_configs/*.yaml` with your local paths.

---

## 📈 Expected Results (template)

| Model | Params (M) | GFLOPs | mAP50 | mAP50-95 | FPS |
|---|---|---|---|---|---|
| YOLOv8n | — | — | — | — | — |
| AgroKD-Net (ours) | — | — | — | — | — |
| ... | | | | | |

*(Fill in after running benchmarks)*

---

## 🧠 Model Architecture

```
Input (3×640×640)
    │
    ▼
AgroKDBackbone
  Stage 1: Conv 3→32   (stride 2)
  Stage 2: DSConv 32→64  (stride 2)
  Stage 3: DSConv 64→128  + SE  (stride 2)  ──→ f3
  Stage 4: DSConv 128→256 + SE  (stride 2)  ──→ f4
  Stage 5: DSConv 256→512 + SE  (stride 2)  ──→ f5
    │
    ▼
LightFPN  (lateral 1×1 + top-down upsample + DSConv)
  p3 (stride-8)
  p4 (stride-16)
  p5 (stride-32)
    │
    ▼
Detection Heads  (DSConv → Conv1×1 → 5+C channels each)
```

---

## 🔬 OJS Loss Functions

| Component | Full Name | Purpose |
|---|---|---|
| GBPR | Gradient-Balanced Per-class Reweighting | Inverse gradient-mag weights |
| AIPL | Adaptive Imbalance-aware Progressive Loss | Focal + inverse-frequency |
| EAKD | Energy-Aware Knowledge Distillation | KL-div KD + energy penalty |
| SCD  | Structural Context Distillation | Feature affinity matrix KD |

---

## 🛡️ Anti-Idle Tip for Colab

Paste this in the browser **Console** (F12) while Colab is open to prevent
the session from going idle:

```javascript
function keepAlive() {
    document.querySelector("colab-toolbar-button#connect").click();
}
setInterval(keepAlive, 60000);
```

The `AutoCheckpointSaver` still protects you: even if the session drops,
your latest weights are already on Google Drive.

---

## ✅ Current Progress

- [x] Project folder structure
- [x] `utils/auto_checkpoint_saver.py` — background thread auto-save
- [x] `utils/drive_syncer.py` — YOLO Drive sync
- [x] `models/agrokd_net.py` — full AgroKD-Net model
- [x] `models/ojs_loss.py` — OJS loss functions
- [x] `scripts/train_agrokd.py` — training script
- [x] `scripts/benchmark_models.py` — 10-model benchmark
- [x] `configs/` — YAML configs
- [x] `notebooks/` — Colab-ready scripts
- [x] `requirements.txt`
- [ ] Real dataset integration
- [ ] Teacher model training
- [ ] Full KD training pipeline
- [ ] Benchmark results table

---

## 📝 Citation

```bibtex
@misc{agrokdnet2024,
  title   = {AgroKD-Net: Energy-Aware Lightweight Crop–Weed Detection
             via Knowledge Distillation and Organic Judgment Surveillance},
  author  = {Rushi Kolla},
  year    = {2024},
  url     = {https://github.com/RUSHI-KOLLA/oja-weed_detection-}
}
```

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.
