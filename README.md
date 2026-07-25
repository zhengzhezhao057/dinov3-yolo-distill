# DINOv3 → YOLO11m Knowledge Distillation

Distill DINOv3 ViT-L into YOLO11m for fine-grained remote sensing object detection (25 classes).

**GPU**: RTX 3090/4090 24GB | **PyTorch**: 2.1+ | **Python**: 3.10+

## Quick Start

```bash
git clone https://github.com/zhengzhezhao057/dinov3-yolo-distill.git
cd dinov3-yolo-distill
bash setup.sh
```

`setup.sh` will:
- Install Python packages
- Clone DINOv3 repository
- Download ViT-L weights (1.2GB) + YOLO11m (40MB)

## Prepare Dataset

```
data/
├── dataset.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

`dataset.yaml`:
```yaml
path: ./data          # relative to DISTILL_HOME, or use absolute path
train: images/train
val: images/val
test: images/test
names: ["HM","LQS","QHS","MS","A1_SU-35","A2_C-130",...]
nc: 25
```

## Verify

```bash
python scripts/verify.py
```

## Train

```bash
python scripts/train_teacher.py    # Stage 1: Teacher (100 epochs, ~10h)
python scripts/extract_signals.py  # Stage 2: Signals (~20min)
python scripts/train_distill.py    # Stage 3: Distill (50 epochs, ~4h)
python scripts/train_finetune.py   # Stage 4: Fine-tune (10 epochs, ~1h)
```

Use `nohup python script.py > log.txt 2>&1 &` for long runs.

## Configuration

All paths in `config.py`. Set `DISTILL_HOME` env var to change root directory:

```bash
export DISTILL_HOME=/your/path
```

Default: the repo directory itself.

## Architecture

```
Teacher (DINOv3 ViT-L 303M)         Student (YOLO11m 20M)
┌──────────────────────────┐       ┌──────────────────┐
│ ViT-L/16                 │       │ YOLO Backbone    │
│ hooks: [5,11,17,23]      │       │       ↓          │
│        ↓                 │ MSE   │ YOLO Neck        │
│ ViTBackbone → LightFPN   │←─────→│       ↓          │
│    ↓              ↓      │  KL   │ YOLO Head        │
│ PredHead      ClsHead    │←─────→│       ↓          │
│ (detection)  (soft label)│       │ L_det + distill  │
└──────────────────────────┘       └──────────────────┘
```

## Distillation

```
L = L_det + a(t)*MSE(feat_stu, feat_tch) + b(t)*KL(logits_stu, logits_tch)

a(t): 0.5 → 0.1  (feature alignment, early focus)
b(t): 0.3 → 0.7  (classification KD, late focus)
KL temperature: T=3.0
```

## Key Features

- Self-contained: zero ultralytics internals for teacher
- 3-phase ViT freeze/unfreeze schedule
- EMA (decay 0.9995), multi-scale [480-800]
- Cosine Warm Restarts, label smoothing
- Progressive distillation weights

## License

MIT.
