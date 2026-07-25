# DINOv3 → YOLO11m Knowledge Distillation

Distill DINOv3 ViT-L visual features into YOLO11m for fine-grained remote sensing object detection.

**GPU**: RTX 3090/4090 24GB | **PyTorch**: 2.1+ | **CUDA**: 11.8+

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/zhengzhezhao057/dinov3-yolo-distill.git
cd dinov3-yolo-distill
pip install -r requirements.txt
```

### 2. Download Dependencies

```bash
# DINOv3 source
git clone https://github.com/facebookresearch/dinov3.git /root/autodl-tmp/dinov3_repo

# ViT-L weights (1.2GB)
# Option A: HuggingFace
huggingface-cli download facebook/dinov3-vitl16-pretrain-sat493m --local-dir /root/autodl-tmp
# Option B: HF mirror (China)
wget https://hf-mirror.com/facebook/dinov3-vitl16-pretrain-sat493m/resolve/main/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth -P /root/autodl-tmp

# YOLO11m base model (40MB)
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt -P /root/autodl-tmp
```

### 3. Prepare Dataset

```
/root/autodl-tmp/split_dataset/
├── dataset.yaml          # path: /root/autodl-tmp/split_dataset
├── images/train/         # training images
├── images/val/           # validation images
├── images/test/          # test images
└── labels/train/val/test # YOLO format .txt files
```

`dataset.yaml`:
```yaml
path: /root/autodl-tmp/split_dataset
train: images/train
val: images/val
test: images/test
names: ["HM","LQS","QHS","MS",...]  # 25 classes
nc: 25
```

### 4. Verify Setup

```bash
python scripts/verify.py
```

### 5. Train

```bash
# Stage 1: Teacher Training (100 epochs, ~10h)
python scripts/train_teacher.py

# Stage 2: Extract Teacher Signals (~20min)
python scripts/extract_signals.py

# Stage 3: Distillation (50 epochs, ~4h)
python scripts/train_distill.py

# Stage 4: Fine-tune (10 epochs, ~1h)
python scripts/train_finetune.py
```

> Use `nohup python script.py > log.txt 2>&1 &` for long runs.

## Architecture

```
Teacher (DINOv3 ViT-L)              Student (YOLO11m)
┌──────────────────────┐           ┌─────────────────┐
│ ViT-L/16 (303M)      │           │ YOLO Backbone   │
│ hooks: [5,11,17,23]  │           │      ↓          │
│        ↓             │   MSE     │ YOLO Neck       │
│ ViTBackbone          │←────────→│      ↓          │
│        ↓             │  feature  │ YOLO Head       │
│ LightFPN             │           │      ↓          │
│    ↓         ↓       │   KL      │ Detection Loss  │
│ PredHead  ClsHead    │←────────→│    + distill    │
│    ↓         ↓       │  logits   │                 │
│ Detection  Soft      │           │                 │
│   Loss    Labels     │           │                 │
└──────────────────────┘           └─────────────────┘
```

## Key Features

- **Self-contained**: Zero dependency on ultralytics internals for teacher
- **3-phase freeze**: Stable ViT unfreezing (frozen → partial → full)
- **EMA**: Exponential moving average (decay 0.9995)
- **Multi-scale**: Random input [480-800] for robustness
- **Cosine Warm Restarts**: T_0=20, T_mult=2
- **Progressive distillation**: Feature weight 0.5→0.1, CLS weight 0.3→0.7

## Distillation Loss

```
L = L_det + a(t)*MSE(feat_stu, feat_tch) + b(t)*KL(logits_stu, logits_tch)

a(t): 0.5 → 0.1 (feature alignment)
b(t): 0.3 → 0.7 (classification KD)
KL temperature: T=3.0
```

## Expected Results (25-class remote sensing)

| Metric | YOLO11m | +Distillation |
|--------|---------|---------------|
| mAP50 | 0.965 | 0.972-0.978 |
| mAP50-95 | 0.780 | 0.795-0.808 |

## File Structure

```
dinov3-yolo-distill/
├── scripts/
│   ├── verify.py           # Environment check
│   ├── train_teacher.py    # Stage 1: Teacher training
│   ├── extract_signals.py  # Stage 2: Signal extraction
│   ├── train_distill.py    # Stage 3: Distillation
│   └── train_finetune.py   # Stage 4: Fine-tune
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## License

MIT. See [LICENSE](LICENSE).
