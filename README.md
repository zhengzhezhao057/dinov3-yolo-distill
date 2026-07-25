# DINOv3 → YOLO11m Knowledge Distillation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Distill DINOv3 ViT-L's remote sensing feature representations into YOLO11m for fine-grained aircraft/vehicle detection (25 classes).

## Overview

```
DINOv3 ViT-L (Teacher)          YOLO11m (Student)
      │                              │
  ViTBackbone                    YOLO Backbone
      │                              │
  LightFPN Neck          ←MSE→   YOLO Neck
      │                              │
  PredHead + ClsHead      ←KD→   YOLO Head
      │                              │
  Detection Loss          + distill loss
```

**3 training stages:**

| Stage | Script | Epochs | Output |
|-------|--------|--------|--------|
| Teacher Training | `train_teacher.py` | 100 | Teacher model weights |
| Distillation | `train_distill.py` | 50 | Distilled YOLO11m |
| Fine-tune | `train_finetune.py` | 10 | Final model |

## Quick Start

### Prerequisites

- GPU: RTX 3090/4090 with 24GB VRAM
- PyTorch 2.1+, CUDA 11.8+
- Ubuntu 22.04 (recommended)

### Installation

```bash
# 1. Clone this repo
git clone https://github.com/YOUR_USERNAME/dinov3-yolo-distill.git
cd dinov3-yolo-distill

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download DINOv3 weights (~1.2GB)
# Option A: from HuggingFace
huggingface-cli download facebook/dinov3-vitl16-pretrain-sat493m --local-dir ./weights

# Option B: from HF mirror (China)
wget https://hf-mirror.com/facebook/dinov3-vitl16-pretrain-lvd1689m/resolve/main/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth -P ./weights

# 4. Download DINOv3 source
git clone https://github.com/facebookresearch/dinov3.git

# 5. Download YOLO11m
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt
```

### Dataset Preparation

Organize your dataset in YOLO format:

```
data/
├── dataset.yaml
├── images/
│   ├── train/    # training images
│   ├── val/      # validation images
│   └── test/     # test images
└── labels/
    ├── train/    # YOLO format labels (.txt)
    ├── val/
    └── test/
```

`dataset.yaml`:
```yaml
path: /path/to/data
train: images/train
val: images/val
test: images/test
names: ["class_0", "class_1", ...]
nc: 25
```

### Training

```bash
# Stage 1: Train teacher (100 epochs, ~10 hours)
python scripts/train_teacher.py

# Stage 2: Distill to YOLO11m (50 epochs, ~4 hours)
python scripts/train_distill.py

# Stage 3: Fine-tune (10 epochs, ~1 hour)
python scripts/train_finetune.py
```

## Architecture

### Teacher Model
```
DINOv3 ViT-L/16 (303M, frozen in early stages)
  → ViTBackbone (multi-scale projection)
  → LightFPN (top-down fusion)
  → PredHead (detection) + ClassifierHead (25-class logits)
```

### Key Innovations
- **Self-contained**: Zero dependency on ultralytics internals for teacher training
- **3-phase freeze schedule**: Gradual ViT unfreezing for stable convergence
- **EMA**: Exponential moving average (decay=0.9995) for all stages
- **Multi-scale training**: Random input size [480-800] for robustness
- **Cosine Warm Restarts**: T₀=20, T_mult=2 for better exploration
- **Progressive distillation weights**: Feature (0.5→0.1), Classification (0.3→0.7)

### Distillation Loss
```
L_total = L_det + α(t)·L_feat_mse + β(t)·L_cls_kd + γ·L_box_smoothl1

α(t): 0.5 → 0.1  (feature alignment, dominant early)
β(t): 0.3 → 0.7  (classification KD, dominant late)
γ: 0.2 (constant)
```

## Expected Results

| Metric | Baseline (YOLO11m) | After Distillation |
|--------|-------------------|--------------------|
| mAP50 | 0.965 | 0.972-0.978 |
| mAP50-95 | 0.780 | 0.795-0.808 |
| Small objects (FSC) | 0.525 | 0.55-0.60 |

## Citation

```bibtex
@misc{dinov3-yolo-distill,
  author = {YOUR_NAME},
  title = {DINOv3 → YOLO11m Knowledge Distillation for Remote Sensing},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/YOUR_USERNAME/dinov3-yolo-distill}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- [DINOv3](https://github.com/facebookresearch/dinov3) - Meta AI
- [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics)
- DINOv3 weights: `facebook/dinov3-vitl16-pretrain-sat493m`
