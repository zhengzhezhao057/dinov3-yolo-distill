# DINOv3 -> YOLO11m Knowledge Distillation (25-Class Remote Sensing)

Distill DINOv3 ViT-L into YOLO11m for fine-grained military aircraft detection.

**GPU**: RTX 3090/4090 24GB | **PyTorch**: 2.1+ | **Python**: 3.10+

---

## Requirements

| Item | Spec |
|------|------|
| GPU | RTX 3090/4090 24GB |
| PyTorch | 2.1+ (CUDA 11.8/12.x) |
| Python | 3.10+ |
| OS | Ubuntu 20.04/22.04 |
| Disk | ~10GB (with dataset & weights) |

---

## Quick Start

### 1. Clone

```bash
git clone https://github.com/zhengzhezhao057/dinov3-yolo-distill.git
cd dinov3-yolo-distill
```

### 2. Install Dependencies

```bash
pip install ultralytics opencv-python-headless tqdm pyyaml numpy timm -q
git clone --depth 1 https://github.com/facebookresearch/dinov3.git dinov3_repo
```

### 3. Download Weights

```bash
mkdir -p weights

# YOLO11m (40MB)
wget -O weights/yolo11m.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt

# DINOv3 ViT-L (1.2GB) - requires HuggingFace login + access approval
pip install huggingface_hub -q
hf download facebook/dinov3-vitl16-pretrain-sat493m --local-dir weights/
```

> If HuggingFace fails, upload manually: `scp -P <port> <local.pth> root@<ip>:/root/dinov3-yolo-distill/weights/`

### 4. Prepare Dataset

Directory structure:
```
data/
  dataset.yaml
  images/train/    (*.jpg / *.png)
  labels/train/    (YOLO format *.txt)
```

**dataset.yaml**:
```yaml
path: ./data
train: images/train
nc: 25
names: ["HM","LQS","QHS","MS","A1_SU-35","A2_C-130","A3_C-17","A4_C-5",
        "A5_F-16","A6_TU-160","A7_E-3","A8_B-52","A9_P-3C","A10_B-1B",
        "A11_E-8","A12_TU-22","A13_F-15","A14_KC-135","A15_F-22",
        "A16_FA-18","A17_TU-95","A18_KC-10","A19_SU-34","A20_SU-24","FSC"]
```

Upload (compress then SCP):
```powershell
# Local: compress dataset
cd your_dataset_folder
tar -czf dataset.tar.gz images labels dataset.yaml

# Upload to server
scp -P <port> dataset.tar.gz root@<ip>:/root/dinov3-yolo-distill/data/

# Server: extract
cd /root/dinov3-yolo-distill/data && tar -xzf dataset.tar.gz
```

### 5. Verify

```bash
python scripts/verify.py
```

Expected: `ALL OK!`

---

## Training Pipeline

| Step | Command | Time | Description |
|------|---------|------|-------------|
| 1 | `python scripts/train_teacher.py` | ~10h | Train DINOv3 teacher (100 epochs) |
| 2 | `python scripts/extract_signals.py` | ~20min | Extract teacher signals |
| 3 | `python scripts/train_distill.py` | ~4h | Distill to YOLO11m (50 epochs) |
| 4 | `python scripts/train_finetune.py` | ~1h | Fine-tune without distillation (10 epochs) |

For long runs use `nohup`:
```bash
nohup python scripts/train_teacher.py > logs/teacher.log 2>&1 &
tail -f logs/teacher.log
```

---

## Distillation Method

```
Teacher (DINOv3 ViT-L 303M)              Student (YOLO11m 20M)
+---------------------------+           +-------------------+
| ViT-L -> ViTBackbone      |    MSE    | YOLO Backbone     |
|        -> LightFPN        |<=========>| -> YOLO Neck      |
|        -> PredHead / Cls  |    KL     | -> YOLO Head      |
+---------------------------+<=========>+-------------------+
                                     L = L_det + a*MSE + b*KL

a(t): 0.5 -> 0.1   (feature alignment, early focus)
b(t): 0.3 -> 0.7   (classification KD, late focus)
KL temperature: T=3.0
```

### Teacher Training Schedule

| Phase | Epochs | Strategy | LR |
|-------|--------|----------|-----|
| 1 | 1-40 | Freeze ViT, train Neck+Head | 1e-3 |
| 2 | 41-70 | Unfreeze last 6 ViT blocks | 3e-4 / 1e-4 |
| 3 | 71-100 | Unfreeze all | 1e-4 |

EMA (0.9995), multi-scale [480-800], CosineWarmRestarts, Label Smoothing (0.05)

---

## Configuration

All paths in `config.py`. To use a custom root directory:

```bash
export DISTILL_HOME=/your/custom/path
```

Default: the repo directory itself.

---

## FAQ

- **HuggingFace download fails?** Login to huggingface.co, request access, or SCP upload from local
- **OOM?** Reduce `BATCH` in config.py, increase `ACCUM` to keep effective batch=32
- **`No module named 'dinov3'`?** Ensure `dinov3_repo/` is cloned in the right place
- **Multi-server migration?** Clone repo, upload `weights/` and `data/`, done

---

## Project Structure

```
dinov3-yolo-distill/
  config.py                   Global paths & hyperparams
  requirements.txt            Python dependencies
  dataset.yaml.example        Copy to data/dataset.yaml
  README.md
  scripts/
    verify.py                 Environment check
    train_teacher.py          Step 1: Teacher training
    extract_signals.py        Step 2: Signal extraction
    train_distill.py          Step 3: Distillation
    train_finetune.py         Step 4: Fine-tuning
  dinov3_repo/                DINOv3 source (git cloned)
  weights/                    Model weights (manual download)
  data/                       Dataset (manual upload)
  features/                   Teacher signals (auto-generated)
  runs/                       Training outputs (auto-generated)
```

## License

MIT