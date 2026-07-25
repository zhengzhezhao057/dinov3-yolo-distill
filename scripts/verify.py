#!/usr/bin/env python3
"""verify.py - Check environment before training"""
import sys, os, torch

BASE = os.environ.get("BASE", "/root/autodl-tmp")
errors = []

print("=" * 50)
print("DINOv3-YOLO Distill - Environment Check")
print("=" * 50)

print(f"\n[1] PyTorch: {torch.__version__}")
print(f"    CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"    GPU: {torch.cuda.get_device_name(0)}")
    mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"    VRAM: {mem:.1f} GB")
    if mem < 20: errors.append("VRAM < 20GB")

v = sys.version_info
print(f"\n[2] Python: {v.major}.{v.minor}.{v.micro}")
if v < (3, 10): errors.append("Need Python 3.10+")

for pkg in ["ultralytics", "cv2", "yaml", "numpy"]:
    try:
        __import__(pkg) if pkg != "cv2" else __import__("cv2")
        print(f"[3] {pkg}: OK")
    except: errors.append(f"Missing: {pkg}")

sys.path.insert(0, f"{BASE}/dinov3_repo")
try:
    from dinov3.models.vision_transformer import DinoVisionTransformer
    print("[4] DINOv3: OK")
except: errors.append("DINOv3 not found. Clone: git clone https://github.com/facebookresearch/dinov3.git /root/autodl-tmp/dinov3_repo")

checks = [
    ("ViT-L weights", f"{BASE}/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"),
    ("YOLO11m", f"{BASE}/yolo11m.pt"),
    ("dataset.yaml", f"{BASE}/split_dataset/dataset.yaml"),
]
for label, path in checks:
    ok = os.path.exists(path)
    print(f"[5] {label}: {'OK' if ok else 'MISSING'}  ({path})")
    if not ok: errors.append(f"Missing: {label}")

print("\n" + "=" * 50)
if errors:
    print(f"FAILED: {len(errors)} issue(s)")
    for e in errors: print(f"  - {e}")
    sys.exit(1)
print("ALL OK! Run: python scripts/train_teacher.py")
