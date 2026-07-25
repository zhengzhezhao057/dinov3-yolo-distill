"""verify.py - Check environment"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
import torch

errors = []
print("=" * 50)
print(f"DINOv3-YOLO Distill  HOME={HOME}")
print("=" * 50)

print(f"\n[1] PyTorch {torch.__version__}  CUDA={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"    GPU: {torch.cuda.get_device_name(0)}")

for pkg, name in [("ultralytics","ultralytics"),(("cv2","cv2")),(("yaml","yaml"))]:
    try: __import__(name); print(f"[2] {pkg}: OK")
    except: errors.append(f"pip install {pkg}")

sys.path.insert(0, DINOV3_REPO)
try:
    from dinov3.models.vision_transformer import DinoVisionTransformer
    print("[3] DINOv3: OK")
except: errors.append(f"DINOv3 clone failed: {DINOV3_REPO}")

for label, path in [("ViT-L",VIT_WEIGHTS),("YOLO11m",YOLO11M_PT),("dataset.yaml",DATASET_YAML)]:
    ok = os.path.exists(path)
    sz = f" ({os.path.getsize(path)/1e6:.0f}MB)" if ok else ""
    print(f"[4] {label}: {'OK' if ok else 'MISSING'}{sz}")
    if not ok: errors.append(f"Missing: {label}")

if os.path.exists(DATASET_YAML):
    import yaml
    with open(DATASET_YAML) as f: cfg = yaml.safe_load(f)
    d = cfg.get("path", DATASET_DIR)
    td = os.path.join(d, cfg.get("train","images/train"))
    if os.path.exists(td):
        n = len([f for f in os.listdir(td) if f.endswith(('.jpg','.png','.bmp','.jpeg'))])
        print(f"[5] Train images: {n}")
        if n < 100: errors.append(f"Only {n} images!")

print("\n" + "=" * 50)
if errors:
    print(f"FAILED ({len(errors)} issues):")
    for e in errors: print(f"  - {e}")
    sys.exit(1)
print("ALL OK! python scripts/train_teacher.py")
