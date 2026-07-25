# train_finetune_v4.py - Phase 3: Close distillation, finetune on real labels
# Takes best distill model and fine-tunes 10 epochs with no distillation
# Server: RTX 3090 24GB, PT 2.9.1, CUDA 12.9

from ultralytics import YOLO
import os

import sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); from config import *; BASE = HOME
DATA_YAML = DATASET_YAML
DISTILL_BEST = os.path.join(DISTILL_DIR, "weights", "best.pt")
PROJECT = RUNS_DIR
EXPERIMENT = "finetune_v4"

print("=" * 50)
print("Phase 3: Finetune (No Distillation)")
print(f"  Input: {DISTILL_BEST}")
print(f"  Epochs: 10")
print("=" * 50)

if not os.path.exists(DISTILL_BEST):
    print(f"WARNING: {DISTILL_BEST} not found!")
    print("Looking for alternative distill model...")
    import glob
    candidates = glob.glob(os.path.join(RUNS_DIR, "distill*", "weights", "best.pt"))
    if candidates:
        DISTILL_BEST = sorted(candidates)[-1]
        print(f"  Using: {DISTILL_BEST}")
    else:
        print("ERROR: No distill model found. Run train_distill_v4.py first.")
        exit(1)

model = YOLO(DISTILL_BEST)

model.train(
    data=DATA_YAML,
    epochs=10,
    imgsz=640,
    batch=16,
    device=0,
    workers=8,
    project=PROJECT,
    name=EXPERIMENT,
    exist_ok=True,
    resume=False,
    amp=True,
    lr0=0.0001,             # Low LR for finetune
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=1,
    cos_lr=True,
    close_mosaic=0,          # No mosaic in finetune
    patience=0,
    save=True,
    plots=True,
)

print(f"\n✅ Finetune complete!")
print(f"Best model: {PROJECT}/{EXPERIMENT}/weights/best.pt")

# ===== Validate on test set =====
print("\nValidating on test set...")
metrics = model.val(data=DATA_YAML, split="test")
print(f"Test mAP50: {metrics.box.map50:.4f}")
print(f"Test mAP50-95: {metrics.box.map:.4f}")

