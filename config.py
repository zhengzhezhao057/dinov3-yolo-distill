"""config.py - All paths in one place. Set DISTILL_HOME env var or edit here."""
import os

HOME = os.environ.get("DISTILL_HOME", os.path.dirname(os.path.abspath(__file__)))

DINOV3_REPO   = os.path.join(HOME, "dinov3_repo")
VIT_WEIGHTS   = os.path.join(HOME, "weights", "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth")
YOLO11M_PT    = os.path.join(HOME, "weights", "yolo11m.pt")
DATASET_DIR   = os.path.join(HOME, "data")
DATASET_YAML  = os.path.join(DATASET_DIR, "dataset.yaml")
FEATURES_DIR  = os.path.join(HOME, "features", "teacher_signals")
RUNS_DIR      = os.path.join(HOME, "runs")
TEACHER_DIR   = os.path.join(RUNS_DIR, "teacher")
DISTILL_DIR   = os.path.join(RUNS_DIR, "distill")
FINETUNE_DIR  = os.path.join(RUNS_DIR, "finetune")

DEVICE = "cuda"; IMGSZ = 640; NC = 25; BATCH = 4; ACCUM = 8
