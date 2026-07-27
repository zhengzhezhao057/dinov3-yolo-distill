"""config.py - DINOv3 ViT-L -> YOLO11m Distillation. Set DISTILL_HOME env var or edit here."""
import os

HOME = os.environ.get("DISTILL_HOME", os.path.dirname(os.path.abspath(__file__)))

DINOV3_REPO   = os.path.join(HOME, "dinov3_repo")
WEIGHTS_DIR   = os.path.join(HOME, "weights")
VIT_WEIGHTS   = os.path.join(WEIGHTS_DIR, "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth")
YOLO11M_PT    = os.path.join(WEIGHTS_DIR, "yolo11m.pt")
DATASET_DIR   = os.path.join(HOME, "data")
DATASET_YAML  = os.path.join(DATASET_DIR, "dataset.yaml")
FEATURES_DIR  = os.path.join(HOME, "features", "teacher_signals")
RUNS_DIR      = os.path.join(HOME, "runs")
TEACHER_DIR   = os.path.join(RUNS_DIR, "teacher")
DISTILL_DIR   = os.path.join(RUNS_DIR, "distill")
FINETUNE_DIR  = os.path.join(RUNS_DIR, "finetune")

DEVICE = "cuda"; IMGSZ = 640; NC = 25; BATCH = 4; ACCUM = 8

HF_REPO = "zzz0917/dinov3-yolo-distill-weights"
YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt"

def ensure_weights():
    """Auto-download weights if missing."""
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    if not os.path.exists(VIT_WEIGHTS):
        print(f"Downloading ViT-L weights from HuggingFace ({HF_REPO})...")
        from huggingface_hub import hf_hub_download
        hf_hub_download(HF_REPO, "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
                        local_dir=WEIGHTS_DIR, local_dir_use_symlinks=False)
    if not os.path.exists(YOLO11M_PT):
        print("Downloading YOLO11m...")
        import urllib.request
        urllib.request.urlretrieve(YOLO_URL, YOLO11M_PT)