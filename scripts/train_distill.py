# train_distill_v4.py - 2-Branch Distillation: Feature MSE + Classification KD
# Student: YOLO11m. Teacher: pre-computed signals.
# Box KD removed (too complex to get right, marginal gain)
# Server: RTX 3090 24GB, PT 2.9.1, CUDA 12.9

from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F, os
from pathlib import Path
from ultralytics import YOLO

# ===== CONFIG =====
BASE = "/root/autodl-tmp"
DATA_YAML = f"{BASE}/split_dataset/dataset.yaml"
TEACHER_SIGNALS_DIR = f"{BASE}/features/teacher_signals"
PROJECT = f"{BASE}/runs"
EXPERIMENT = "distill_v4"
EPOCHS = 50
BATCH = 16
NC = 25
KD_T = 3.0

# Progressive weights
FEAT_W_START, FEAT_W_END = 0.5, 0.1
CLS_KD_W_START, CLS_KD_W_END = 0.3, 0.7
TEACHER_FEAT_DIM = 1280  # 256+512+512 from neck

print("=" * 60)
print("2-Branch Distillation (Feature MSE + Classification KD)")
print(f"  FEAT weight: {FEAT_W_START:.1f} -> {FEAT_W_END:.1f}")
print(f"  CLS_KD weight: {CLS_KD_W_START:.1f} -> {CLS_KD_W_END:.1f}")
print(f"  Epochs: {EPOCHS}  Batch: {BATCH}")
print("=" * 60)

# ===== Load teacher signals =====
print("\n[1/4] Loading teacher signals...")
teacher_signals = {}
signal_files = list(Path(TEACHER_SIGNALS_DIR).glob("*.pt"))
for ptf in signal_files:
    try:
        teacher_signals[ptf.stem] = torch.load(ptf, map_location="cpu", weights_only=False)
    except: pass
print(f"  Loaded {len(teacher_signals)} signals")

if len(teacher_signals) == 0:
    print("ERROR: No signals! Run extract_teacher_signals.py first")
    exit(1)

sample = teacher_signals[next(iter(teacher_signals.keys()))]
print(f"  pooled_neck: {sample['pooled_neck'].shape}")
print(f"  cls_logits: {sample['cls_logits'].shape}")

# ===== Load YOLO11m =====
print("\n[2/4] Loading YOLO11m...")
model = YOLO(f"{BASE}/yolo11m.pt")
device = next(model.model.parameters()).device
yolo_model = model.model

mse = nn.MSELoss()
kl_loss = nn.KLDivLoss(reduction="batchmean")

# ===== Capture student neck features =====
neck_feat_cache = {"feat": None, "raw": None}
student_feat_dim = None

def neck_hook(module, inputs):
    global student_feat_dim
    if isinstance(inputs, (list, tuple)):
        feats = [x for x in inputs if isinstance(x, torch.Tensor) and x.dim() == 4]
    else:
        return
    if len(feats) >= 3:
        feats = feats[-3:]
        pooled = torch.cat([F.adaptive_avg_pool2d(f, 1).flatten(1) for f in feats], 1)
        neck_feat_cache["feat"] = pooled
        neck_feat_cache["raw"] = feats
        if student_feat_dim is None:
            student_feat_dim = pooled.shape[1]
            print(f"  Student neck dim: {student_feat_dim}")

# Register hook on Detect module
found = False
for name, mod in yolo_model.named_modules():
    if hasattr(mod, "no") and hasattr(mod, "nl") and hasattr(mod, "stride"):
        mod.register_forward_pre_hook(neck_hook)
        print(f"  Hook registered on {type(mod).__name__}")
        found = True
        break

if not found:
    print("WARNING: Could not find Detect module, trying alternative...")
    # Fallback: hook on the last layer before Detect
    for name, mod in yolo_model.named_modules():
        if hasattr(mod, "cv2") and hasattr(mod, "cv3"):
            mod.register_forward_pre_hook(neck_hook)
            print(f"  Hook registered on {type(mod).__name__}")
            found = True
            break

# ===== Projection head =====
proj_head = None
cls_proj = None

def ensure_heads():
    global proj_head, cls_proj
    if proj_head is None and student_feat_dim is not None:
        proj_head = nn.Sequential(
            nn.Linear(student_feat_dim, 512),
            nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, TEACHER_FEAT_DIM),
        ).to(device)
        cls_proj = nn.Linear(student_feat_dim, NC).to(device)
        print(f"  Projection heads created: {student_feat_dim}->{TEACHER_FEAT_DIM} (feat), {student_feat_dim}->{NC} (cls)")

# ===== Distillation loss =====
original_loss_fn = yolo_model.loss
current_epoch = [0]

def get_weights(epoch):
    r = min(epoch / max(EPOCHS - 1, 1), 1.0)
    return FEAT_W_START + (FEAT_W_END - FEAT_W_START) * r, CLS_KD_W_START + (CLS_KD_W_END - CLS_KD_W_START) * r

def distill_loss_fn(preds, batch):
    # Original YOLO loss
    loss, loss_items = original_loss_fn(preds, batch)
    
    im_files = batch.get("im_file", [])
    feat = neck_feat_cache.get("feat")
    if feat is None or not im_files:
        neck_feat_cache["feat"] = None
        return loss, loss_items
    
    B = feat.shape[0]
    ensure_heads()
    if proj_head is None:
        neck_feat_cache["feat"] = None
        return loss, loss_items
    
    feat_w, cls_w = get_weights(current_epoch[0])
    
    proj_feat = proj_head(feat.float())   # (B, teacher_dim)
    stu_logits = cls_proj(feat.float())   # (B, NC)
    
    # Feature MSE
    feat_loss = torch.tensor(0.0, device=device)
    feat_count = 0
    for j in range(min(B, len(im_files))):
        stem = Path(im_files[j]).stem
        if stem in teacher_signals:
            t_feat = teacher_signals[stem]["pooled_neck"].to(device).float().unsqueeze(0)
            feat_loss = feat_loss + mse(proj_feat[j:j+1], t_feat)
            feat_count += 1
    if feat_count > 0:
        feat_loss = feat_loss / feat_count
    
    # Classification KD
    cls_kd_loss = torch.tensor(0.0, device=device)
    cls_count = 0
    for j in range(min(B, len(im_files))):
        stem = Path(im_files[j]).stem
        if stem in teacher_signals:
            t_logits = teacher_signals[stem]["cls_logits"].to(device).float().unsqueeze(0)
            teacher_prob = (t_logits / KD_T).softmax(-1)
            student_log_prob = (stu_logits[j:j+1] / KD_T).log_softmax(-1)
            cls_kd_loss = cls_kd_loss + kl_loss(student_log_prob, teacher_prob)
            cls_count += 1
    if cls_count > 0:
        cls_kd_loss = cls_kd_loss / cls_count
    
    total_distill = feat_w * feat_loss + cls_w * cls_kd_loss
    if torch.isfinite(total_distill) and total_distill > 0:
        loss = loss + total_distill
    
    neck_feat_cache["feat"] = None
    return loss, loss_items

yolo_model.loss = distill_loss_fn
print(f"  Loss override complete")

# ===== Train with epoch callback =====
print(f"\n[3/4] Training ({EPOCHS} epochs)...")

# Use add_callback to update current_epoch
def on_train_epoch_end(trainer):
    current_epoch[0] = trainer.epoch + 1
    if current_epoch[0] % 10 == 1:
        fw, cw = get_weights(current_epoch[0])
        print(f"  Epoch {current_epoch[0]}: feat_w={fw:.3f}, cls_w={cw:.3f}")

model.add_callback("on_train_epoch_end", on_train_epoch_end)

model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=640,
    batch=BATCH,
    device=0,
    workers=8,
    project=PROJECT,
    name=EXPERIMENT,
    exist_ok=True,
    resume=False,
    amp=True,
    lr0=0.0005,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    warmup_momentum=0.8,
    cos_lr=True,
    close_mosaic=15,
    patience=0,
    save=True,
    plots=True,
)

print(f"\nDone! {PROJECT}/{EXPERIMENT}/weights/best.pt")