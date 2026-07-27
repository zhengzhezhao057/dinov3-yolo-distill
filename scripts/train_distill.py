# train_distill_v4_mgd.py - 2-Branch Distillation: MGD Feature + Focal + Classification KD
# MGD: Masked student features → generate teacher features (not direct MSE)
# FGD: Foreground-focused distillation (only distill where objects exist)
# Student: YOLO11m. Teacher: pre-computed signals.
# Server: RTX 3090 24GB, PT 2.9.1, CUDA 12.9

from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F, os
from pathlib import Path
from ultralytics import YOLO

# ===== CONFIG =====
import sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); from config import *; BASE = HOME
DATA_YAML = DATASET_YAML
TEACHER_SIGNALS_DIR = FEATURES_DIR
PROJECT = RUNS_DIR
EXPERIMENT = "distill_mgd"
EPOCHS = 50
BATCH = 16
NC = 25
KD_T_START, KD_T_END = 10.0, 2.0  # Temperature annealing

# Progressive weights
FEAT_W_START, FEAT_W_END = 0.5, 0.1
CLS_KD_W_START, CLS_KD_W_END = 0.3, 0.7
TEACHER_FEAT_DIM = 1280
MGD_MASK_RATIO = 0.3  # Mask 30% of student feature dims

print("=" * 60)
print("2-Branch Distillation: MGD Feature + Focal + Classification KD")
print(f"  MGD mask ratio: {MGD_MASK_RATIO}")
print(f"  FEAT weight: {FEAT_W_START:.1f} -> {FEAT_W_END:.1f}")
print(f"  CLS_KD weight: {CLS_KD_W_START:.1f} -> {CLS_KD_W_END:.1f}")
print(f"  KL temp: {KD_T_START:.1f} -> {KD_T_END:.1f}")
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
model = YOLO(YOLO11M_PT)
device = next(model.model.parameters()).device
yolo_model = model.model

kl_loss = nn.KLDivLoss(reduction="batchmean")

# ===== Capture student neck features =====
neck_feat_cache = {"feat": None}
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
        if student_feat_dim is None:
            student_feat_dim = pooled.shape[1]
            print(f"  Student neck dim: {student_feat_dim}")

found = False
for name, mod in yolo_model.named_modules():
    if hasattr(mod, "no") and hasattr(mod, "nl") and hasattr(mod, "stride"):
        mod.register_forward_pre_hook(neck_hook)
        print(f"  Hook registered on {type(mod).__name__}")
        found = True
        break

# ===== Projection + MGD Generator =====
proj_head = None
mgd_generator = None  # MGD: generates teacher features from masked student
cls_proj = None

def ensure_heads():
    global proj_head, mgd_generator, cls_proj
    if proj_head is None and student_feat_dim is not None:
        # Feature projection (kept for classification path)
        proj_head = nn.Sequential(
            nn.Linear(student_feat_dim, 512),
            nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, TEACHER_FEAT_DIM),
        ).to(device)
        # MGD Generator: masked student -> teacher features
        mgd_generator = nn.Sequential(
            nn.Linear(student_feat_dim, 512),
            nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, TEACHER_FEAT_DIM),
        ).to(device)
        cls_proj = nn.Linear(student_feat_dim, NC).to(device)
        print(f"  Heads created: feat_proj + MGD_gen + cls_proj")
        print(f"  MGD generator: {student_feat_dim}->512->512->{TEACHER_FEAT_DIM}")

# ===== Distillation loss =====
original_loss_fn = yolo_model.loss
current_epoch = [0]

def get_weights(epoch):
    r = min(epoch / max(EPOCHS - 1, 1), 1.0)
    return (
        FEAT_W_START + (FEAT_W_END - FEAT_W_START) * r,
        CLS_KD_W_START + (CLS_KD_W_END - CLS_KD_W_START) * r,
        KD_T_START + (KD_T_END - KD_T_START) * r,  # Temperature annealing
    )

def compute_focal_weight(batch, B, hw_ratio=20):
    """FGD: compute per-image foreground weight based on bbox count/area."""
    bboxes = batch.get("bboxes", None)
    if bboxes is None or bboxes.numel() == 0:
        return torch.ones(B, device=device)
    batch_idx = batch.get("batch_idx", None)
    if batch_idx is None:
        return torch.ones(B, device=device)
    weights = torch.ones(B, device=device)
    for i in range(B):
        n = (batch_idx == i).sum().item()
        if n > 0:
            # More objects -> higher weight (capped at 3x)
            weights[i] = min(1.0 + n / hw_ratio, 3.0)
    return weights

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
    
    feat_w, cls_w, kd_t = get_weights(current_epoch[0])
    
    # === MGD Feature Distillation ===
    # 1. Randomly mask student feature dimensions
    # 2. Generator produces teacher-like features from masked input
    # 3. MSE loss between generated and teacher features
    mgd_loss = torch.tensor(0.0, device=device)
    mgd_count = 0
    focal_weights = compute_focal_weight(batch, B)  # FGD weights
    
    for j in range(min(B, len(im_files))):
        stem = Path(im_files[j]).stem
        if stem in teacher_signals:
            stu_feat = feat[j:j+1].float()  # (1, stu_dim)
            tch_feat = teacher_signals[stem]["pooled_neck"].to(device).float().unsqueeze(0)
            
            # MGD: mask random dimensions of student feature
            mask = (torch.rand_like(stu_feat) > MGD_MASK_RATIO).float()
            masked_feat = stu_feat * mask
            gen_feat = mgd_generator(masked_feat)  # (1, teacher_dim)
            
            # FGD: weight by foreground presence
            fw = focal_weights[j]
            mgd_loss = mgd_loss + fw * F.mse_loss(gen_feat, tch_feat)
            mgd_count += 1
    
    if mgd_count > 0:
        mgd_loss = mgd_loss / mgd_count
    
    # === Classification KD (with temperature annealing) ===
    cls_kd_loss = torch.tensor(0.0, device=device)
    cls_count = 0
    stu_logits = cls_proj(feat.float())
    
    for j in range(min(B, len(im_files))):
        stem = Path(im_files[j]).stem
        if stem in teacher_signals:
            t_logits = teacher_signals[stem]["cls_logits"].to(device).float().unsqueeze(0)
            teacher_prob = (t_logits / kd_t).softmax(-1)
            student_log_prob = (stu_logits[j:j+1] / kd_t).log_softmax(-1)
            cls_kd_loss = cls_kd_loss + kl_loss(student_log_prob, teacher_prob)
            cls_count += 1
    
    if cls_count > 0:
        cls_kd_loss = cls_kd_loss / cls_count
    
    total_distill = feat_w * mgd_loss + cls_w * cls_kd_loss
    if torch.isfinite(total_distill) and total_distill > 0:
        loss = loss + total_distill
    
    neck_feat_cache["feat"] = None
    return loss, loss_items

yolo_model.loss = distill_loss_fn
print(f"  Loss override complete (MGD + Focal + Temp Annealing)")

# ===== Train =====
print(f"\n[3/4] Training ({EPOCHS} epochs)...")

def on_train_epoch_end(trainer):
    current_epoch[0] = trainer.epoch + 1
    if current_epoch[0] % 10 == 1:
        fw, cw, kdt = get_weights(current_epoch[0])
        print(f"  Epoch {current_epoch[0]}: feat_w={fw:.3f}, cls_w={cw:.3f}, KD_T={kdt:.1f}")

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