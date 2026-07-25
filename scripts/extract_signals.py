# extract_teacher_signals.py - Pre-compute teacher outputs for distillation
# Must match train_teacher_v4_custom.py architecture EXACTLY
# Server: RTX 3090 24GB, PT 2.9.1, CUDA 12.9

from __future__ import annotations
import sys, os, torch, torch.nn as nn, torch.nn.functional as F
import cv2, yaml
from pathlib import Path
from tqdm import tqdm

import sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); from config import *; BASE = HOME
sys.path.insert(0, DINOV3_REPO)
from dinov3.models.vision_transformer import DinoVisionTransformer

DEVICE = "cuda"
IMGSZ = 640
PATCH_SIZE = 16
FEAT_DIM = 1024
SPATIAL = IMGSZ // PATCH_SIZE
NC = 25
HOOK_BLOCKS = [5, 11, 17, 23]

print("=" * 50)
print("Extracting Teacher Signals")
print("=" * 50)

# ---- Load ViT-L (IDENTICAL to train_teacher_v4_custom.py) ----
print("Loading DINOv3 ViT-L...")
vit = DinoVisionTransformer(img_size=IMGSZ, patch_size=PATCH_SIZE, in_chans=3,
    embed_dim=FEAT_DIM, depth=24, num_heads=16, ffn_ratio=4.0, qkv_bias=True,
    drop_path_rate=0.0, layerscale_init=1e-5, norm_layer="layernormbf16",
    ffn_layer="mlp", ffn_bias=True, proj_bias=True, n_storage_tokens=4,
    mask_k_bias=True, untie_global_and_local_cls_norm=True)
sd = torch.load(VIT_WEIGHTS, map_location="cpu", weights_only=True)
vit.load_state_dict(sd, strict=True); del sd
print(f"  ViT-L: {sum(p.numel() for p in vit.parameters())/1e6:.1f}M params")

# Hooks
lf_store = {}
for idx in HOOK_BLOCKS:
    def mk_hook(i):
        def h(m, inp, out):
            o = out[0] if isinstance(out, (list, tuple)) else out
            if isinstance(o, torch.Tensor): lf_store[i] = o
        return h
    vit.blocks[idx].register_forward_hook(mk_hook(idx))

# ---- Models (IDENTICAL to train_teacher_v4_custom.py) ----
class ViTBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = vit; self.N = SPATIAL
        self.p3_proj = nn.Conv2d(FEAT_DIM, 256, 1)
        self.p4_proj = nn.Conv2d(FEAT_DIM, 512, 1)
        self.p5a_proj = nn.Conv2d(FEAT_DIM, 256, 1)
        self.p5b_proj = nn.Conv2d(FEAT_DIM, 256, 1)
        self.p5_fuse = nn.Conv2d(512, 512, 1)
    def _spatial(self, tokens, proj, hw):
        p = tokens[:, 5:self.N*self.N+5, :]
        B, _, D = p.shape
        f = p.permute(0,2,1).reshape(B,D,self.N,self.N)
        return F.interpolate(proj(f), size=hw, mode="bilinear", align_corners=False)
    def forward(self, x):
        _ = self.vit.forward_features(x)
        p3 = self._spatial(lf_store[5], self.p3_proj, (80,80))
        p4 = self._spatial(lf_store[11], self.p4_proj, (40,40))
        f17 = self._spatial(lf_store[17], self.p5a_proj, (self.N,self.N))
        f23 = self._spatial(lf_store[23], self.p5b_proj, (self.N,self.N))
        p5 = F.interpolate(self.p5_fuse(torch.cat([f17,f23],1)), size=(20,20), mode="bilinear", align_corners=False)
        return [p3, p4, p5]

class LightFPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.p5c = nn.Sequential(nn.Conv2d(512,256,1,bias=False),nn.BatchNorm2d(256),nn.SiLU(),nn.Conv2d(256,512,3,1,1,bias=False),nn.BatchNorm2d(512),nn.SiLU())
        self.p4r = nn.Sequential(nn.Conv2d(512,256,1,bias=False),nn.BatchNorm2d(256),nn.SiLU())
        self.p4f = nn.Sequential(nn.Conv2d(768,512,3,1,1,bias=False),nn.BatchNorm2d(512),nn.SiLU())
        self.p3r = nn.Sequential(nn.Conv2d(256,128,1,bias=False),nn.BatchNorm2d(128),nn.SiLU())
        self.p3f = nn.Sequential(nn.Conv2d(640,256,3,1,1,bias=False),nn.BatchNorm2d(256),nn.SiLU())
    def forward(self, xs):
        p3,p4,p5 = xs
        p5o = self.p5c(p5)
        p4o = self.p4f(torch.cat([self.p4r(p4), F.interpolate(p5o,p4.shape[2:],mode="nearest")],1))
        p3o = self.p3f(torch.cat([self.p3r(p3), F.interpolate(p4o,p3.shape[2:],mode="nearest")],1))
        return [p3o, p4o, p5o]

class ClassifierHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1280,512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512,256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256,NC))
    def forward(self, x): return self.net(x)

# Minimal teacher for extraction (same structure, no PredHead needed for signals)
class ExtractTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ViTBackbone()
        self.neck = LightFPN()
        self.cls_head = ClassifierHead()
    def forward(self, x):
        feats = self.backbone(x)
        neck = self.neck(feats)
        pooled = torch.cat([F.adaptive_avg_pool2d(f,1).flatten(1) for f in neck], 1)
        return pooled, self.cls_head(pooled)

teacher = ExtractTeacher().to(DEVICE)

# ---- Load trained weights ----
TEACHER_PT = os.path.join(TEACHER_DIR, "weights", "best.pt")
if not os.path.exists(TEACHER_PT):
    print(f"ERROR: {TEACHER_PT} not found!")
    sys.exit(1)

ckpt = torch.load(TEACHER_PT, map_location=DEVICE, weights_only=False)
# Handle both bare state_dict and wrapped checkpoints
if "model" in ckpt:
    sd = ckpt["model"]
else:
    sd = ckpt
# Remove pred head keys (not present in ExtractTeacher)
sd_filtered = {k:v for k,v in sd.items() if not k.startswith("head.")}
teacher.load_state_dict(sd_filtered, strict=False)
teacher.eval()
print(f"Teacher loaded from {TEACHER_PT}")
print(f"Params: {sum(p.numel() for p in teacher.parameters())/1e6:.1f}M")

# ---- Extract ----
with open(DATASET_YAML) as f:
    cfg = yaml.safe_load(f)
img_dir = cfg.get("path", DATASET_DIR) + "/images/train"
all_imgs = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))])
print(f"Images: {len(all_imgs)}")

OUT_DIR = Path(FEATURES_DIR)
OUT_DIR.mkdir(parents=True, exist_ok=True)
BATCH = 16

def preprocess(img_path):
    img = cv2.imread(img_path)
    if img is None: return None
    h, w = img.shape[:2]
    r = IMGSZ / max(h, w)
    nh, nw = int(h*r), int(w*r)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = IMGSZ-nw, IMGSZ-nh
    if dw < 0: dw = 0
    if dh < 0: dh = 0
    top, left = dh//2, dw//2
    img = cv2.copyMakeBorder(img, top, dh-top, left, dw-left, cv2.BORDER_CONSTANT, value=(114,114,114))
    return torch.from_numpy(img).float().permute(2,0,1)/255.0

print("Extracting...")
n_done = 0
with torch.no_grad():
    for i in tqdm(range(0, len(all_imgs), BATCH), desc="Signals"):
        batch_files = all_imgs[i:i+BATCH]
        imgs_t, stems = [], []
        for f in batch_files:
            r = preprocess(os.path.join(img_dir, f))
            if r is None: continue
            imgs_t.append(r); stems.append(f.rsplit(".",1)[0])
        if not imgs_t: continue
        imgs_batch = torch.stack(imgs_t).to(DEVICE)
        pooled, cls_logits = teacher(imgs_batch)
        for j, stem in enumerate(stems):
            torch.save({
                "pooled_neck": pooled[j].cpu(),      # (1280,)
                "cls_logits": cls_logits[j].cpu(),   # (25,)
            }, str(OUT_DIR / f"{stem}.pt"))
        n_done += len(stems)

print(f"Done! {n_done} signals saved to {OUT_DIR}")
sample = torch.load(str(OUT_DIR / f"{stems[0]}.pt"), map_location="cpu", weights_only=False)
print(f"Sample: pooled_neck={sample['pooled_neck'].shape}, cls_logits={sample['cls_logits'].shape}")
