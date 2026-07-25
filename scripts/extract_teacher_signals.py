# extract_teacher_signals.py - Pre-compute teacher outputs for distillation
# Run after teacher training. Extracts neck features + detection + classification logits.
# Server: RTX 3090 24GB, PT 2.9.1, CUDA 12.9

from __future__ import annotations
import sys, os, torch, torch.nn as nn, torch.nn.functional as F
import cv2, yaml, numpy as np
from pathlib import Path
from tqdm import tqdm

BASE = "/root/autodl-tmp"
sys.path.insert(0, f"{BASE}/dinov3_repo/dinov3-main")

DEVICE = "cuda"
IMGSZ = 640

# ===== IMPORTANT: Rebuild teacher model identically =====
# Must match train_teacher_v4_trainer.py architecture exactly
from dinov3.models.vision_transformer import DinoVisionTransformer
from ultralytics.nn.modules.head import Detect

FEAT_DIM = 384
SPATIAL = 46
N_PATCHES = SPATIAL ** 2

# Hooks
lf_store = {}
for idx in [2, 5, 8, 11]:
    def make_hook(i):
        def hook(m, inp, out):
            o = out[0] if isinstance(out, (list, tuple)) else out
            if isinstance(o, torch.Tensor): lf_store[i] = o
        return hook

# Load ViT
vit = DinoVisionTransformer(
    img_size=644, patch_size=14, in_chans=3, embed_dim=384, depth=12, num_heads=6,
    ffn_ratio=4.0, qkv_bias=True, drop_path_rate=0.0, layerscale_init=1e-5,
    norm_layer="layernormbf16", ffn_layer="mlp", ffn_bias=True, proj_bias=True,
    n_storage_tokens=4, mask_k_bias=True, untie_global_and_local_cls_norm=True,
)
weight_path = f"{BASE}/dinov3_vits14_pretrain_lvd1689m.pth"
if not os.path.exists(weight_path):
    for f in Path(f"{BASE}/dinov3_s_weights").rglob("*.pth"): weight_path = str(f); break
sd = torch.load(weight_path, map_location="cpu", weights_only=True)
vit.load_state_dict(sd, strict=True); del sd

for idx in [2, 5, 8, 11]:
    vit.blocks[idx].register_forward_hook(make_hook(idx))

# ===== Rebuild teacher architecture =====
class DINOv3BackboneWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = vit; self.N = SPATIAL
        self.p3_proj = nn.Conv2d(384, 256, 1)
        self.p4_proj = nn.Conv2d(384, 256, 1)
        self.p5a_proj = nn.Conv2d(384, 256, 1)
        self.p5b_proj = nn.Conv2d(384, 256, 1)
        self.p5_fuse = nn.Conv2d(512, 512, 1)
    def _to_spatial(self, tokens, proj, hw):
        p = tokens[:, 5:self.N*self.N+5, :]
        B, _, D = p.shape
        f = p.permute(0, 2, 1).reshape(B, D, self.N, self.N)
        f = proj(f)
        return F.interpolate(f, size=hw, mode="bilinear", align_corners=False) if hw != (self.N, self.N) else f
    def forward(self, x):
        if x.shape[2] != 644: x = F.interpolate(x, size=(644, 644), mode="bilinear", align_corners=False)
        _ = self.vit.forward_features(x)
        p3 = self._to_spatial(lf_store[2], self.p3_proj, (80, 80))
        p4 = self._to_spatial(lf_store[5], self.p4_proj, (40, 40))
        f8 = self._to_spatial(lf_store[8], self.p5a_proj, (self.N, self.N))
        f11 = self._to_spatial(lf_store[11], self.p5b_proj, (self.N, self.N))
        p5 = self.p5_fuse(torch.cat([f8, f11], 1))
        p5 = F.interpolate(p5, size=(40, 40), mode="bilinear", align_corners=False)
        return [p3, p4, p5]

class LightFPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.p5_up = nn.Sequential(nn.Conv2d(512, 256, 1, bias=False), nn.BatchNorm2d(256), nn.SiLU())
        self.p4_merge = nn.Sequential(nn.Conv2d(512, 256, 1, bias=False), nn.BatchNorm2d(256), nn.SiLU())
        self.p4_out = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512), nn.SiLU())
        self.p4_up = nn.Sequential(nn.Conv2d(512, 128, 1, bias=False), nn.BatchNorm2d(128), nn.SiLU())
        self.p3_merge = nn.Sequential(nn.Conv2d(256, 128, 1, bias=False), nn.BatchNorm2d(128), nn.SiLU())
        self.p3_out = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256), nn.SiLU())
    def forward(self, xs):
        p3, p4, p5 = xs
        p5_f = self.p5_up(p5)
        p5_up = F.interpolate(p5_f, size=p4.shape[2:], mode="nearest")
        p4_cat = torch.cat([self.p4_merge(p4), p5_up], 1)
        p4_out = self.p4_out(p4_cat)
        p4_up = self.p4_up(p4_out)
        p4_up = F.interpolate(p4_up, size=p3.shape[2:], mode="nearest")
        p3_cat = torch.cat([self.p3_merge(p3), p4_up], 1)
        p3_out = self.p3_out(p3_cat)
        return [p3_out, p4_out, p5_f]

class ClassifierHead(nn.Module):
    def __init__(self, in_dim=1280, nc=25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, nc),
        )
    def forward(self, x): return self.net(x)

NC = 25
detect = Detect(nc=NC, ch=[256, 512, 256])

class FullTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = DINOv3BackboneWrapper()
        self.neck = LightFPN()
        self.detect = detect
        self.cls_head = ClassifierHead(1280, NC)
    def forward(self, x):
        feats = self.backbone(x)
        neck_feats = self.neck(feats)
        det_out = self.detect(neck_feats)
        pooled = torch.cat([F.adaptive_avg_pool2d(f, 1).flatten(1) for f in neck_feats], 1)
        cls_logits = self.cls_head(pooled)
        return det_out, pooled, cls_logits

# ===== Load trained teacher =====
TEACHER_PT = f"{BASE}/runs/teacher_v4/weights/best.pt"
if not os.path.exists(TEACHER_PT):
    TEACHER_PT = f"{BASE}/runs/teacher_v4/weights/teacher_init.pt"

teacher = FullTeacher().to(DEVICE)
ckpt = torch.load(TEACHER_PT, map_location=DEVICE, weights_only=False)
teacher.load_state_dict(ckpt if "detect" in ckpt else ckpt, strict=False)
teacher.eval()
print(f"Teacher loaded: {sum(p.numel() for p in teacher.parameters())/1e6:.1f}M params")

# ===== Extract signals =====
with open(f"{BASE}/split_dataset/dataset.yaml") as f:
    cfg = yaml.safe_load(f)

img_dir = cfg["path"] + "/images/train"
all_imgs = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))])

OUT_DIR = Path(f"{BASE}/features/teacher_signals")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH = 16

print(f"Extracting teacher signals for {len(all_imgs)} images...")

def preprocess(img_path):
    img = cv2.imread(img_path)
    if img is None: return None
    h, w = img.shape[:2]
    r = 640 / max(h, w)
    nh, nw = int(h * r), int(w * r)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = 640 - nw, 640 - nh
    top, left = dh // 2, dw // 2
    img = cv2.copyMakeBorder(img, top, dh - top, left, dw - left, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    img_t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
    return img_t, (dw, dh, left, top)

n_done = 0
with torch.no_grad():
    for i in tqdm(range(0, len(all_imgs), BATCH), desc="Teacher signals"):
        batch_files = all_imgs[i:i+BATCH]
        imgs_t, stems = [], []
        
        for f in batch_files:
            r = preprocess(os.path.join(img_dir, f))
            if r is None: continue
            imgs_t.append(r[0])
            stems.append(f.rsplit(".", 1)[0])
        
        if not imgs_t: continue
        
        imgs_batch = torch.stack(imgs_t).to(DEVICE)
        det_out, pooled, cls_logits = teacher(imgs_batch)
        
        for j, stem in enumerate(stems):
            data = {
                "pooled_neck": pooled[j].cpu(),
                "cls_logits": cls_logits[j].cpu(),
                # Save detection outputs per scale for box KD
                "det_p3": det_out[0][j].cpu() if len(det_out) > 0 else None,
                "det_p4": det_out[1][j].cpu() if len(det_out) > 1 else None,
                "det_p5": det_out[2][j].cpu() if len(det_out) > 2 else None,
            }
            torch.save(data, str(OUT_DIR / f"{stem}.pt"))
        
        n_done += len(stems)

print(f"Done! {n_done} signals saved to {OUT_DIR}")
sample = torch.load(str(OUT_DIR / f"{stems[0]}.pt") if stems else None, map_location="cpu", weights_only=False)
if sample:
    print(f"  Sample: pooled_neck={sample['pooled_neck'].shape}, cls_logits={sample['cls_logits'].shape}")
