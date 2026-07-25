# train_teacher_v4_custom.py - DINOv3 ViT-L Teacher Training
# Self-contained, ZERO ultralytics internals
# Server: RTX 3090 24GB, PT 2.9.1, CUDA 12.9

from __future__ import annotations
import sys, os, torch, torch.nn as nn, torch.nn.functional as F
import time, cv2, yaml, numpy as np, random, math
from pathlib import Path
from tqdm import tqdm

BASE = "/root/autodl-tmp"
sys.path.insert(0, f"{BASE}/dinov3_repo")
from dinov3.models.vision_transformer import DinoVisionTransformer

DEVICE = "cuda"; IMGSZ = 640; PATCH_SIZE = 16; FEAT_DIM = 1024
SPATIAL = IMGSZ // PATCH_SIZE; NC = 25
BATCH = 4; ACCUM = 8; EPOCHS = 100; EMA_DECAY = 0.9995; LABEL_SMOOTH = 0.05
PHASE1_END = 40; PHASE2_END = 70; HOOK_BLOCKS = [5, 11, 17, 23]
REG_MAX = 16; NL = 3; NA = 3
NO = NC + 4 * REG_MAX  # 89

print("="*60)
print(f"DINOv3 ViT-L Teacher: {EPOCHS} epochs, batch {BATCH}x{ACCUM}={BATCH*ACCUM}")
print(f"Phase1(1-{PHASE1_END}) freeze | Phase2({PHASE1_END+1}-{PHASE2_END}) unfreeze6 | Phase3({PHASE2_END+1}-{EPOCHS}) all")
print("="*60)

# ---- Load ViT-L ----
print("Loading DINOv3 ViT-L...")
vit = DinoVisionTransformer(img_size=IMGSZ, patch_size=PATCH_SIZE, in_chans=3,
    embed_dim=FEAT_DIM, depth=24, num_heads=16, ffn_ratio=4.0, qkv_bias=True,
    drop_path_rate=0.0, layerscale_init=1e-5, norm_layer="layernormbf16",
    ffn_layer="mlp", ffn_bias=True, proj_bias=True, n_storage_tokens=4,
    mask_k_bias=True, untie_global_and_local_cls_norm=True)
sd = torch.load(f"{BASE}/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth", map_location="cpu", weights_only=True)
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

# ---- ViT Backbone ----
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

# ---- LightFPN ----
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

# ---- Prediction Heads (replaces Detect module) ----
class PredHead(nn.Module):
    """Self-contained prediction heads for P3/P4/P5 -> boxes + classes"""
    def __init__(self):
        super().__init__()
        # P3: 256 -> 3*(25+4*16) = 267
        self.cv3 = nn.ModuleList([
            nn.Sequential(nn.Conv2d(256, 64, 3, 1, 1), nn.SiLU(), nn.Conv2d(64, 64, 3, 1, 1), nn.SiLU(), nn.Conv2d(64, NA*NO, 1)),
            nn.Sequential(nn.Conv2d(256, 64, 3, 1, 1), nn.SiLU(), nn.Conv2d(64, 64, 3, 1, 1), nn.SiLU(), nn.Conv2d(64, NA*NO, 1)),
        ])
        # P4: 512 -> 267
        self.cv4 = nn.ModuleList([
            nn.Sequential(nn.Conv2d(512, 64, 3, 1, 1), nn.SiLU(), nn.Conv2d(64, 64, 3, 1, 1), nn.SiLU(), nn.Conv2d(64, NA*NO, 1)),
            nn.Sequential(nn.Conv2d(512, 64, 3, 1, 1), nn.SiLU(), nn.Conv2d(64, 64, 3, 1, 1), nn.SiLU(), nn.Conv2d(64, NA*NO, 1)),
        ])
        # P5: 512 -> 267
        self.cv5 = nn.ModuleList([
            nn.Sequential(nn.Conv2d(512, 64, 3, 1, 1), nn.SiLU(), nn.Conv2d(64, 64, 3, 1, 1), nn.SiLU(), nn.Conv2d(64, NA*NO, 1)),
            nn.Sequential(nn.Conv2d(512, 64, 3, 1, 1), nn.SiLU(), nn.Conv2d(64, 64, 3, 1, 1), nn.SiLU(), nn.Conv2d(64, NA*NO, 1)),
        ])
        self._init_weights()
    def _init_weights(self):
        for cv_list in [self.cv3, self.cv4, self.cv5]:
            for seq in cv_list:
                for m in seq.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None: m.bias.data.zero_()
    def forward(self, feats):
        """feats: [P3@80, P4@40, P5@20] -> list of [(B,NA*NO,H,W), (B,NA*NO,H,W)]"""
        p3, p4, p5 = feats
        return [
            self.cv3[0](p3), self.cv3[1](p3),
            self.cv4[0](p4), self.cv4[1](p4),
            self.cv5[0](p5), self.cv5[1](p5),
        ]

# ---- ClassifierHead ----
class ClassifierHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1280, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, NC))
        for m in self.modules():
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight)
    def forward(self, x): return self.net(x)

# ---- Teacher ----
class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ViTBackbone()
        self.neck = LightFPN()
        self.head = PredHead()
        self.cls_head = ClassifierHead()
        self.nc = NC; self.nl = NL; self.na = NA; self.reg_max = REG_MAX
        self.no = NO; self.stride = [8, 16, 32]
    def forward(self, x):
        feats = self.backbone(x)
        neck = self.neck(feats)
        preds = self.head(neck)  # 6 tensors: [cv3a, cv3b, cv4a, cv4b, cv5a, cv5b]
        pooled = torch.cat([F.adaptive_avg_pool2d(f, 1).flatten(1) for f in neck], 1)
        return {"det": preds, "pooled": pooled, "cls_logits": self.cls_head(pooled)}

teacher = Teacher().to(DEVICE)
print(f"Teacher: {sum(p.numel() for p in teacher.parameters())/1e6:.1f}M params")

# Test forward
teacher.eval()
with torch.no_grad():
    d = torch.randn(2, 3, IMGSZ, IMGSZ).to(DEVICE)
    o = teacher(d)
    for i, x in enumerate(o["det"]):
        print(f"  Pred head {i}: {x.shape}")
    print(f"  pooled: {o['pooled'].shape}, cls_logits: {o['cls_logits'].shape}")

# ---- EMA ----
class EMA:
    def __init__(self, model, decay=EMA_DECAY):
        self.model = model; self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}
    def update(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad: self.shadow[n].mul_(self.decay).add_(p.data, alpha=1-self.decay)
    def apply(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad: self.backup[n]=p.data.clone(); p.data.copy_(self.shadow[n])
    def restore(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad: p.data.copy_(self.backup[n])
        self.backup.clear()
ema = EMA(teacher)

# ---- Detection Loss (self-contained) ----
def bbox_iou(b1, b2, CIoU=True):
    x1=b1[:,0]-b1[:,2]/2;y1=b1[:,1]-b1[:,3]/2;x2=b1[:,0]+b1[:,2]/2;y2=b1[:,1]+b1[:,3]/2
    X1=b2[:,0]-b2[:,2]/2;Y1=b2[:,1]-b2[:,3]/2;X2=b2[:,0]+b2[:,2]/2;Y2=b2[:,1]+b2[:,3]/2
    inter=(torch.min(x2,X2)-torch.max(x1,X1)).clamp(0)*(torch.min(y2,Y2)-torch.max(y1,Y1)).clamp(0)
    union=b1[:,2]*b1[:,3]+b2[:,2]*b2[:,3]-inter+1e-7; iou=inter/union
    if not CIoU: return iou
    cw=torch.max(x2,X2)-torch.min(x1,X1);ch=torch.max(y2,Y2)-torch.min(y1,Y1)
    c2=cw**2+ch**2+1e-7; rho2=((b1[:,:2]-b2[:,:2])**2).sum(1)
    v=(4/(math.pi**2))*((b2[:,3]/(b2[:,2]+1e-7)).atan()-(b1[:,3]/(b1[:,2]+1e-7)).atan())**2
    alpha=v/(1-iou+v+1e-7); return iou-(rho2/c2+alpha*v)

def compute_det_loss(preds, batch, teacher):
    det = preds["det"]; device = batch["cls"].device; B = len(batch["img"])
    gtc = batch["cls"].squeeze(-1); gtb = batch["bboxes"]; bidx = batch["batch_idx"].long()
    nc, na, reg = teacher.nc, teacher.na, teacher.reg_max
    
    # Group predictions: 2 branches per scale, 3 scales = 6 preds
    # Each pred: (B, NA*NO, H, W)
    scales_pred = []
    scales_stride = []
    for si in range(3):
        a = det[si*2]     # branch a: (B, NA*NO, H, W)
        b = det[si*2+1]   # branch b: (B, NA*NO, H, W)
        h, w = a.shape[2], a.shape[3]
        # Reshape to (B, NA, NO, H, W) -> (B, NA*H*W, NO)
        a = a.view(B, na, NO, h, w).permute(0,1,3,4,2).reshape(B, -1, NO)
        b = b.view(B, na, NO, h, w).permute(0,1,3,4,2).reshape(B, -1, NO)
        scales_pred.append((a, b))
        scales_stride.append(teacher.stride[si])
    
    box_l = cls_l = torch.tensor(0.0, device=device)
    np_ = 0
    
    for si, ((pa, pb), s_val) in enumerate(zip(scales_pred, scales_stride)):
        # Use branch a for scoring, branch b for box regression (YOLO convention)
        scores_a = pa[:, :, :nc]  # (B, N, 25)
        scores_b = pb[:, :, :nc]
        box_a = pa[:, :, nc:]     # (B, N, 64)
        box_b = pb[:, :, nc:]
        
        # Average scores from both branches
        scores = (scores_a + scores_b) / 2
        
        # Decode boxes from branch b (regression branch)
        NA_s = box_b.shape[1]
        pdist = box_b.view(B, NA_s, 4, reg).softmax(-1)
        w_idx = torch.arange(reg, device=device).float()
        lt = (pdist[:,:,:2]*w_idx).sum(-1)
        rb = (pdist[:,:,2:]*w_idx).sum(-1)
        
        # Anchor centers in pixel coords
        h_s, w_s = (IMGSZ // s_val, IMGSZ // s_val) if si != 0 else (80, 80)
        if h_s != pa.shape[1]//na:
            h_s = int(math.sqrt(pa.shape[1]//na))
            w_s = h_s
        sy, sx = torch.meshgrid(torch.arange(h_s, device=device), torch.arange(w_s, device=device), indexing='ij')
        anchor_xy = torch.stack((sx.float()+0.5, sy.float()+0.5),-1).view(-1,2)*s_val  # (h*w, 2)
        anchor_xy = anchor_xy.unsqueeze(0).unsqueeze(1).expand(B, na, -1, -1).reshape(B, -1, 2)  # (B, NA*N, 2)
        
        x1y1 = anchor_xy - lt * s_val
        x2y2 = anchor_xy + rb * s_val
        pbox = torch.cat([(x1y1+x2y2)/2, x2y2-x1y1], -1)
        
        for b in range(B):
            m = bidx == b
            if not m.any(): continue
            for j in range(m.sum()):
                cid = gtc[m][j].long().item()
                if not (0 <= cid < nc): continue
                gt_xywh = gtb[m][j].clone()
                gt_xywh[0] *= IMGSZ; gt_xywh[1] *= IMGSZ
                gt_xywh[2] *= IMGSZ; gt_xywh[3] *= IMGSZ
                
                # Simple: match by nearest grid cell
                gx = (gt_xywh[0] / s_val).long().clamp(0, w_s-1)
                gy = (gt_xywh[1] / s_val).long().clamp(0, h_s-1)
                for a_idx in range(na):
                    anchor_i = (a_idx * h_s * w_s + gy * w_s + gx).long()
                    if anchor_i >= NA_s: continue
                    
                    gt_exp = gt_xywh.unsqueeze(0)
                    box_l = box_l + (1.0 - bbox_iou(pbox[b, anchor_i:anchor_i+1], gt_exp, CIoU=True)).squeeze()
                    tgt = torch.zeros(nc, device=device)
                    tgt[cid] = 1.0
                    cls_l = cls_l + F.binary_cross_entropy_with_logits(scores[b, anchor_i:anchor_i+1], tgt.unsqueeze(0))
                    np_ += 1
    
    if np_ > 0:
        box_l = box_l * 7.5 / np_
        cls_l = cls_l * 0.5 / np_
    return box_l + cls_l, (box_l.item(), cls_l.item(), 0.0)

# ---- Data ----
with open(f"{BASE}/split_dataset/dataset.yaml") as f: cfg = yaml.safe_load(f)
img_dir = cfg["path"] + "/images/train"
lab_dir = cfg["path"] + "/labels/train"
all_imgs = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))])
print(f"Images: {len(all_imgs)}")

def load_one(fname, aug=True):
    ip = os.path.join(img_dir, fname)
    lp = os.path.join(lab_dir, fname.rsplit(".",1)[0]+".txt")
    img = cv2.imread(ip)
    if img is None: return None
    h0, w0 = img.shape[:2]
    sc = IMGSZ * random.uniform(0.75, 1.25) / max(h0, w0) if aug else IMGSZ / max(h0, w0)
    sc = min(sc, IMGSZ / max(h0, w0))
    nh, nw = int(h0 * sc), int(w0 * sc)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = IMGSZ - nw, IMGSZ - nh
    if dw < 0: dw = 0
    if dh < 0: dh = 0
    top, left = dh // 2, dw // 2
    img = cv2.copyMakeBorder(img, top, dh-top, left, dw-left, cv2.BORDER_CONSTANT, value=(114,114,114))
    flip = aug and random.random() < 0.5
    if flip: img = cv2.flip(img, 1)
    img_t = torch.from_numpy(img).float().permute(2,0,1) / 255.0
    cls, box = [], []
    if os.path.exists(lp):
        try:
            for line in open(lp).read().strip().splitlines():
                p = line.strip().split()
                if len(p) < 5: continue
                cid = int(p[0])
                if not (0 <= cid < NC): continue
                bx, by, bw, bh = [float(x) for x in p[1:5]]
                bx = (bx * w0 * sc + left) / IMGSZ
                by = (by * h0 * sc + top) / IMGSZ
                bw = bw * w0 * sc / IMGSZ
                bh = bh * h0 * sc / IMGSZ
                if flip: bx = 1.0 - bx
                cls.append(cid)
                box.append([max(0,bx), max(0,by), min(bw,1.0), min(bh,1.0)])
        except: pass
    if cls:
        return img_t, torch.tensor(cls, dtype=torch.float32), torch.tensor(box, dtype=torch.float32), len(cls)
    return img_t, torch.zeros(0, dtype=torch.float32), torch.zeros(0, 4, dtype=torch.float32), 0

# ---- Optimizer ----
def setup_opt(ep):
    if ep < PHASE1_END:
        for p in teacher.backbone.vit.parameters(): p.requires_grad = False
        params = [p for n,p in teacher.named_parameters() if not n.startswith("backbone.vit")]
        return torch.optim.AdamW(params, lr=1e-3, weight_decay=5e-4)
    elif ep < PHASE2_END:
        unf = list(teacher.backbone.vit.blocks[-6:].parameters())
        for p in teacher.backbone.vit.parameters(): p.requires_grad = False
        for p in unf: p.requires_grad = True
        nh = [p for n,p in teacher.named_parameters() if not n.startswith("backbone.vit")]
        return torch.optim.AdamW([{"params":nh,"lr":3e-4},{"params":unf,"lr":1e-4}], weight_decay=5e-4)
    else:
        for p in teacher.parameters(): p.requires_grad = True
        return torch.optim.AdamW(teacher.parameters(), lr=1e-4, weight_decay=5e-4)

# ---- Train ----
SDIR = Path(f"{BASE}/runs/teacher_v4/weights")
SDIR.mkdir(parents=True, exist_ok=True)
best_l, st_ep = float("inf"), 0
if (SDIR/"state.pt").exists():
    st = torch.load(SDIR/"state.pt", map_location=DEVICE, weights_only=False)
    teacher.load_state_dict(st["model"]); st_ep = st.get("epoch",0); best_l = st.get("best_loss",float("inf"))
    print(f"Resumed epoch {st_ep}")

scaler = torch.amp.GradScaler("cuda")

for ep in range(st_ep, EPOCHS):
    if ep == st_ep or ep == 0 or ep == PHASE1_END or ep == PHASE2_END:
        opt = setup_opt(ep)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2, eta_min=1e-6)
        ph = 1 if ep < PHASE1_END else 2 if ep < PHASE2_END else 3
        print(f"\nPhase {ph}: lr={opt.param_groups[0]['lr']:.2e}")

    random.shuffle(all_imgs); teacher.train(); opt.zero_grad()
    tloss, nbatch = 0.0, 0; t0 = time.time()

    for i in tqdm(range(0, len(all_imgs), BATCH), desc=f"E{ep+1}/{EPOCHS}", ncols=90):
        bf = all_imgs[i:i+BATCH]; il, cl, bl = [], [], []
        for f in bf:
            r = load_one(f, True)
            if r is None: continue
            il.append(r[0]); cl.append(r[1]); bl.append(r[2])
        if len(il) < 2: continue
        im = torch.stack(il).to(DEVICE); Ba = len(il)
        bc = torch.cat(cl,0).to(DEVICE) if cl else torch.zeros(0,device=DEVICE)
        bb = torch.cat(bl,0).to(DEVICE) if bl else torch.zeros(0,4,device=DEVICE)
        bd = torch.cat([torch.full((len(c),), j, device=DEVICE) for j,c in enumerate(cl)],0) if cl else torch.zeros(0,device=DEVICE)
        batch = {"img":im, "cls":bc.unsqueeze(-1), "batch_idx":bd, "bboxes":bb}

        with torch.amp.autocast("cuda"):
            out = teacher(im)
            dl, _ = compute_det_loss(out, batch, teacher)
            clsl = torch.tensor(0.0, device=DEVICE)
            if all(c.numel()>0 for c in cl):
                tgt = torch.zeros(Ba, NC, device=DEVICE)
                for j,c in enumerate(cl):
                    if c.numel()>0:
                        for u in c.long().unique():
                            if 0<=int(u)<NC: tgt[j,int(u)]=1.0
                tgt = tgt*(1-LABEL_SMOOTH)+LABEL_SMOOTH/NC
                clsl = F.binary_cross_entropy_with_logits(out["cls_logits"], tgt)
            loss = dl + 0.3*clsl

        loss = loss/ACCUM; scaler.scale(loss).backward()

        if (nbatch+1)%ACCUM==0 or i+BATCH>=len(all_imgs):
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), 10.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(); ema.update()

        tloss += loss.item()*ACCUM; nbatch += 1
        del im, out, loss, dl, clsl

    sched.step(); avg = tloss/max(nbatch,1)
    print(f"  loss={avg:.4f} time={time.time()-t0:.0f}s")

    torch.save({"model":teacher.state_dict(),"epoch":ep+1,"best_loss":best_l}, str(SDIR/"state.pt"))
    if avg < best_l:
        best_l = avg; ema.apply(); torch.save(teacher.state_dict(), str(SDIR/"best.pt")); ema.restore()
        print(f"  *** BEST ({best_l:.4f}) ***")
    ema.apply(); torch.save(teacher.state_dict(), str(SDIR/"last.pt")); ema.restore()

print(f"\nDone! {SDIR}/best.pt (loss={best_l:.4f})")