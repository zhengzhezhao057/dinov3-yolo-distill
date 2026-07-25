# DINOv3 → YOLO11m 知识蒸馏 (25类遥感目标检测)

将 DINOv3 ViT-L 的知识蒸馏到 YOLO11m，提升细粒度遥感军事目标检测。

**GPU**: RTX 3090/4090 24GB | **PyTorch**: 2.1+ | **Python**: 3.10+

---

## 环境要求

| 组件 | 要求 |
|------|------|
| GPU | RTX 3090/4090 24GB |
| PyTorch | 2.1+ (CUDA 11.8/12.x) |
| Python | 3.10+ |
| 系统 | Ubuntu 20.04/22.04 |
| 硬盘 | ~10GB（含数据集和权重） |

---

## 快速开始

### 1. 克隆仓库

`ash
git clone https://github.com/zhengzhezhao057/dinov3-yolo-distill.git
cd dinov3-yolo-distill
`

### 2. 安装依赖

`ash
# 基础包
pip install ultralytics opencv-python-headless tqdm pyyaml numpy timm -q

# DINOv3 源码
git clone --depth 1 https://github.com/facebookresearch/dinov3.git dinov3_repo
`

### 3. 下载权重

`ash
mkdir -p weights

# YOLO11m (40MB)
wget -O weights/yolo11m.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt

# DINOv3 ViT-L (1.2GB) — 需要先登录 huggingface.co 申请访问
pip install huggingface_hub -q
hf download facebook/dinov3-vitl16-pretrain-sat493m --local-dir weights/
`

> ⚠️ 如果 HuggingFace 下载失败，从本地上传：
> scp -P <端口> dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth root@<IP>:/root/dinov3-yolo-distill/weights/

### 4. 准备数据集

目录结构：
`
data/
├── dataset.yaml
├── images/train/    # .jpg/.png
└── labels/train/    # YOLO格式 .txt
`

**dataset.yaml**:
`yaml
path: ./data
train: images/train
nc: 25
names: ["HM","LQS","QHS","MS","A1_SU-35","A2_C-130","A3_C-17","A4_C-5",
        "A5_F-16","A6_TU-160","A7_E-3","A8_B-52","A9_P-3C","A10_B-1B",
        "A11_E-8","A12_TU-22","A13_F-15","A14_KC-135","A15_F-22",
        "A16_FA-18","A17_TU-95","A18_KC-10","A19_SU-34","A20_SU-24","FSC"]
`

上传方式（推荐压缩后上传）：
`powershell
# 本地压缩 (Windows PowerShell)
cd E:\deeplearning\split_dataset\split_dataset
tar -czf E:\deeplearning\split_dataset\dataset.tar.gz images labels dataset.yaml

# SCP 上传
scp -P <端口> E:\deeplearning\split_dataset\dataset.tar.gz root@<IP>:/root/dinov3-yolo-distill/data/
`

服务器解压：
`ash
cd /root/dinov3-yolo-distill/data && tar -xzf dataset.tar.gz
`

### 5. 验证

`ash
python scripts/verify.py
`

看到 ALL OK! 即可开始训练。

---

## 训练流程

| 步骤 | 命令 | 时间 | 说明 |
|------|------|------|------|
| ① | python scripts/train_teacher.py | ~10h | 训练教师 (100 epochs) |
| ② | python scripts/extract_signals.py | ~20min | 提取教师信号 |
| ③ | python scripts/train_distill.py | ~4h | 蒸馏 YOLO11m (50 epochs) |
| ④ | python scripts/train_finetune.py | ~1h | 关闭蒸馏微调 (10 epochs) |

长时间训练建议 
ohup：
`ash
nohup python scripts/train_teacher.py > logs/teacher.log 2>&1 &
tail -f logs/teacher.log
`

---

## 蒸馏方案

`
教师 (DINOv3 ViT-L 303M)                学生 (YOLO11m 20M)
┌──────────────────────────┐           ┌──────────────────┐
│ ViT-L → ViTBackbone      │    MSE    │ YOLO Backbone    │
│         → LightFPN       │←─────────→│ → YOLO Neck      │
│         → PredHead  Cls  │    KL     │ → YOLO Head      │
└──────────────────────────┘←─────────→└──────────────────┘
                                      L = L_det + α·MSE + β·KL

α(t): 0.5→0.1   β(t): 0.3→0.7   KL T=3.0
`

### 教师训练策略

| 阶段 | Epoch | 策略 | LR |
|------|-------|------|-----|
| Phase 1 | 1-40 | 冻结 ViT，训练 Neck+Head | 1e-3 |
| Phase 2 | 41-70 | 解冻最后6个block | 3e-4/1e-4 |
| Phase 3 | 71-100 | 全部解冻 | 1e-4 |

---

## 配置说明

所有路径集中在 config.py。更换目录只需设置环境变量：

`ash
export DISTILL_HOME=/你的自定义路径
`

默认就是 repo 所在目录。

---

## 常见问题

- **HuggingFace 下载失败？** → 登录申请权限，或 SCP 本地上传
- **OOM？** → 减小 BATCH，增大 ACCUM 保持乘积不变
- **No module named 'dinov3'？** → 确认 dinov3_repo/ 已克隆
- **多服务器迁移？** → clone 仓库 + 上传 weights/ 和 data/ 即可

---

## 项目结构

`
dinov3-yolo-distill/
├── config.py              # 全局路径和超参
├── requirements.txt       # Python 依赖
├── README.md
├── scripts/
│   ├── verify.py          # 环境验证
│   ├── train_teacher.py   # ① 教师训练
│   ├── extract_signals.py # ② 提取信号
│   ├── train_distill.py   # ③ 蒸馏
│   └── train_finetune.py  # ④ 微调
├── dinov3_repo/           # DINOv3 源码 (git clone 生成)
├── weights/               # 权重 (手动下载)
├── data/                  # 数据集 (手动准备)
├── features/              # 教师信号 (自动生成)
└── runs/                  # 训练输出 (自动生成)
`

## License

MIT
