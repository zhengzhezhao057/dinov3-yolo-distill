#!/bin/bash
set -e
HOME="${DISTILL_HOME:-$(cd "$(dirname "$0")" && pwd)}"
echo "DISTILL_HOME=$HOME"

echo "[1/5] Installing packages..."
pip install -r requirements.txt -q

echo "[2/5] DINOv3..."
if [ ! -d "$HOME/dinov3_repo" ]; then
    git clone --depth 1 https://github.com/facebookresearch/dinov3.git "$HOME/dinov3_repo"
fi

echo "[3/5] Weights..."
mkdir -p "$HOME/weights"
if [ ! -f "$HOME/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth" ]; then
    echo "  ViT-L (1.2GB)..."
    huggingface-cli download facebook/dinov3-vitl16-pretrain-sat493m --local-dir "$HOME/weights"
    find "$HOME/weights" -name "*.pth" -exec mv {} "$HOME/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth" \;
fi
if [ ! -f "$HOME/weights/yolo11m.pt" ]; then
    echo "  YOLO11m..."
    wget -q -O "$HOME/weights/yolo11m.pt" https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt
fi

echo "[4/5] Dataset..."
if [ ! -d "$HOME/data" ]; then
    echo "  Place dataset in $HOME/data/"
    echo "  data/images/train, data/labels/train, data/dataset.yaml"
fi

echo "[5/5] Verify..."
python scripts/verify.py
echo "Done! python scripts/train_teacher.py"
