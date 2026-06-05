#!/bin/bash
# AutoDL 一键训练脚本 - Sentence-Transformers版
# 用法: bash run_autodl.sh [mode]
# mode: text (默认) | audio | full

set -e

MODE=${1:-text}

echo "========================================"
echo "Civis Lucri-Faber Training"
echo "Mode: $MODE"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'CPU')"
echo "========================================"

# 安装依赖 (固定兼容版本)
echo "[1/3] Installing dependencies..."
pip install -q 'transformers==4.51.0' 'sentence-transformers==3.3.1' datasets soundfile tqdm python-dotenv 2>&1 | tail -3

# 验证GPU
echo "[2/3] Checking GPU..."
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# 训练
echo "[3/3] Training..."
if [ "$MODE" = "text" ]; then
    python train_real.py --epochs 100 --batch_size 32 --lr 0.001 --mode full 2>&1 | tee training_log.txt
elif [ "$MODE" = "audio" ]; then
    python train_audio.py --epochs 30 --batch_size 16 --lr 0.0005 2>&1 | tee training_log.txt
elif [ "$MODE" = "full" ]; then
    echo "=== Text Training ==="
    python train_real.py --epochs 100 --batch_size 32 --lr 0.001 --mode full 2>&1 | tee training_log_text.txt
    echo "=== Audio Training ==="
    python train_audio.py --epochs 30 --batch_size 16 --lr 0.0005 2>&1 | tee training_log_audio.txt
fi

echo ""
echo "========================================"
echo "Training complete!"
echo "Results in: checkpoints/"
echo "========================================"
