#!/bin/bash
# AutoDL部署打包脚本
# 在本地 (Git Bash) 执行: bash pack_for_autodl.sh

set -e
echo "=== Civis Lucri-Faber AutoDL 打包 ==="

# 排除不需要的文件
tar -czvf civis_lucri_faber.tar.gz \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.claude' \
  --exclude='electron_app' \
  --exclude='*.vmdk' \
  --exclude='*.mp3' \
  --exclude='*.lrc' \
  --exclude='checkpoints' \
  --exclude='autodl_checkpoints' \
  --exclude='.anaconda' \
  --exclude='.conda' \
  --exclude='.cache' \
  --exclude='.vscode' \
  --exclude='.cursor' \
  --exclude='.idea' \
  core/ \
  data/ \
  train_real.py \
  train_autodl.py \
  tests/ \
  docs/ \
  *.py \
  *.md \
  requirements.txt

echo ""
echo "=== 打包完成 ==="
ls -lh civis_lucri_faber.tar.gz
echo ""
echo "上传到AutoDL后执行:"
echo "  cd /root"
echo "  tar -xzvf civis_lucri_faber.tar.gz -C civis_lucri_faber"
echo "  cd civis_lucri_faber"
echo "  pip install -r requirements.txt"
echo "  python train_real.py --mode full --epochs 50 --batch_size 32"
