#!/bin/bash
# =============================================================
# 32-Card Cluster Setup Script
# 在新集群上一键拉取代码 + 数据 + 重建环境
# =============================================================
set -e

export https_proxy=http://agent.baidu.com:8891
BOS_BASE="bos:/nlp-data-app-models/huanghj/MLLM-MSR"
WORK_DIR="/root/paddlejob/workspace/codelab/projects/rodpo"

echo "============================================"
echo "MLLM-MSR Cluster Setup"
echo "============================================"

# --- Step 1: Pull code from GitHub ---
echo "[1/4] Cloning code from GitHub..."
mkdir -p ${WORK_DIR}
cd ${WORK_DIR}
if [ -d "MLLM-MSR/.git" ]; then
    echo "  Repo exists, pulling latest..."
    cd MLLM-MSR && git pull && cd ..
else
    git clone https://github.com/CyberSculptor96/MLLM-MSR.git
fi
echo "  Done."

# --- Step 2: Configure bcecmd ---
echo "[2/4] Configuring bcecmd..."
export PATH=$PATH:/root/paddlejob/workspace/env_run/linux-bcecmd-0.5.10
chmod +x /root/paddlejob/workspace/env_run/linux-bcecmd-0.5.10/bcecmd
rm -f /root/.go-bcecli/credentials /root/.bcecmd/config

# Replace YOUR_AK and YOUR_SK below with actual credentials
printf "<YOUR_AK>\n<YOUR_SK>\n\nbj\nbj.bcebos.com\nyes\n\n7\nyes\n10\n10\n10\n12\nno\n" | \
  bcecmd --configure > /dev/null 2>&1
echo "  Done."

# --- Step 3: Download data + models from BOS ---
echo "[3/4] Downloading from BOS..."
cd ${WORK_DIR}/MLLM-MSR

echo "  [3a] LLaVA model (15G)..."
mkdir -p models
bcecmd bos cp -r ${BOS_BASE}/models/llava-v1.6-mistral-7b-hf/ ./models/llava-v1.6-mistral-7b-hf/

echo "  [3b] MicroLens data..."
mkdir -p MLLM-MSR/data/microlens
bcecmd bos cp -r ${BOS_BASE}/data/microlens/dpo_ready/ ./MLLM-MSR/data/microlens/dpo_ready/
bcecmd bos cp -r ${BOS_BASE}/data/microlens/images/ ./MLLM-MSR/data/microlens/images/

echo "  [3c] Amazon data..."
mkdir -p MLLM-MSR/data/amazon
bcecmd bos cp -r ${BOS_BASE}/data/amazon/dpo_ready/ ./MLLM-MSR/data/amazon/dpo_ready/
bcecmd bos cp -r ${BOS_BASE}/data/amazon/images/ ./MLLM-MSR/data/amazon/images/

echo "  [3d] Checkpoints (if available)..."
bcecmd bos cp -r ${BOS_BASE}/checkpoints/ ./MLLM-MSR/train/dpo/checkpoints/ 2>/dev/null || echo "  No checkpoints on BOS yet."

echo "  Done."

# --- Step 4: Rebuild conda environment ---
echo "[4/4] Setting up conda environment..."
bash ${WORK_DIR}/MLLM-MSR/setup_env.sh
echo "  Done."

echo ""
echo "============================================"
echo "Setup Complete!"
echo "============================================"
echo "Next steps:"
echo "  1. Activate env: conda activate rodpo"
echo "  2. Run training: bash scripts/run_sft_microlens.sh"
echo "============================================"
