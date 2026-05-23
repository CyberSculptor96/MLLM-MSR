#!/bin/bash
# =============================================================
# RoDPO Conda Environment Setup (v2 - 预编译 wheel)
# =============================================================
set -e

export https_proxy=http://agent.baidu.com:8891

CONDA=/root/miniforge3/bin/conda
ENV_NAME=rodpo
PYTHON=/root/miniforge3/envs/${ENV_NAME}/bin/python
PIP="/root/miniforge3/envs/${ENV_NAME}/bin/pip"

echo "========================================"
echo "Creating ${ENV_NAME} conda environment..."
echo "========================================"

# Step 1: Create conda env (如果已存在则跳过)
if [ -d "/root/miniforge3/envs/${ENV_NAME}" ]; then
    echo "[1/6] Conda env already exists, skipping creation."
else
    ${CONDA} create -n ${ENV_NAME} python=3.11 -y
    echo "[1/6] Conda env created."
fi

echo "  Python: ${PYTHON}"
${PYTHON} --version

# Step 2: PyTorch with CUDA 12.1
echo "[2/6] Installing PyTorch 2.4.0 + cu121..."
${PIP} install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121 -q 2>&1 | tail -3
echo "  PyTorch installed."

# Step 3: Transformers ecosystem
echo "[3/6] Installing transformers + peft + accelerate..."
${PIP} install transformers==4.44.0 accelerate==0.33.0 peft==0.12.0 bitsandbytes==0.43.1 -q 2>&1 | tail -3
echo "  Transformers ecosystem installed."

# Step 4: Lightning + DeepSpeed
echo "[4/6] Installing lightning + deepspeed..."
${PIP} install lightning==2.4.0 deepspeed==0.14.4 -q 2>&1 | tail -3
echo "  Lightning + DeepSpeed installed."

# Step 5: Utilities
echo "[5/6] Installing utilities..."
${PIP} install pillow scikit-learn tqdm pandas numpy scipy -q 2>&1 | tail -3
echo "  Utilities installed."

# Step 6: Flash Attention (预编译 wheel)
echo "[6/6] Installing flash-attn (precompiled wheel)..."

# torch 2.4 + python 3.11 + cu12 → flash-attn 2.6.3
# wheel URL 格式: flash_attn-{ver}+cu122torch2.4cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
FLASH_ATTN_VERSION="2.6.3"
FLASH_ATTN_WHEEL="flash_attn-${FLASH_ATTN_VERSION}+cu122torch2.4cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
FLASH_ATTN_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VERSION}/${FLASH_ATTN_WHEEL}"

echo "  URL: ${FLASH_ATTN_URL}"
${PIP} install "${FLASH_ATTN_URL}" -q 2>&1 | tail -3 || {
    echo "  [WARN] Precompiled wheel failed, trying pip install from source..."
    ${PIP} install flash-attn==2.6.3 --no-build-isolation -q 2>&1 | tail -3 || {
        echo "  [WARN] flash-attn install failed entirely. Will use sdpa attention."
    }
}
echo "  flash-attn step done."

# Verification
echo ""
echo "========================================"
echo "Verification:"
echo "========================================"
${PYTHON} -c "
import torch
import transformers
import peft
import lightning
import deepspeed
print(f'  torch:          {torch.__version__}')
print(f'  transformers:   {transformers.__version__}')
print(f'  peft:           {peft.__version__}')
print(f'  lightning:      {lightning.__version__}')
print(f'  deepspeed:      {deepspeed.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  GPU count:      {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'  GPU 0:          {torch.cuda.get_device_name(0)}')
try:
    from flash_attn import flash_attn_func
    print(f'  flash-attn:     OK')
except ImportError:
    print(f'  flash-attn:     NOT available (will use sdpa)')
"

echo ""
echo "========================================"
echo "DONE! Environment ready."
echo "Python: ${PYTHON}"
echo "Activate: conda activate ${ENV_NAME}"
echo "========================================"
