#!/bin/bash
# =============================================================
# Qwen2.5-VL SFT Training - MicroLens Dataset
# 预计时间: ~3-4h (100k samples × 2 epochs, 4x H800, bs=1, acc=4)
# =============================================================
export https_proxy=http://agent.baidu.com:8891
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

PYTHON=/root/miniforge3/envs/rodpo/bin/python
BASE_DIR=/root/paddlejob/workspace/codelab/projects/rodpo/MLLM-MSR
TRAIN_DIR=${BASE_DIR}/train/dpo
MODEL_PATH=${BASE_DIR}/models/Qwen2.5-VL-7B-Instruct
LOG_DIR=${BASE_DIR}/logs
LOG_FILE=${LOG_DIR}/qwen_sft_$(date +%Y%m%d_%H%M%S).log

mkdir -p ${LOG_DIR}

echo "======================================"
echo "Qwen2.5-VL SFT Training - MicroLens"
echo "Start: $(date)"
echo "Log: ${LOG_FILE}"
echo "======================================"

cd ${TRAIN_DIR}
${PYTHON} train_qwen_sft.py \
    --dataset microlens \
    --model_path ${MODEL_PATH} \
    --devices 8 \
    --epochs 2 \
    2>&1 | tee ${LOG_FILE}

echo ""
echo "======================================"
echo "Qwen2.5-VL SFT Training Complete!"
echo "End: $(date)"
echo "Checkpoints: ${BASE_DIR}/checkpoints/qwen_sft_microlens/"
echo "======================================"
