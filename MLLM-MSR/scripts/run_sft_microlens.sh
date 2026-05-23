#!/bin/bash
# =============================================================
# SFT Training - MicroLens Dataset
# 预计时间: ~4-6h (100k samples × 4 epochs, 8x H800, bs=1, acc=4)
# =============================================================
export https_proxy=http://agent.baidu.com:8891
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

PYTHON=/root/miniforge3/envs/rodpo/bin/python
TRAIN_DIR=/root/paddlejob/workspace/codelab/projects/rodpo/MLLM-MSR/MLLM-MSR/train/dpo
MODEL_PATH=/root/paddlejob/workspace/codelab/projects/rodpo/MLLM-MSR/models/llava-v1.6-mistral-7b-hf

echo "======================================"
echo "SFT Training - MicroLens"
echo "Start: $(date)"
echo "======================================"

cd ${TRAIN_DIR}
${PYTHON} train_llava_sft.py \
    --dataset microlens \
    --model_path ${MODEL_PATH} \
    --devices 8 \
    --epochs 4

echo ""
echo "======================================"
echo "SFT Training Complete!"
echo "End: $(date)"
echo "======================================"
