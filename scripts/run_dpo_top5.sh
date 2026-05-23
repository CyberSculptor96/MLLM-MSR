#!/bin/bash
# DPO Training: top_k with K=5 (true stochastic hard negative)
# 4 GPUs, pre-computed SFT score cache
export PATH=/usr/bin:/usr/local/bin:/bin:$PATH
export https_proxy=http://agent.baidu.com:8891

PYTHON=/root/miniforge3/envs/rodpo/bin/python
PROJECT_DIR=/root/paddlejob/workspace/codelab/projects/rodpo/MLLM-MSR
TRAIN_SCRIPT=${PROJECT_DIR}/train/dpo/train_llava_dpo.py
MODEL_PATH=${PROJECT_DIR}/models/llava-v1.6-mistral-7b-hf
SFT_CKPT=${PROJECT_DIR}/checkpoints/sft_microlens/epoch_0
SCORE_CACHE=${PROJECT_DIR}/checkpoints/score_cache_sft.json
LOG_DIR=${PROJECT_DIR}/logs

mkdir -p ${LOG_DIR}
cd ${PROJECT_DIR}/train/dpo

echo "============================================"
echo "DPO Training: top_k K=5"
echo "  GPUs: 0,1,2,3"
echo "  Score cache: ${SCORE_CACHE}"
echo "  Start: $(date)"
echo "============================================"

CUDA_VISIBLE_DEVICES=0,1,2,3 ${PYTHON} ${TRAIN_SCRIPT} \
    --strategy top_k \
    --top_k 5 \
    --dataset microlens \
    --model_path ${MODEL_PATH} \
    --sft_lora_path ${SFT_CKPT} \
    --devices 4 \
    --epochs 2 \
    --cache_max_users 2000 \
    --score_cache_path ${SCORE_CACHE}

echo "============================================"
echo "Done! $(date)"
echo "============================================"
