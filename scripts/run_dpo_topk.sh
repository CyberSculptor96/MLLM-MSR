#!/bin/bash
# DPO Training: top_k K=5 and K=10 in parallel
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
echo "DPO Training: top_k K=5 + K=10 (parallel)"
echo "  GPUs 0-3: top_k K=5"
echo "  GPUs 4-7: top_k K=10"
echo "  Start: $(date)"
echo "============================================"

# --- K=5 on GPU 0,1,2,3 ---
CUDA_VISIBLE_DEVICES=0,1,2,3 ${PYTHON} ${TRAIN_SCRIPT} \
    --strategy top_k \
    --top_k 5 \
    --dataset microlens \
    --model_path ${MODEL_PATH} \
    --sft_lora_path ${SFT_CKPT} \
    --devices 4 \
    --epochs 2 \
    --cache_max_users 2000 \
    --score_cache_path ${SCORE_CACHE} \
    > ${LOG_DIR}/dpo_topk5.log 2>&1 &
PID_K5=$!
echo "  K=5 PID: ${PID_K5}, Log: ${LOG_DIR}/dpo_topk5.log"

# --- K=10 on GPU 4,5,6,7 ---
CUDA_VISIBLE_DEVICES=4,5,6,7 ${PYTHON} ${TRAIN_SCRIPT} \
    --strategy top_k \
    --top_k 10 \
    --dataset microlens \
    --model_path ${MODEL_PATH} \
    --sft_lora_path ${SFT_CKPT} \
    --devices 4 \
    --epochs 2 \
    --cache_max_users 2000 \
    --score_cache_path ${SCORE_CACHE} \
    > ${LOG_DIR}/dpo_topk10.log 2>&1 &
PID_K10=$!
echo "  K=10 PID: ${PID_K10}, Log: ${LOG_DIR}/dpo_topk10.log"

echo ""
echo "Monitor:"
echo "  tail -f ${LOG_DIR}/dpo_topk5.log"
echo "  tail -f ${LOG_DIR}/dpo_topk10.log"
echo "============================================"

wait ${PID_K5} ${PID_K10}

echo ""
echo "============================================"
echo "Both top_k experiments finished! $(date)"
echo "============================================"
