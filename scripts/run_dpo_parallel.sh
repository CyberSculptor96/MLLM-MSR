#!/bin/bash
# =============================================================
# DPO Training: argmax (hard) + top_k in parallel
# 4 GPUs each, with pre-computed SFT score cache (no warm-up epoch)
# =============================================================
export PATH=/usr/bin:/usr/local/bin:/bin:$PATH
export https_proxy=http://agent.baidu.com:8891

PYTHON=/root/miniforge3/envs/rodpo/bin/python
PROJECT_DIR=/root/paddlejob/workspace/codelab/projects/rodpo/MLLM-MSR
TRAIN_SCRIPT=${PROJECT_DIR}/train/dpo/train_llava_dpo.py
MODEL_PATH=${PROJECT_DIR}/models/llava-v1.6-mistral-7b-hf
SFT_CKPT=${PROJECT_DIR}/checkpoints/sft_microlens/epoch_0
SCORE_CACHE=${PROJECT_DIR}/checkpoints/score_cache_sft.json
LOG_DIR=${PROJECT_DIR}/logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p ${LOG_DIR}
cd ${PROJECT_DIR}/train/dpo

echo "============================================"
echo "DPO Training: argmax + top_k (parallel)"
echo "  GPUs 0-3: argmax (hard)"
echo "  GPUs 4-7: top_k (stochastic)"
echo "  Score cache: ${SCORE_CACHE}"
echo "  Epochs: 2 (no warm-up needed)"
echo "  Start: $(date)"
echo "============================================"

# --- Group 1: Hard (argmax) on GPU 0,1,2,3 ---
echo ""
echo ">>> Launching DPO [hard/argmax] on cuda:0,1,2,3..."
CUDA_VISIBLE_DEVICES=0,1,2,3 ${PYTHON} ${TRAIN_SCRIPT} \
    --strategy hard \
    --dataset microlens \
    --model_path ${MODEL_PATH} \
    --sft_lora_path ${SFT_CKPT} \
    --devices 4 \
    --epochs 2 \
    --cache_max_users 2000 \
    --score_cache_path ${SCORE_CACHE} \
    --skip_cache_refresh \
    > ${LOG_DIR}/dpo_hard_${TIMESTAMP}.log 2>&1 &
PID_HARD=$!
echo "  PID: ${PID_HARD}, Log: ${LOG_DIR}/dpo_hard_${TIMESTAMP}.log"

# --- Group 2: Top-K on GPU 4,5,6,7 ---
echo ""
echo ">>> Launching DPO [top_k] on cuda:4,5,6,7..."
CUDA_VISIBLE_DEVICES=4,5,6,7 ${PYTHON} ${TRAIN_SCRIPT} \
    --strategy top_k \
    --dataset microlens \
    --model_path ${MODEL_PATH} \
    --sft_lora_path ${SFT_CKPT} \
    --devices 4 \
    --epochs 2 \
    --cache_max_users 2000 \
    --score_cache_path ${SCORE_CACHE} \
    --skip_cache_refresh \
    > ${LOG_DIR}/dpo_top_k_${TIMESTAMP}.log 2>&1 &
PID_TOPK=$!
echo "  PID: ${PID_TOPK}, Log: ${LOG_DIR}/dpo_top_k_${TIMESTAMP}.log"

echo ""
echo "============================================"
echo "Both DPO experiments launched!"
echo "  Hard  PID: ${PID_HARD}"
echo "  Top-K PID: ${PID_TOPK}"
echo ""
echo "Monitor:"
echo "  tail -f ${LOG_DIR}/dpo_hard.log"
echo "  tail -f ${LOG_DIR}/dpo_top_k.log"
echo "  tensorboard --logdir ${PROJECT_DIR}/tb_logs"
echo "============================================"

# Wait for both to complete
wait ${PID_HARD} ${PID_TOPK}

echo ""
echo "============================================"
echo "Both DPO experiments finished! $(date)"
echo "============================================"
