#!/bin/bash
# =============================================================
# Qwen2.5-VL DPO Training: Hard + Random (parallel, this machine)
# Step 1: Generate score cache (single GPU, ~30 min)
# Step 2: Launch Hard (GPU 0-3) + Random (GPU 4-7)
# =============================================================
export PATH=/usr/bin:/usr/local/bin:/bin:$PATH
export https_proxy=http://agent.baidu.com:8891

PYTHON=/root/miniforge3/envs/rodpo/bin/python
PROJECT_DIR=/root/paddlejob/workspace/codelab/projects/rodpo/MLLM-MSR
TRAIN_SCRIPT=${PROJECT_DIR}/train/dpo/train_qwen_dpo.py
MODEL_PATH=${PROJECT_DIR}/models/Qwen2.5-VL-7B-Instruct
SFT_CKPT=${PROJECT_DIR}/checkpoints/qwen_sft_microlens/epoch_0
SCORE_CACHE=${PROJECT_DIR}/checkpoints/qwen_score_cache_sft.json
LOG_DIR=${PROJECT_DIR}/logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p ${LOG_DIR}

echo "============================================"
echo "Qwen2.5-VL DPO: Hard + Random"
echo "  SFT base: ${SFT_CKPT}"
echo "  Start: $(date)"
echo "============================================"

# --- Step 1: Generate score cache (needed for hard strategy) ---
if [ ! -f "${SCORE_CACHE}" ]; then
    echo ""
    echo ">>> [Step 1] Generating score cache (GPU 0, ~30 min)..."
    cd ${PROJECT_DIR}/train/dpo
    CUDA_VISIBLE_DEVICES=0 ${PYTHON} generate_score_cache_qwen.py \
        --dataset microlens \
        --sft_lora_path ${SFT_CKPT} \
        --max_users 2000 \
        --batch_size 4 \
        --output qwen_score_cache_sft.json \
        2>&1 | tee ${LOG_DIR}/qwen_score_cache_${TIMESTAMP}.log

    if [ $? -ne 0 ]; then
        echo "ERROR: Score cache generation failed!"
        exit 1
    fi
    echo ">>> Score cache ready: ${SCORE_CACHE}"
else
    echo ">>> Score cache already exists: ${SCORE_CACHE}"
fi

# --- Step 2: Launch DPO training ---
echo ""
echo ">>> [Step 2] Launching DPO training..."
cd ${PROJECT_DIR}/train/dpo

# Group 1: Hard (argmax) on GPU 0,1,2,3
echo ""
echo ">>> Launching DPO [hard/argmax] on GPU 0,1,2,3..."
CUDA_VISIBLE_DEVICES=0,1,2,3 ${PYTHON} ${TRAIN_SCRIPT} \
    --config configs/qwen_hard.yaml \
    --dataset microlens \
    --model_path ${MODEL_PATH} \
    --sft_lora_path ${SFT_CKPT} \
    --devices 4 \
    --score_cache_path ${SCORE_CACHE} \
    --skip_cache_refresh \
    > ${LOG_DIR}/qwen_dpo_hard_${TIMESTAMP}.log 2>&1 &
PID_HARD=$!
echo "  PID: ${PID_HARD}, Log: ${LOG_DIR}/qwen_dpo_hard_${TIMESTAMP}.log"

# Group 2: Random (top_k=50) on GPU 4,5,6,7
echo ""
echo ">>> Launching DPO [random/top_k=50] on GPU 4,5,6,7..."
CUDA_VISIBLE_DEVICES=4,5,6,7 ${PYTHON} ${TRAIN_SCRIPT} \
    --config configs/qwen_random.yaml \
    --dataset microlens \
    --model_path ${MODEL_PATH} \
    --sft_lora_path ${SFT_CKPT} \
    --devices 4 \
    --skip_cache_refresh \
    > ${LOG_DIR}/qwen_dpo_random_${TIMESTAMP}.log 2>&1 &
PID_RANDOM=$!
echo "  PID: ${PID_RANDOM}, Log: ${LOG_DIR}/qwen_dpo_random_${TIMESTAMP}.log"

echo ""
echo "============================================"
echo "Both DPO experiments launched!"
echo "  Hard   PID: ${PID_HARD}"
echo "  Random PID: ${PID_RANDOM}"
echo ""
echo "Monitor:"
echo "  tail -f ${LOG_DIR}/qwen_dpo_hard_${TIMESTAMP}.log"
echo "  tail -f ${LOG_DIR}/qwen_dpo_random_${TIMESTAMP}.log"
echo "  tensorboard --logdir ${PROJECT_DIR}/tb_logs"
echo "============================================"

# Wait for both
wait ${PID_HARD} ${PID_RANDOM}

echo ""
echo "============================================"
echo "Both DPO experiments finished! $(date)"
echo "============================================"
