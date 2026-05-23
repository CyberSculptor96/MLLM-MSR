#!/bin/bash
# Parallel evaluation (v2 - fixed token IDs): Base / Epoch 0 / Epoch 1 / Epoch 2
# Each on a separate GPU
# Usage: bash scripts/run_eval.sh

PROJECT_DIR="/root/paddlejob/gpfsspace/baidu/personal-code/sys2-all-tools/MLLM-MSR"
PYTHON="${PROJECT_DIR}/.venv/bin/python"
LOG_DIR="${PROJECT_DIR}/logs"
MODEL_PATH="${PROJECT_DIR}/models/llava-v1.6-mistral-7b-hf"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_DIR}"

echo "[$(date)] Launching 4 parallel evaluations (fixed token IDs)..."

# 1. Base model on cuda:0
${PYTHON} train/dpo/quick_eval_sft.py \
    --model_path "${MODEL_PATH}" \
    --checkpoint_path "__NONE__" \
    --dataset microlens \
    --num_users 100 \
    --device cuda:0 \
    > "${LOG_DIR}/eval_base.log" 2>&1 &
PID_BASE=$!
echo "  [cuda:0] Base model -> PID ${PID_BASE}"

# 2. SFT epoch_0 on cuda:1
${PYTHON} train/dpo/quick_eval_sft.py \
    --model_path "${MODEL_PATH}" \
    --checkpoint_path "${PROJECT_DIR}/checkpoints/sft_microlens/epoch_0" \
    --dataset microlens \
    --num_users 100 \
    --device cuda:1 \
    > "${LOG_DIR}/eval_sft_epoch0.log" 2>&1 &
PID_E0=$!
echo "  [cuda:1] SFT epoch_0 -> PID ${PID_E0}"

# 3. SFT epoch_1 on cuda:2
${PYTHON} train/dpo/quick_eval_sft.py \
    --model_path "${MODEL_PATH}" \
    --checkpoint_path "${PROJECT_DIR}/checkpoints/sft_microlens/epoch_1" \
    --dataset microlens \
    --num_users 100 \
    --device cuda:2 \
    > "${LOG_DIR}/eval_sft_epoch1.log" 2>&1 &
PID_E1=$!
echo "  [cuda:2] SFT epoch_1 -> PID ${PID_E1}"

# 4. SFT epoch_2 on cuda:3
${PYTHON} train/dpo/quick_eval_sft.py \
    --model_path "${MODEL_PATH}" \
    --checkpoint_path "${PROJECT_DIR}/checkpoints/sft_microlens/epoch_2" \
    --dataset microlens \
    --num_users 100 \
    --device cuda:3 \
    > "${LOG_DIR}/eval_sft_epoch2.log" 2>&1 &
PID_E2=$!
echo "  [cuda:3] SFT epoch_2 -> PID ${PID_E2}"

echo ""
echo "[$(date)] Waiting for all evaluations to finish..."
wait ${PID_BASE} ${PID_E0} ${PID_E1} ${PID_E2}

echo ""
echo "=========================================="
echo "  All evaluations complete! Results:"
echo "=========================================="
echo ""
echo "--- Base Model ---"
tail -15 "${LOG_DIR}/eval_base.log"
echo ""
echo "--- SFT Epoch 0 ---"
tail -15 "${LOG_DIR}/eval_sft_epoch0.log"
echo ""
echo "--- SFT Epoch 1 ---"
tail -15 "${LOG_DIR}/eval_sft_epoch1.log"
echo ""
echo "--- SFT Epoch 2 ---"
tail -15 "${LOG_DIR}/eval_sft_epoch2.log"
echo ""
echo "[$(date)] Done!"
