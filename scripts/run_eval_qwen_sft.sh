#!/bin/bash
# =============================================================
# Qwen2.5-VL SFT Evaluation: Base / Epoch 0 / Epoch 1
# Parallel evaluation on separate GPUs
# Usage: bash scripts/run_eval_qwen_sft.sh
# =============================================================

PROJECT_DIR="/root/paddlejob/workspace/codelab/projects/rodpo/MLLM-MSR"
PYTHON="/root/miniforge3/envs/rodpo/bin/python"
LOG_DIR="${PROJECT_DIR}/logs"
MODEL_PATH="${PROJECT_DIR}/models/Qwen2.5-VL-7B-Instruct"
CKPT_DIR="${PROJECT_DIR}/checkpoints/qwen_sft_microlens"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_DIR}"

echo "=========================================="
echo "  Qwen2.5-VL SFT Evaluation"
echo "  Start: $(date)"
echo "=========================================="
echo ""

# 1. Base model (no LoRA) on cuda:0
CUDA_VISIBLE_DEVICES=0 ${PYTHON} train/dpo/quick_eval_qwen_sft.py \
    --model_path "${MODEL_PATH}" \
    --checkpoint_path "__NONE__" \
    --dataset microlens \
    --num_users 100 \
    --device cuda:0 \
    > "${LOG_DIR}/eval_qwen_base.log" 2>&1 &
PID_BASE=$!
echo "  [GPU 0] Base model (no LoRA) -> PID ${PID_BASE}"

# 2. SFT epoch_0 on cuda:1
CUDA_VISIBLE_DEVICES=1 ${PYTHON} train/dpo/quick_eval_qwen_sft.py \
    --model_path "${MODEL_PATH}" \
    --checkpoint_path "${CKPT_DIR}/epoch_0" \
    --dataset microlens \
    --num_users 100 \
    --device cuda:0 \
    > "${LOG_DIR}/eval_qwen_sft_epoch0.log" 2>&1 &
PID_E0=$!
echo "  [GPU 1] SFT epoch_0 -> PID ${PID_E0}"

# 3. SFT epoch_1 on cuda:2
CUDA_VISIBLE_DEVICES=2 ${PYTHON} train/dpo/quick_eval_qwen_sft.py \
    --model_path "${MODEL_PATH}" \
    --checkpoint_path "${CKPT_DIR}/epoch_1" \
    --dataset microlens \
    --num_users 100 \
    --device cuda:0 \
    > "${LOG_DIR}/eval_qwen_sft_epoch1.log" 2>&1 &
PID_E1=$!
echo "  [GPU 2] SFT epoch_1 -> PID ${PID_E1}"

echo ""
echo "[$(date)] Waiting for all evaluations to finish..."
echo "  (Estimated: ~5-10 min per evaluation, running in parallel)"
echo ""
wait ${PID_BASE} ${PID_E0} ${PID_E1}

echo ""
echo "=========================================="
echo "  All evaluations complete!"
echo "  End: $(date)"
echo "=========================================="
echo ""
echo "--- Base Model (no LoRA) ---"
tail -15 "${LOG_DIR}/eval_qwen_base.log"
echo ""
echo "--- SFT Epoch 0 ---"
tail -15 "${LOG_DIR}/eval_qwen_epoch0.log"
echo ""
echo "--- SFT Epoch 1 ---"
tail -15 "${LOG_DIR}/eval_qwen_epoch1.log"
echo ""
echo "[$(date)] Done!"
