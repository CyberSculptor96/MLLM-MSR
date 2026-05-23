#!/bin/bash
# =============================================================
# RoDPO on MLLM-MSR: Full Training Pipeline
# =============================================================
# Usage:
#   bash run_all.sh [stage]
#   Stages: sft_amazon, sft_microlens, dpo_amazon, dpo_microlens, eval
# =============================================================

set -e

export https_proxy=http://agent.baidu.com:8891
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_DIR="${BASE_DIR}/train/dpo"
CKPT_DIR="${BASE_DIR}/checkpoints"
MODEL_PATH="${BASE_DIR}/../models/llava-v1.6-mistral-7b-hf"

STAGE=${1:-"all"}

echo "======================================"
echo "RoDPO on MLLM-MSR Training Pipeline"
echo "Stage: ${STAGE}"
echo "======================================"

# ---- Stage 1: SFT Training ----
run_sft() {
    local dataset=$1
    echo ""
    echo "[SFT] Training on ${dataset}..."
    cd "${TRAIN_DIR}"
    python train_llava_sft.py \
        --dataset ${dataset} \
        --model_path ${MODEL_PATH} \
        --devices 8 \
        --epochs 4

    echo "[SFT] ${dataset} done. Checkpoint: ${CKPT_DIR}/sft_${dataset}/"
}

# ---- Stage 2: DPO Training (3 strategies) ----
run_dpo() {
    local dataset=$1
    local sft_path="${CKPT_DIR}/sft_${dataset}/epoch_3"

    if [ ! -d "${sft_path}" ]; then
        echo "[ERROR] SFT checkpoint not found: ${sft_path}"
        echo "  Please run SFT training first."
        exit 1
    fi

    for strategy in random hard top_k; do
        echo ""
        echo "[DPO] Training ${dataset} with strategy=${strategy}..."
        cd "${TRAIN_DIR}"
        python train_llava_dpo.py \
            --strategy ${strategy} \
            --dataset ${dataset} \
            --sft_lora_path ${sft_path} \
            --devices 8 \
            --epochs 3
        echo "[DPO] ${dataset}/${strategy} done."
    done
}

# ---- Stage 3: Evaluation ----
run_eval() {
    local dataset=$1
    local sft_path="${CKPT_DIR}/sft_${dataset}/epoch_3"

    echo ""
    echo "[EVAL] Evaluating SFT-only on ${dataset}..."
    cd "${TRAIN_DIR}"
    python eval_dpo.py \
        --model_path ${MODEL_PATH} \
        --sft_lora_path ${sft_path} \
        --dataset ${dataset} \
        --batch_size 4

    for strategy in random hard top_k; do
        local dpo_path="${CKPT_DIR}/dpo_${dataset}_${strategy}/epoch_2"
        if [ -d "${dpo_path}" ]; then
            echo ""
            echo "[EVAL] Evaluating DPO (${strategy}) on ${dataset}..."
            python eval_dpo.py \
                --model_path ${MODEL_PATH} \
                --sft_lora_path ${sft_path} \
                --dpo_lora_path ${dpo_path} \
                --dataset ${dataset} \
                --batch_size 4
        else
            echo "[WARN] DPO checkpoint not found: ${dpo_path}"
        fi
    done
}

# ---- Execute based on stage ----
case ${STAGE} in
    sft_amazon)
        run_sft video_games
        ;;
    sft_microlens)
        run_sft microlens
        ;;
    dpo_amazon)
        run_dpo video_games
        ;;
    dpo_microlens)
        run_dpo microlens
        ;;
    eval_amazon)
        run_eval video_games
        ;;
    eval_microlens)
        run_eval microlens
        ;;
    eval)
        run_eval video_games
        run_eval microlens
        ;;
    all)
        run_sft video_games
        run_sft microlens
        run_dpo video_games
        run_dpo microlens
        run_eval video_games
        run_eval microlens
        ;;
    *)
        echo "Unknown stage: ${STAGE}"
        echo "Available: sft_amazon, sft_microlens, dpo_amazon, dpo_microlens, eval_amazon, eval_microlens, eval, all"
        exit 1
        ;;
esac

echo ""
echo "======================================"
echo "Pipeline stage '${STAGE}' complete!"
echo "======================================"
