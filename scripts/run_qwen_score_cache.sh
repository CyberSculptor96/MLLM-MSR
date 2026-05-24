#!/bin/bash
# =============================================================
# Generate Qwen2.5-VL Score Cache (8 GPU parallel shards)
# Output: checkpoints/qwen_score_cache_sft.json
# =============================================================
export PATH=/usr/bin:/usr/local/bin:/bin:$PATH

PYTHON=/root/miniforge3/envs/rodpo/bin/python
PROJECT_DIR=/root/paddlejob/workspace/codelab/projects/rodpo/MLLM-MSR
SCRIPT=${PROJECT_DIR}/train/dpo/generate_score_cache_qwen.py
SFT_CKPT=${PROJECT_DIR}/checkpoints/qwen_sft_microlens/epoch_0
OUTPUT_NAME=qwen_score_cache_sft.json
NUM_SHARDS=8
LOG_DIR=${PROJECT_DIR}/logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p ${LOG_DIR}
cd ${PROJECT_DIR}/train/dpo

echo "============================================"
echo "Qwen2.5-VL Score Cache Generation (8 shards)"
echo "  SFT LoRA: ${SFT_CKPT}"
echo "  Output: ${PROJECT_DIR}/checkpoints/${OUTPUT_NAME}"
echo "  Start: $(date)"
echo "============================================"

# Launch 8 shards in parallel (1 GPU each)
PIDS=()
for i in $(seq 0 7); do
    echo "  [GPU ${i}] Shard ${i}/${NUM_SHARDS}..."
    CUDA_VISIBLE_DEVICES=${i} ${PYTHON} ${SCRIPT} \
        --dataset microlens \
        --sft_lora_path ${SFT_CKPT} \
        --max_users 2000 \
        --batch_size 4 \
        --output ${OUTPUT_NAME} \
        --shard ${i} \
        --num_shards ${NUM_SHARDS} \
        > ${LOG_DIR}/qwen_cache_shard${i}_${TIMESTAMP}.log 2>&1 &
    PIDS+=($!)
done

echo ""
echo "All 8 shards launched. Waiting..."
for pid in "${PIDS[@]}"; do
    wait ${pid}
    if [ $? -ne 0 ]; then
        echo "ERROR: Shard PID ${pid} failed!"
    fi
done
echo "All shards complete. $(date)"

# Merge shards
echo ""
echo ">>> Merging shards..."
${PYTHON} -c "
import json, os

base = '${PROJECT_DIR}/checkpoints'
output = os.path.join(base, '${OUTPUT_NAME}')
merged = {}
for i in range(${NUM_SHARDS}):
    shard_path = os.path.join(base, '${OUTPUT_NAME}'.replace('.json', f'_shard{i}.json'))
    if os.path.exists(shard_path):
        with open(shard_path, 'r') as f:
            data = json.load(f)
        merged.update(data)
        print(f'  Shard {i}: {len(data)} users')
        os.remove(shard_path)
    else:
        print(f'  WARNING: Shard {i} not found at {shard_path}')

with open(output, 'w') as f:
    json.dump(merged, f)
print(f'\nMerged: {len(merged)} users -> {output}')
print(f'Total scores: {sum(len(v) for v in merged.values())}')
"

echo ""
echo "============================================"
echo "Score cache generation complete!"
echo "  Output: ${PROJECT_DIR}/checkpoints/${OUTPUT_NAME}"
echo "  End: $(date)"
echo "============================================"
