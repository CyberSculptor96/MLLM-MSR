#!/bin/bash
# Launch 8 shards of score cache generation in parallel (1 GPU each)
export PATH=/usr/bin:/usr/local/bin:/bin:$PATH

PYTHON=/root/miniforge3/envs/rodpo/bin/python
PROJECT_DIR=/root/paddlejob/workspace/codelab/projects/rodpo/MLLM-MSR
SCRIPT=${PROJECT_DIR}/train/dpo/generate_score_cache.py
MODEL_PATH=${PROJECT_DIR}/models/llava-v1.6-mistral-7b-hf
SFT_CKPT=${PROJECT_DIR}/checkpoints/sft_microlens/epoch_0
LOG_DIR=${PROJECT_DIR}/logs
NUM_SHARDS=8

echo "============================================"
echo "ScoreCache Generation: 8 shards parallel"
echo "  Start: $(date)"
echo "============================================"

PIDS=()
for i in $(seq 0 7); do
    echo "  Launching shard $i on GPU $i..."
    CUDA_VISIBLE_DEVICES=$i ${PYTHON} ${SCRIPT} \
        --dataset microlens \
        --model_path ${MODEL_PATH} \
        --sft_lora_path ${SFT_CKPT} \
        --max_users 2000 \
        --batch_size 16 \
        --shard $i \
        --num_shards ${NUM_SHARDS} \
        --output score_cache_sft.json \
        > ${LOG_DIR}/score_cache_shard${i}.log 2>&1 &
    PIDS+=($!)
done

echo ""
echo "All shards launched. PIDs: ${PIDS[*]}"
echo "Waiting for completion..."
wait "${PIDS[@]}"

echo ""
echo "All shards done! Merging..."

# Merge shards
${PYTHON} -c "
import json, glob, os
base = '${PROJECT_DIR}/checkpoints'
merged = {}
for i in range(${NUM_SHARDS}):
    path = os.path.join(base, f'score_cache_sft_shard{i}.json')
    with open(path) as f:
        shard = json.load(f)
    merged.update(shard)
    os.remove(path)
out = os.path.join(base, 'score_cache_sft.json')
with open(out, 'w') as f:
    json.dump(merged, f)
print(f'Merged {len(merged)} users -> {out}')
"

echo "============================================"
echo "ScoreCache complete! $(date)"
echo "============================================"
