#!/usr/bin/env python
"""Generate ScoreCache using SFT model (base + SFT LoRA merged).

Usage:
    CUDA_VISIBLE_DEVICES=0 python generate_score_cache.py \
        --dataset microlens \
        --model_path ../../../models/llava-v1.6-mistral-7b-hf \
        --sft_lora_path ../../../checkpoints/sft_microlens/epoch_0 \
        --max_users 2000 \
        --batch_size 8 \
        --output score_cache_sft.json
"""
import os, sys, json, argparse
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaNextForConditionalGeneration
from peft import PeftModel

PROMPT_TEMPLATE = (
    "Based on the previous interaction history, the user's preference "
    "can be summarized as: {user_preference}\n"
    "Please predict whether this user would interact with the item. "
    "The item's title is '{title}'.\n"
    "Please only response 'yes' or 'no'."
)


def resize_images_to_uniform(images):
    if not images:
        return images
    max_w = max(img.width for img in images)
    max_h = max(img.height for img in images)
    result = []
    for img in images:
        if img.width == max_w and img.height == max_h:
            result.append(img)
        else:
            new_img = Image.new('RGB', (max_w, max_h), (0, 0, 0))
            new_img.paste(img, (0, 0))
            result.append(new_img)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='microlens')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--sft_lora_path', type=str, required=True)
    parser.add_argument('--max_users', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--output', type=str, default='score_cache_sft.json')
    parser.add_argument('--shard', type=int, default=0, help='Shard index (0-based)')
    parser.add_argument('--num_shards', type=int, default=1, help='Total number of shards')
    args = parser.parse_args()

    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    if args.dataset == 'microlens':
        data_path = os.path.join(base_dir, 'data', 'microlens', 'dpo_ready', 'dpo_train.json')
        image_dir = os.path.join(base_dir, 'data', 'microlens', 'images')
    else:
        data_path = os.path.join(base_dir, 'data', 'amazon', 'dpo_ready', 'dpo_train.json')
        image_dir = os.path.join(base_dir, 'data', 'amazon', 'images')

    # Load model: base + SFT LoRA merged
    print("[1/3] Loading SFT model...")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.float16,
        _attn_implementation="flash_attention_2"
    )
    model = PeftModel.from_pretrained(model, args.sft_lora_path)
    model = model.merge_and_unload()
    model.eval().cuda()

    processor = AutoProcessor.from_pretrained(args.model_path)
    processor.tokenizer.padding_side = "left"
    yes_token_id = processor.tokenizer.convert_tokens_to_ids("▁Yes")
    if yes_token_id is None or yes_token_id == 0:
        yes_token_id = processor.tokenizer.convert_tokens_to_ids("Yes")
    print(f"  Yes token ID: {yes_token_id}")

    # Load data
    print("[2/3] Loading data...")
    with open(data_path, 'r') as f:
        all_data = json.load(f)
    sampled = all_data[:args.max_users]
    # Shard the data
    shard_size = len(sampled) // args.num_shards
    start = args.shard * shard_size
    end = len(sampled) if args.shard == args.num_shards - 1 else start + shard_size
    sampled = sampled[start:end]
    print(f"  Shard {args.shard}/{args.num_shards}: users [{start}:{end}] = {len(sampled)}")

    # Compute scores
    print("[3/3] Computing P(Yes) scores...")
    cache = {}
    for sample in tqdm(sampled, desc="ScoreCache"):
        user_id = sample['user_id']
        user_pref = sample['user_preference']
        neg_ids = sample['neg_item_ids']

        items = []
        for nid in neg_ids:
            items.append({
                'title': sample['neg_titles'][str(nid)],
                'image_path': sample['neg_image_paths'][str(nid)],
            })

        scores = {}
        for i in range(0, len(items), args.batch_size):
            batch_items = items[i:i + args.batch_size]
            batch_ids = neg_ids[i:i + args.batch_size]

            texts = []
            images = []
            for item in batch_items:
                prompt = PROMPT_TEMPLATE.format(
                    user_preference=user_pref, title=item['title'])
                texts.append(f"[INST] <image>\n{prompt} [/INST]")
                img_path = os.path.join(image_dir, item['image_path'])
                if os.path.exists(img_path):
                    try:
                        images.append(Image.open(img_path).convert('RGB'))
                    except Exception:
                        images.append(Image.new('RGB', (336, 336), (0, 0, 0)))
                else:
                    images.append(Image.new('RGB', (336, 336), (0, 0, 0)))

            images = resize_images_to_uniform(images)
            batch = processor(
                text=texts, images=images,
                padding=True, truncation=True,
                max_length=512, return_tensors="pt"
            )
            batch = {k: v.cuda() for k, v in batch.items()
                     if isinstance(v, torch.Tensor)}

            with torch.no_grad():
                outputs = model(**batch)
                logits = outputs.logits
                last_pos = batch['attention_mask'].sum(dim=1) - 1
                bs = logits.size(0)
                last_logits = logits[torch.arange(bs, device=logits.device), last_pos]
                log_probs = F.log_softmax(last_logits, dim=-1)
                batch_scores = log_probs[:, yes_token_id].cpu().tolist()

            for nid, score in zip(batch_ids, batch_scores):
                scores[nid] = score

        cache[user_id] = scores

    # Save
    if args.num_shards > 1:
        out_name = args.output.replace('.json', f'_shard{args.shard}.json')
    else:
        out_name = args.output
    output_path = os.path.join(base_dir, 'checkpoints', out_name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(cache, f)
    print(f"\nDone! Score cache saved to: {output_path}")
    print(f"  Users: {len(cache)}, Total scores: {sum(len(v) for v in cache.values())}")


if __name__ == '__main__':
    main()
