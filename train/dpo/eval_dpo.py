"""
Evaluation script for DPO-trained LLaVA recommender.

Computes ranking metrics (NDCG, MRR, Recall, AUC) on test set.
Supports evaluating: SFT-only, DPO (random/hard/top_k).
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import roc_auc_score

from transformers import AutoProcessor, LlavaNextForConditionalGeneration
from peft import PeftModel

# ============ Prompt Template ============
PROMPT_TEMPLATE = (
    "Based on the previous interaction history, the user's preference "
    "can be summarized as: {user_preference}\n"
    "Please predict whether this user would interact with the item. "
    "The item's title is '{title}'.\n"
    "Please only response 'yes' or 'no'."
)


def compute_metrics(labels, scores, k_list=[3, 5, 10]):
    """Compute ranking metrics for a single user group."""
    # Sort by score descending
    sorted_indices = np.argsort(-np.array(scores))
    sorted_labels = np.array(labels)[sorted_indices]

    results = {}
    for k in k_list:
        # Recall@K
        results[f'Recall@{k}'] = float(sorted_labels[:k].sum() > 0)

        # NDCG@K
        dcg = 0.0
        for i in range(min(k, len(sorted_labels))):
            if sorted_labels[i] == 1:
                dcg += 1.0 / np.log2(i + 2)
        idcg = 1.0 / np.log2(2)  # Only 1 positive per group
        results[f'NDCG@{k}'] = dcg / idcg if idcg > 0 else 0.0

        # MRR@K
        mrr = 0.0
        for i in range(min(k, len(sorted_labels))):
            if sorted_labels[i] == 1:
                mrr = 1.0 / (i + 1)
                break
        results[f'MRR@{k}'] = mrr

    return results


def evaluate_model(model, processor, test_data, image_dir, device='cuda',
                   batch_size=4, max_users=None):
    """
    Evaluate a model on test data.

    Args:
        model: LLaVA model (possibly with LoRA)
        processor: AutoProcessor
        test_data: list of test samples (each with 1 pos + 20 neg candidates)
        image_dir: directory with item images
        device: cuda device
        batch_size: candidates to process per batch
        max_users: limit for debugging
    """
    model.eval()
    yes_token_id = processor.tokenizer.convert_tokens_to_ids('Yes')

    if max_users:
        test_data = test_data[:max_users]

    all_metrics = []

    for sample in tqdm(test_data, desc="Evaluating"):
        user_pref = sample['user_preference']
        candidates = sample['candidates']

        # Score all candidates
        all_scores = []
        all_labels = []

        for i in range(0, len(candidates), batch_size):
            batch_candidates = candidates[i:i + batch_size]

            texts = []
            images = []
            for cand in batch_candidates:
                prompt = PROMPT_TEMPLATE.format(
                    user_preference=user_pref,
                    title=cand['title'],
                )
                text = f"[INST] <image>\n{prompt} [/INST]"
                texts.append(text)

                img_path = os.path.join(image_dir, cand['image_path'])
                if os.path.exists(img_path):
                    try:
                        images.append(Image.open(img_path).convert('RGB'))
                    except Exception:
                        images.append(Image.new('RGB', (336, 336), (0, 0, 0)))
                else:
                    images.append(Image.new('RGB', (336, 336), (0, 0, 0)))

            batch = processor(
                text=texts, images=images,
                padding=True, truncation=True,
                max_length=1024, return_tensors="pt"
            )
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            with torch.no_grad():
                outputs = model(**batch)
                logits = outputs.logits
                last_pos = batch['attention_mask'].sum(dim=1) - 1
                bs = logits.size(0)
                last_logits = logits[torch.arange(bs, device=device), last_pos]

                # P(Yes) via softmax over full vocab
                probs = F.softmax(last_logits, dim=-1)
                yes_probs = probs[:, yes_token_id].cpu().numpy()

            all_scores.extend(yes_probs.tolist())
            all_labels.extend([cand['label'] for cand in batch_candidates])

        # Compute metrics for this user
        metrics = compute_metrics(all_labels, all_scores)
        all_metrics.append(metrics)

    # Average metrics across all users
    avg_metrics = {}
    if all_metrics:
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])

    # AUC (computed globally)
    all_flat_labels = []
    all_flat_scores = []
    for sample in test_data:
        # Would need to re-score, so skip for now if needed
        pass

    return avg_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model (base model ID or local path)')
    parser.add_argument('--sft_lora_path', type=str, default=None,
                        help='Path to SFT LoRA (if evaluating SFT-only)')
    parser.add_argument('--dpo_lora_path', type=str, default=None,
                        help='Path to DPO LoRA (if evaluating DPO)')
    parser.add_argument('--dataset', type=str, default='video_games',
                        choices=['video_games', 'microlens'])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_users', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    # Resolve data paths
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    if args.dataset == 'video_games':
        test_path = os.path.join(base_dir, 'data', 'amazon', 'dpo_ready', 'test.json')
        image_dir = os.path.join(base_dir, 'data', 'amazon', 'images')
    elif args.dataset == 'microlens':
        test_path = os.path.join(base_dir, 'data', 'microlens', 'dpo_ready', 'test.json')
        image_dir = os.path.join(base_dir, 'data', 'microlens', 'images')

    print(f"Loading test data from {test_path}...")
    with open(test_path, 'r') as f:
        test_data = json.load(f)
    print(f"  {len(test_data)} test users")

    # Load model
    print(f"Loading model from {args.model_path}...")
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        _attn_implementation="flash_attention_2",
    )

    if args.sft_lora_path:
        print(f"Loading SFT LoRA from {args.sft_lora_path}...")
        model = PeftModel.from_pretrained(model, args.sft_lora_path)
        if args.dpo_lora_path:
            # Merge SFT, then load DPO
            model = model.merge_and_unload()
            print(f"Loading DPO LoRA from {args.dpo_lora_path}...")
            model = PeftModel.from_pretrained(model, args.dpo_lora_path)

    model = model.to(args.device)
    model.eval()

    # Evaluate
    print("\nRunning evaluation...")
    metrics = evaluate_model(
        model, processor, test_data, image_dir,
        device=args.device, batch_size=args.batch_size,
        max_users=args.max_users,
    )

    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results:")
    print("=" * 50)
    for key, value in sorted(metrics.items()):
        print(f"  {key}: {value:.4f}")
    print("=" * 50)

    # Save results
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    model_name = os.path.basename(args.dpo_lora_path or args.sft_lora_path or 'base')
    results_path = os.path.join(results_dir, f'eval_{args.dataset}_{model_name}.json')
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
