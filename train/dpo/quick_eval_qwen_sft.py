"""
Quick SFT Validation Script for Qwen2.5-VL.

Evaluates Qwen SFT checkpoint on a small subset of test data.
Checks if model actually learned to distinguish Yes/No correctly.

Usage:
    # Evaluate base model (no LoRA):
    python quick_eval_qwen_sft.py --checkpoint_path __NONE__ --num_users 50

    # Evaluate SFT epoch_0:
    python quick_eval_qwen_sft.py --checkpoint_path ../../checkpoints/qwen_sft_microlens/epoch_0 --num_users 50

    # Evaluate SFT epoch_1:
    python quick_eval_qwen_sft.py --checkpoint_path ../../checkpoints/qwen_sft_microlens/epoch_1 --num_users 50
"""

import os
import sys
import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

# Prompt template (must match training)
PROMPT_TEMPLATE = (
    "Based on the previous interaction history, the user's preference "
    "can be summarized as: {user_preference}\n"
    "Please predict whether this user would interact with the item. "
    "The item's title is '{title}'.\n"
    "Please only response 'yes' or 'no'."
)


def evaluate(model, processor, test_data, image_dir, device='cuda:0', num_users=50):
    """Quick evaluation: compute Recall@5, MRR@5 on small subset."""
    model.eval()

    # Get Yes/No token IDs for Qwen2.5
    yes_token_id = processor.tokenizer.encode('Yes', add_special_tokens=False)[0]
    no_token_id = processor.tokenizer.encode('No', add_special_tokens=False)[0]
    print(f"  Yes token ID: {yes_token_id}, No token ID: {no_token_id}")
    print(f"  Yes token: {processor.tokenizer.convert_ids_to_tokens(yes_token_id)}")
    print(f"  No token: {processor.tokenizer.convert_ids_to_tokens(no_token_id)}")

    test_subset = test_data[:num_users]
    recall_at_5 = []
    mrr_at_5 = []
    yes_prob_pos = []
    yes_prob_neg = []

    for sample in tqdm(test_subset, desc="Evaluating"):
        user_pref = sample['user_preference']
        candidates = sample['candidates']

        scores = []
        labels = []

        for cand in candidates:
            prompt = PROMPT_TEMPLATE.format(
                user_preference=user_pref,
                title=cand['title'],
            )

            # Build messages for Qwen chat template
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            img_path = os.path.join(image_dir, cand['image_path'])
            if os.path.exists(img_path):
                try:
                    image = Image.open(img_path).convert('RGB')
                except Exception:
                    image = Image.new('RGB', (224, 224), (0, 0, 0))
            else:
                image = Image.new('RGB', (224, 224), (0, 0, 0))

            batch = processor(
                text=[text], images=[image],
                padding=True, truncation=True,
                max_length=1024, return_tensors="pt"
            )
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            with torch.no_grad():
                outputs = model(**batch)
                logits = outputs.logits
                last_pos = batch['attention_mask'].sum(dim=1) - 1
                last_logits = logits[0, last_pos[0]]  # (V,)

                # P(Yes) via softmax over [Yes, No] only
                yes_no_logits = torch.tensor([last_logits[yes_token_id], last_logits[no_token_id]])
                probs = F.softmax(yes_no_logits, dim=0)
                p_yes = probs[0].item()

            scores.append(p_yes)
            labels.append(cand['label'])

            if cand['label'] == 1:
                yes_prob_pos.append(p_yes)
            else:
                yes_prob_neg.append(p_yes)

        # Compute ranking metrics
        sorted_indices = np.argsort(-np.array(scores))
        sorted_labels = np.array(labels)[sorted_indices]

        # Recall@5
        recall_at_5.append(float(sorted_labels[:5].sum() > 0))

        # MRR@5
        mrr = 0.0
        for i in range(min(5, len(sorted_labels))):
            if sorted_labels[i] == 1:
                mrr = 1.0 / (i + 1)
                break
        mrr_at_5.append(mrr)

    # Summary
    print("\n" + "=" * 50)
    print("Quick Evaluation Results:")
    print("=" * 50)
    print(f"  Users evaluated: {len(test_subset)}")
    print(f"  Recall@5:  {np.mean(recall_at_5):.4f}")
    print(f"  MRR@5:     {np.mean(mrr_at_5):.4f}")
    print(f"  Avg P(Yes) for POSITIVE items: {np.mean(yes_prob_pos):.4f}")
    print(f"  Avg P(Yes) for NEGATIVE items: {np.mean(yes_prob_neg):.4f}")
    print(f"  Separation (pos - neg):        {np.mean(yes_prob_pos) - np.mean(yes_prob_neg):.4f}")
    print("=" * 50)

    sep = np.mean(yes_prob_pos) - np.mean(yes_prob_neg)
    if sep > 0.05:
        print("  Model shows meaningful separation between pos/neg items.")
    elif sep > 0:
        print("  Model shows weak separation. Training might need more epochs.")
    else:
        print("  Model does NOT distinguish pos from neg! Training may be broken.")

    return {
        'recall_at_5': float(np.mean(recall_at_5)),
        'mrr_at_5': float(np.mean(mrr_at_5)),
        'avg_p_yes_pos': float(np.mean(yes_prob_pos)),
        'avg_p_yes_neg': float(np.mean(yes_prob_neg)),
        'separation': float(sep),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='/root/paddlejob/workspace/codelab/projects/rodpo/MLLM-MSR/models/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to SFT LoRA checkpoint (use __NONE__ for base model)')
    parser.add_argument('--dataset', type=str, default='microlens',
                        choices=['video_games', 'microlens'])
    parser.add_argument('--num_users', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    # Resolve paths
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    if args.dataset == 'video_games':
        test_path = os.path.join(base_dir, 'data', 'amazon', 'dpo_ready', 'test.json')
        image_dir = os.path.join(base_dir, 'data', 'amazon', 'images')
    else:
        test_path = os.path.join(base_dir, 'data', 'microlens', 'dpo_ready', 'test.json')
        image_dir = os.path.join(base_dir, 'data', 'microlens', 'images')

    print(f"Loading test data from {test_path}...")
    with open(test_path, 'r') as f:
        test_data = json.load(f)
    print(f"  {len(test_data)} users total, evaluating {args.num_users}")

    print(f"Loading base model from {args.model_path}...")
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
    )

    if args.checkpoint_path and args.checkpoint_path != "__NONE__":
        print(f"Loading SFT LoRA from {args.checkpoint_path}...")
        model = PeftModel.from_pretrained(model, args.checkpoint_path)
    else:
        print("Evaluating BASE model (no LoRA)...")

    model = model.to(args.device)
    model.eval()

    print("\nRunning quick evaluation...")
    metrics = evaluate(model, processor, test_data, image_dir,
                       device=args.device, num_users=args.num_users)

    # Save results
    if args.checkpoint_path and args.checkpoint_path != "__NONE__":
        results_path = os.path.join(args.checkpoint_path, 'quick_eval_results.json')
    else:
        results_path = os.path.join(base_dir, 'logs', 'eval_qwen_base_results.json')
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()