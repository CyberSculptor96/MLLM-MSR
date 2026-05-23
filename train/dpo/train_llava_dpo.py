"""
DPO Training Script for LLaVA-based Multimodal Sequential Recommendation.

Implements RoDPO (Stochastic Top-K Negative Sampling) on MLLM-MSR.
Supports three negative sampling strategies: random, hard (argmax), top_k.

Features:
  - Validation DPO loss computed every N steps
  - Periodic ranking evaluation (Recall@5, MRR@5) on test subset
  - TensorBoard logging for all metrics
  - Early stopping based on validation loss
  - Cosine LR scheduler with warmup

Usage:
    python train_llava_dpo.py --strategy top_k --dataset microlens --sft_lora_path <path>
    python train_llava_dpo.py --strategy random --dataset microlens --sft_lora_path <path>
    python train_llava_dpo.py --strategy hard --dataset microlens --sft_lora_path <path>
"""

import os
import sys
import json
import argparse
import functools
import random
from typing import List, Dict

import torch
import torch.nn.functional as F
import numpy as np
import lightning as L
from lightning.pytorch.callbacks import Callback, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from transformers import AutoProcessor, LlavaNextForConditionalGeneration
from peft import PeftModel, LoraConfig, get_peft_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dpo_dataset import DPOPreferenceDataset

# ============ Hyperparameters ============
DEFAULT_CONFIG = {
    'dpo_beta': 0.1,
    'lr': 5e-6,
    'max_epochs': 3,
    'batch_size': 1,
    'accumulate_grad_batches': 2,
    'max_length': 1024,
    'gradient_clip_val': 1.0,
    'top_k': 50,
    'lora_r': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.1,
    # Validation & logging
    'val_check_interval': 2000,    # validate every N training steps
    'val_num_samples': 500,       # DPO validation set size
    'eval_num_users': 50,         # ranking eval users (lightweight)
    'warmup_steps': 100,
}

# ============ Prompt Template ============
PROMPT_TEMPLATE = (
    "Based on the previous interaction history, the user's preference "
    "can be summarized as: {user_preference}\n"
    "Please predict whether this user would interact with the item. "
    "The item's title is '{title}'.\n"
    "Please only response 'yes' or 'no'."
)


def find_all_linear_names(model):
    """Find all linear layer names for LoRA targeting (exclude vision/projector/lm_head)."""
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['multi_modal_projector', 'vision_model', 'lm_head']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[-1])
    return list(lora_module_names)


def resize_images_to_uniform(images: List[Image.Image]) -> List[Image.Image]:
    """Pad images to uniform dimensions within a batch."""
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


class LlavaDPOModule(L.LightningModule):
    """PyTorch Lightning module for DPO training with validation."""

    def __init__(self, config: dict, processor, model, val_dataset=None, test_data=None, image_dir=None):
        super().__init__()
        self.config = config
        self.model = model
        self.processor = processor
        self.beta = config['dpo_beta']
        self.val_dataset = val_dataset
        self.test_data = test_data
        self.image_dir = image_dir

        # Correct token IDs (with space prefix, matching training)
        self.yes_token_id = processor.tokenizer.encode('Yes', add_special_tokens=False)[0]
        self.no_token_id = processor.tokenizer.encode('No', add_special_tokens=False)[0]
        print(f"Token IDs - Yes: {self.yes_token_id} (▁Yes), No: {self.no_token_id} (▁No)")

        # Track best val loss for model selection
        self.best_val_loss = float('inf')
        self.best_epoch = -1

    def extract_log_prob_yes(self, input_ids, attention_mask, pixel_values, image_sizes):
        """Extract log P("Yes") at the first response position."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
        )
        logits = outputs.logits  # (B, seq_len, vocab_size)
        last_pos = attention_mask.sum(dim=1) - 1  # (B,)
        batch_size = logits.size(0)
        last_logits = logits[torch.arange(batch_size, device=logits.device), last_pos]  # (B, V)
        log_probs = F.log_softmax(last_logits, dim=-1)
        return log_probs[:, self.yes_token_id]  # (B,)

    def _build_batch(self, items_data: List[dict], key_prefix: str) -> dict:
        """Build a processor-ready batch for pos or neg items."""
        texts = []
        images = []
        for item in items_data:
            prompt = PROMPT_TEMPLATE.format(
                user_preference=item['user_preference'],
                title=item[f'{key_prefix}_title'],
            )
            text = f"[INST] <image>\n{prompt} [/INST]"
            texts.append(text)
            images.append(item[f'{key_prefix}_image'])

        images = resize_images_to_uniform(images)
        batch = self.processor(
            text=texts, images=images,
            padding=True, truncation=True,
            max_length=self.config['max_length'],
            return_tensors="pt"
        )
        return {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

    def _compute_dpo_loss(self, batch):
        """Compute DPO loss and metrics. Shared by train and val."""
        pos_batch = self._build_batch(batch, 'pos')
        neg_batch = self._build_batch(batch, 'neg')

        # Policy forward (DPO adapter enabled)
        policy_log_p_pos = self.extract_log_prob_yes(**pos_batch)
        policy_log_p_neg = self.extract_log_prob_yes(**neg_batch)

        # Reference forward (DPO adapter disabled, no grad)
        with torch.no_grad():
            self.model.disable_adapter_layers()
            ref_log_p_pos = self.extract_log_prob_yes(**pos_batch)
            ref_log_p_neg = self.extract_log_prob_yes(**neg_batch)
            self.model.enable_adapter_layers()

        # DPO Loss
        pi_logratios = policy_log_p_pos - policy_log_p_neg
        ref_logratios = ref_log_p_pos - ref_log_p_neg
        logits = pi_logratios - ref_logratios
        dpo_loss = -F.logsigmoid(self.beta * logits).mean()

        # Metrics
        reward_acc = (logits > 0).float().mean()
        margin = logits.mean()

        return dpo_loss, reward_acc, margin, policy_log_p_pos.mean(), policy_log_p_neg.mean()

    def training_step(self, batch, batch_idx):
        dpo_loss, reward_acc, margin, log_p_pos, log_p_neg = self._compute_dpo_loss(batch)

        # Log training metrics (explicit batch_size for List[dict] batch)
        bs = len(batch)
        self.log('train/dpo_loss', dpo_loss, prog_bar=True, sync_dist=True, batch_size=bs)
        self.log('train/reward_acc', reward_acc, prog_bar=True, sync_dist=True, batch_size=bs)
        self.log('train/margin', margin, sync_dist=True, batch_size=bs)
        self.log('train/log_p_pos', log_p_pos, sync_dist=True, batch_size=bs)
        self.log('train/log_p_neg', log_p_neg, sync_dist=True, batch_size=bs)

        return dpo_loss

    def validation_step(self, batch, batch_idx):
        dpo_loss, reward_acc, margin, log_p_pos, log_p_neg = self._compute_dpo_loss(batch)

        bs = len(batch)
        self.log('val/dpo_loss', dpo_loss, prog_bar=True, sync_dist=True, batch_size=bs)
        self.log('val/reward_acc', reward_acc, prog_bar=True, sync_dist=True, batch_size=bs)
        self.log('val/margin', margin, sync_dist=True, batch_size=bs)

        return dpo_loss

    def on_validation_epoch_end(self):
        """Run ranking evaluation on test subset after validation."""
        if self.test_data is None or self.image_dir is None:
            return
        if self.trainer.global_rank != 0:
            return

        # Lightweight ranking eval on small subset
        eval_users = min(self.config['eval_num_users'], len(self.test_data))
        test_subset = self.test_data[:eval_users]

        self.model.eval()
        recall_5_list = []
        mrr_5_list = []

        for sample in test_subset:
            user_pref = sample['user_preference']
            candidates = sample['candidates']
            scores = []
            labels = []

            for cand in candidates:
                prompt = PROMPT_TEMPLATE.format(
                    user_preference=user_pref,
                    title=cand['title'],
                )
                text = f"[INST] <image>\n{prompt} [/INST]"
                img_path = os.path.join(self.image_dir, cand['image_path'])
                if os.path.exists(img_path):
                    try:
                        image = Image.open(img_path).convert('RGB')
                    except Exception:
                        image = Image.new('RGB', (336, 336), (0, 0, 0))
                else:
                    image = Image.new('RGB', (336, 336), (0, 0, 0))

                batch = self.processor(
                    text=[text], images=[image],
                    padding=True, truncation=True,
                    max_length=1024, return_tensors="pt"
                )
                batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

                with torch.no_grad():
                    outputs = self.model(**batch)
                    logits = outputs.logits
                    last_pos = batch['attention_mask'].sum(dim=1) - 1
                    last_logits = logits[0, last_pos[0]]
                    yes_no_logits = torch.tensor([last_logits[self.yes_token_id], last_logits[self.no_token_id]])
                    probs = F.softmax(yes_no_logits, dim=0)
                    p_yes = probs[0].item()

                scores.append(p_yes)
                labels.append(cand['label'])

            # Ranking metrics
            sorted_indices = np.argsort(-np.array(scores))
            sorted_labels = np.array(labels)[sorted_indices]
            recall_5_list.append(float(sorted_labels[:5].sum() > 0))
            mrr = 0.0
            for i in range(min(5, len(sorted_labels))):
                if sorted_labels[i] == 1:
                    mrr = 1.0 / (i + 1)
                    break
            mrr_5_list.append(mrr)

        recall_5 = np.mean(recall_5_list)
        mrr_5 = np.mean(mrr_5_list)

        self.log('val/recall@5', recall_5, rank_zero_only=True)
        self.log('val/mrr@5', mrr_5, rank_zero_only=True)
        print(f"\n  [Ranking Eval] Recall@5: {recall_5:.4f}, MRR@5: {mrr_5:.4f} (on {eval_users} users)")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config['lr'],
            weight_decay=0.01,
        )
        # Cosine LR with warmup
        warmup_steps = self.config['warmup_steps']
        total_steps = self.trainer.estimated_stepping_batches

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.1, 0.5 * (1 + np.cos(np.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            }
        }

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def set_train_dataloader(self, dataloader):
        self._train_dataloader = dataloader

    def set_val_dataloader(self, dataloader):
        self._val_dataloader = dataloader


class ScoreCacheRefreshCallback(Callback):
    """
    Refresh score cache after each epoch for hard/top_k strategies.

    RoDPO approach: epoch 0 uses random negatives (no cache),
    subsequent epochs use cached scores from the previous epoch's model.
    Only computes on a sampled subset for efficiency (~5k users × 10 neg per batch).
    """

    def __init__(self, train_dataset: 'DPOPreferenceDataset', processor,
                 image_dir: str, strategy: str, cache_dir: str,
                 max_users: int = 5000, batch_size: int = 8):
        self.train_dataset = train_dataset
        self.processor = processor
        self.image_dir = image_dir
        self.strategy = strategy
        self.cache_dir = cache_dir
        self.max_users = max_users
        self.batch_size = batch_size

    def on_train_epoch_end(self, trainer, pl_module):
        """Refresh cache at end of epoch, ready for next epoch."""
        if self.strategy == 'random':
            return  # Random never needs cache

        if trainer.global_rank != 0:
            return

        epoch = trainer.current_epoch
        print(f"\n[ScoreCache] Refreshing score cache after epoch {epoch}...")
        print(f"  Sampling up to {self.max_users} users, batch_size={self.batch_size}")

        pl_module.model.eval()
        yes_token_id = pl_module.yes_token_id

        data = self.train_dataset.data
        sampled = data[:self.max_users] if len(data) > self.max_users else data

        cache = {}
        from tqdm import tqdm

        for sample in tqdm(sampled, desc=f"[ScoreCache] Epoch {epoch}"):
            user_id = sample['user_id']
            user_pref = sample['user_preference']
            neg_ids = sample['neg_item_ids']

            # Score all negatives for this user in batches
            scores = {}
            items = []
            for nid in neg_ids:
                items.append({
                    'title': sample['neg_titles'][str(nid)],
                    'image_path': sample['neg_image_paths'][str(nid)],
                })

            for i in range(0, len(items), self.batch_size):
                batch_items = items[i:i + self.batch_size]
                batch_ids = neg_ids[i:i + self.batch_size]

                texts = []
                images = []
                for item in batch_items:
                    prompt = PROMPT_TEMPLATE.format(
                        user_preference=user_pref,
                        title=item['title'],
                    )
                    text = f"[INST] <image>\n{prompt} [/INST]"
                    texts.append(text)
                    img_path = os.path.join(self.image_dir, item['image_path'])
                    if os.path.exists(img_path):
                        try:
                            images.append(Image.open(img_path).convert('RGB'))
                        except Exception:
                            images.append(Image.new('RGB', (336, 336), (0, 0, 0)))
                    else:
                        images.append(Image.new('RGB', (336, 336), (0, 0, 0)))

                images = resize_images_to_uniform(images)
                batch = self.processor(
                    text=texts, images=images,
                    padding=True, truncation=True,
                    max_length=512, return_tensors="pt"
                )
                batch = {k: v.to(pl_module.device) for k, v in batch.items()
                         if isinstance(v, torch.Tensor)}

                with torch.no_grad():
                    outputs = pl_module.model(**batch)
                    logits = outputs.logits
                    last_pos = batch['attention_mask'].sum(dim=1) - 1
                    bs = logits.size(0)
                    last_logits = logits[torch.arange(bs, device=logits.device), last_pos]
                    log_probs = F.log_softmax(last_logits, dim=-1)
                    batch_scores = log_probs[:, yes_token_id].cpu().tolist()

                for nid, score in zip(batch_ids, batch_scores):
                    scores[nid] = score

            cache[user_id] = scores

        # Update dataset's score cache
        self.train_dataset.set_score_cache(cache)
        print(f"[ScoreCache] Done: {len(cache)} users cached. "
              f"Next epoch will use {self.strategy} sampling.")

        # Save cache to disk
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_path = os.path.join(self.cache_dir, f"score_cache_epoch{epoch}.json")
        serializable = {
            str(uid): {str(iid): s for iid, s in user_scores.items()}
            for uid, user_scores in cache.items()
        }
        with open(cache_path, 'w') as f:
            json.dump(serializable, f)
        print(f"[ScoreCache] Saved to {cache_path}")

        pl_module.model.train()


class SaveDPOCallback(Callback):
    """Save DPO LoRA adapter after each epoch."""

    def __init__(self, save_dir: str, processor):
        self.save_dir = save_dir
        self.processor = processor

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.global_rank == 0:
            epoch = trainer.current_epoch
            save_path = os.path.join(self.save_dir, f"epoch_{epoch}")
            os.makedirs(save_path, exist_ok=True)
            pl_module.model.save_pretrained(save_path)
            self.processor.save_pretrained(save_path)
            print(f"\n[Saved DPO LoRA to {save_path}]")


class LogLRCallback(Callback):
    """Log learning rate to TensorBoard."""

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % 10 == 0:
            lr = trainer.optimizers[0].param_groups[0]['lr']
            pl_module.log('train/lr', lr, sync_dist=False)


def dpo_collate_fn(examples: List[dict]) -> List[dict]:
    """Simple pass-through collate. Processing happens in training_step."""
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', type=str, default='top_k',
                        choices=['random', 'hard', 'top_k'])
    parser.add_argument('--dataset', type=str, default='microlens',
                        choices=['video_games', 'microlens'])
    parser.add_argument('--model_path', type=str, default=None,
                        help='Local path to LLaVA model')
    parser.add_argument('--sft_lora_path', type=str, required=True,
                        help='Path to pre-trained SFT LoRA checkpoint')
    parser.add_argument('--beta', type=float, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--devices', type=int, default=8)
    parser.add_argument('--val_check_interval', type=int, default=None,
                        help='Validate every N training steps')
    parser.add_argument('--score_cache_path', type=str, default=None,
                        help='Pre-computed score cache JSON (skip random epoch 0)')
    parser.add_argument('--cache_max_users', type=int, default=5000,
                        help='Max users to score during cache refresh')
    parser.add_argument('--top_k', type=int, default=None,
                        help='K value for top_k negative sampling strategy')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='Resume training from this checkpoint path')
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    if args.beta:
        config['dpo_beta'] = args.beta
    if args.lr:
        config['lr'] = args.lr
    if args.epochs:
        config['max_epochs'] = args.epochs
    if args.val_check_interval:
        config['val_check_interval'] = args.val_check_interval
    if args.top_k:
        config['top_k'] = args.top_k

    # Resolve data paths
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    model_id = args.model_path or os.path.join(base_dir, 'models', 'llava-v1.6-mistral-7b-hf')

    if args.dataset == 'video_games':
        data_path = os.path.join(base_dir, 'data', 'amazon', 'dpo_ready', 'dpo_train.json')
        image_dir = os.path.join(base_dir, 'data', 'amazon', 'images')
        test_path = os.path.join(base_dir, 'data', 'amazon', 'dpo_ready', 'test.json')
    elif args.dataset == 'microlens':
        data_path = os.path.join(base_dir, 'data', 'microlens', 'dpo_ready', 'dpo_train.json')
        image_dir = os.path.join(base_dir, 'data', 'microlens', 'images')
        test_path = os.path.join(base_dir, 'data', 'microlens', 'dpo_ready', 'test.json')

    save_dir = os.path.join(base_dir, 'checkpoints', f'dpo_{args.dataset}_{args.strategy}')
    log_dir = os.path.join(base_dir, 'tb_logs')

    print("=" * 60)
    print("DPO Training Configuration:")
    print(f"  Strategy: {args.strategy}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Beta: {config['dpo_beta']}")
    print(f"  LR: {config['lr']}")
    print(f"  Epochs: {config['max_epochs']}")
    print(f"  Warmup steps: {config['warmup_steps']}")
    print(f"  Batch size per GPU: {config['batch_size']}")
    print(f"  Accumulate grad: {config['accumulate_grad_batches']}")
    print(f"  Effective batch: {config['batch_size'] * args.devices * config['accumulate_grad_batches']}")
    print(f"  Val check interval: {config['val_check_interval']} steps")
    print(f"  Val samples: {config['val_num_samples']}")
    print(f"  Eval users: {config['eval_num_users']}")
    print(f"  SFT LoRA: {args.sft_lora_path}")
    print(f"  Model: {model_id}")
    print(f"  Save dir: {save_dir}")
    print(f"  TensorBoard: {log_dir}")
    print("=" * 60)

    # ---- Step 1: Load base model ----
    print("\n[1/5] Loading base model...")
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        _attn_implementation="flash_attention_2",
    )

    # ---- Step 2: Merge SFT LoRA into base (becomes reference) ----
    print("[2/5] Merging SFT LoRA into base model (reference)...")
    model = PeftModel.from_pretrained(model, args.sft_lora_path)
    model = model.merge_and_unload()
    print("  SFT merged. This merged model = reference policy.")

    # ---- Step 3: Add new DPO LoRA adapter ----
    print("[3/5] Adding DPO LoRA adapter...")
    target_modules = find_all_linear_names(model)
    print(f"  LoRA target modules: {target_modules}")

    dpo_lora_config = LoraConfig(
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        target_modules=target_modules,
        init_lora_weights="gaussian",
    )
    model = get_peft_model(model, dpo_lora_config)
    model.print_trainable_parameters()

    # ---- Step 4: Load datasets ----
    print("[4/5] Loading datasets...")

    # Full DPO training data
    with open(data_path, 'r') as f:
        all_dpo_data = json.load(f)
    print(f"  Full DPO data: {len(all_dpo_data)} samples")

    # Split: last val_num_samples for validation
    val_size = config['val_num_samples']
    train_data_raw = all_dpo_data[:-val_size]
    val_data_raw = all_dpo_data[-val_size:]
    print(f"  Train split: {len(train_data_raw)}, Val split: {len(val_data_raw)}")

    # Create datasets
    # Write splits to tmp files for DPOPreferenceDataset (unique per strategy to avoid collision)
    train_tmp = os.path.join(base_dir, 'data', args.dataset if args.dataset != 'video_games' else 'amazon',
                             'dpo_ready', f'_dpo_train_split_{args.strategy}.json')
    val_tmp = os.path.join(base_dir, 'data', args.dataset if args.dataset != 'video_games' else 'amazon',
                           'dpo_ready', f'_dpo_val_split_{args.strategy}.json')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if local_rank == 0:
        with open(train_tmp, 'w') as f:
            json.dump(train_data_raw, f)
        with open(val_tmp, 'w') as f:
            json.dump(val_data_raw, f)
    if args.devices > 1:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.barrier()
        else:
            import time
            while not os.path.exists(train_tmp) or os.path.getsize(train_tmp) < 100:
                time.sleep(1)

    train_dataset = DPOPreferenceDataset(
        data_path=train_tmp,
        image_dir=image_dir,
        negative_strategy=args.strategy,
        top_k=config['top_k'],
    )
    val_dataset = DPOPreferenceDataset(
        data_path=val_tmp,
        image_dir=image_dir,
        negative_strategy='random',  # Val always uses random neg for stable measurement
    )
    print(f"  Train dataset: {len(train_dataset)}, Val dataset: {len(val_dataset)}")

    # Load pre-computed score cache if available
    if args.score_cache_path and os.path.exists(args.score_cache_path):
        print(f"  Loading pre-computed score cache: {args.score_cache_path}")
        with open(args.score_cache_path, 'r') as f:
            raw_cache = json.load(f)
        cache = {
            int(uid): {int(iid): score for iid, score in items.items()}
            for uid, items in raw_cache.items()
        }
        train_dataset.set_score_cache(cache)
        print(f"  Score cache loaded: {len(cache)} users → {args.strategy} from epoch 0!")
    elif args.strategy != 'random':
        print(f"  No score cache → epoch 0 will use random negatives (RoDPO paper approach)")

    # Load test data for ranking evaluation
    test_data = None
    if os.path.exists(test_path):
        with open(test_path, 'r') as f:
            test_data = json.load(f)
        print(f"  Test data: {len(test_data)} users (will eval {config['eval_num_users']})")

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        collate_fn=dpo_collate_fn,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2,
        collate_fn=dpo_collate_fn,
    )

    # ---- Step 5: Train ----
    print("[5/5] Setting up trainer...")

    pl_module = LlavaDPOModule(
        config, processor, model,
        val_dataset=val_dataset,
        test_data=test_data,
        image_dir=image_dir,
    )
    pl_module.set_train_dataloader(train_dataloader)
    pl_module.set_val_dataloader(val_dataloader)

    # TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=log_dir,
        name=f"dpo_{args.dataset}_{args.strategy}",
    )

    # Callbacks
    cache_dir = os.path.join(save_dir, 'score_caches')
    callbacks = [
        ScoreCacheRefreshCallback(
            train_dataset=train_dataset,
            processor=processor,
            image_dir=image_dir,
            strategy=args.strategy,
            cache_dir=cache_dir,
            max_users=args.cache_max_users,
            batch_size=8,
        ),
        SaveDPOCallback(save_dir, processor),
        LogLRCallback(),
        EarlyStopping(
            monitor='val/dpo_loss',
            patience=3,  # stop if val loss doesn't improve for 3 checks
            mode='min',
            verbose=True,
        ),
    ]

    trainer = L.Trainer(
        accelerator="gpu",
        devices=args.devices,
        strategy="deepspeed_stage_2",
        max_epochs=config['max_epochs'],
        accumulate_grad_batches=config['accumulate_grad_batches'],
        gradient_clip_val=config['gradient_clip_val'],
        precision="16-mixed",
        log_every_n_steps=10,
        val_check_interval=config['val_check_interval'],
        limit_train_batches=0.5,
        num_sanity_val_steps=2,
        callbacks=callbacks,
        logger=tb_logger,
    )

    print(f"\nStarting DPO training ({args.strategy})...")
    print(f"  Monitor: tensorboard --logdir {log_dir}")
    trainer.fit(pl_module, ckpt_path=args.ckpt_path)
    print(f"\nTraining complete! Checkpoints saved to: {save_dir}")

    # Clean up tmp files
    for f in [train_tmp, val_tmp]:
        if os.path.exists(f):
            os.remove(f)


if __name__ == '__main__':
    main()
