"""
DPO Training Script for Qwen2.5-VL-based Multimodal Sequential Recommendation.

Adapted from train_llava_dpo.py for Qwen2.5-VL-7B-Instruct architecture.
Implements RoDPO (Stochastic Top-K Negative Sampling) on MLLM-MSR.

Key differences from LLaVA version:
  - Qwen2_5_VLForConditionalGeneration model class
  - Chat template (messages format) for prompts
  - Different processor API (pixel_values + image_grid_thw)
  - LoRA targets exclude visual and projector modules

Usage:
    python train_qwen_dpo.py --config configs/hard.yaml --sft_lora_path <path>
    python train_qwen_dpo.py --config configs/top_k_5.yaml --sft_lora_path <path>
"""

import os
import sys
import json
import argparse
import functools
import random
from typing import List, Dict

import yaml

import torch
import torch.nn.functional as F
import numpy as np
import lightning as L
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel, LoraConfig, get_peft_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dpo_dataset import DPOPreferenceDataset

# ============ Config Loading ============
CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs')


def load_config(config_path=None):
    """Load config: default.yaml -> config_path (override) -> returns dict."""
    default_path = os.path.join(CONFIG_DIR, 'default.yaml')
    with open(default_path, 'r') as f:
        config = yaml.safe_load(f)
    if config_path:
        with open(config_path, 'r') as f:
            overrides = yaml.safe_load(f) or {}
        config.update(overrides)
    return config


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
    multimodal_keywords = ['visual', 'multi_modal_projector', 'merger', 'lm_head']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[-1])
    return list(lora_module_names)


def build_qwen_messages(user_preference: str, title: str, image_path: str) -> List[dict]:
    """Build Qwen2.5-VL chat messages for a single item."""
    prompt = PROMPT_TEMPLATE.format(user_preference=user_preference, title=title)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return messages


class QwenDPOModule(L.LightningModule):
    """PyTorch Lightning module for DPO training with Qwen2.5-VL."""

    def __init__(self, config: dict, processor, model, val_dataset=None,
                 test_data=None, image_dir=None):
        super().__init__()
        self.config = config
        self.model = model
        self.processor = processor
        self.beta = config['dpo_beta']
        self.val_dataset = val_dataset
        self.test_data = test_data
        self.image_dir = image_dir

        # Token IDs for Yes/No
        self.yes_token_id = processor.tokenizer.encode('Yes', add_special_tokens=False)[0]
        self.no_token_id = processor.tokenizer.encode('No', add_special_tokens=False)[0]
        print(f"Token IDs - Yes: {self.yes_token_id}, No: {self.no_token_id}")

        self.best_val_loss = float('inf')

    def _prepare_inputs(self, items_data: List[dict], key_prefix: str) -> dict:
        """Build processor-ready inputs for pos or neg items using Qwen chat template."""
        all_texts = []
        all_images = []

        for item in items_data:
            title = item[f'{key_prefix}_title']
            image = item[f'{key_prefix}_image']
            all_images.append(image)

            # Build chat messages (without actual image processing - we handle images separately)
            prompt = PROMPT_TEMPLATE.format(
                user_preference=item['user_preference'],
                title=title,
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            all_texts.append(text)

        # Process batch through Qwen processor
        batch = self.processor(
            text=all_texts,
            images=all_images,
            padding=True,
            truncation=True,
            max_length=self.config['max_length'],
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

    def extract_log_prob_yes(self, **batch_inputs) -> torch.Tensor:
        """Extract log P("Yes") at the last token position."""
        outputs = self.model(**batch_inputs)
        logits = outputs.logits  # (B, seq_len, vocab_size)
        attention_mask = batch_inputs['attention_mask']
        last_pos = attention_mask.sum(dim=1) - 1  # (B,)
        batch_size = logits.size(0)
        last_logits = logits[torch.arange(batch_size, device=logits.device), last_pos]
        log_probs = F.log_softmax(last_logits, dim=-1)
        return log_probs[:, self.yes_token_id]  # (B,)

    def _compute_dpo_loss(self, batch):
        """Compute DPO loss and metrics."""
        pos_inputs = self._prepare_inputs(batch, 'pos')
        neg_inputs = self._prepare_inputs(batch, 'neg')

        # Policy forward
        policy_log_p_pos = self.extract_log_prob_yes(**pos_inputs)
        policy_log_p_neg = self.extract_log_prob_yes(**neg_inputs)

        # Reference forward (disable DPO adapter)
        with torch.no_grad():
            self.model.disable_adapter_layers()
            ref_log_p_pos = self.extract_log_prob_yes(**pos_inputs)
            ref_log_p_neg = self.extract_log_prob_yes(**neg_inputs)
            self.model.enable_adapter_layers()

        # DPO Loss
        pi_logratios = policy_log_p_pos - policy_log_p_neg
        ref_logratios = ref_log_p_pos - ref_log_p_neg
        logits = pi_logratios - ref_logratios
        dpo_loss = -F.logsigmoid(self.beta * logits).mean()

        reward_acc = (logits > 0).float().mean()
        margin = logits.mean()

        return dpo_loss, reward_acc, margin, policy_log_p_pos.mean(), policy_log_p_neg.mean()

    def training_step(self, batch, batch_idx):
        dpo_loss, reward_acc, margin, log_p_pos, log_p_neg = self._compute_dpo_loss(batch)

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
                # Build single-item input
                prompt = PROMPT_TEMPLATE.format(
                    user_preference=user_pref,
                    title=cand['title'],
                )
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                img_path = os.path.join(self.image_dir, cand['image_path'])
                if os.path.exists(img_path):
                    try:
                        image = Image.open(img_path).convert('RGB')
                    except Exception:
                        image = Image.new('RGB', (336, 336), (0, 0, 0))
                else:
                    image = Image.new('RGB', (336, 336), (0, 0, 0))

                batch_inputs = self.processor(
                    text=[text], images=[image],
                    padding=True, truncation=True,
                    max_length=1024, return_tensors="pt"
                )
                batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()
                                if isinstance(v, torch.Tensor)}

                with torch.no_grad():
                    outputs = self.model(**batch_inputs)
                    logits = outputs.logits
                    last_pos = batch_inputs['attention_mask'].sum(dim=1) - 1
                    last_logits = logits[0, last_pos[0]]
                    yes_no_logits = torch.tensor([
                        last_logits[self.yes_token_id],
                        last_logits[self.no_token_id]
                    ])
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
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}
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
    """Refresh score cache after each epoch for hard/top_k strategies."""

    def __init__(self, train_dataset, processor, image_dir, strategy,
                 cache_dir, max_users=5000, batch_size=8, skip=False):
        self.train_dataset = train_dataset
        self.processor = processor
        self.image_dir = image_dir
        self.strategy = strategy
        self.cache_dir = cache_dir
        self.max_users = max_users
        self.batch_size = batch_size
        self.skip = skip

    def on_train_epoch_end(self, trainer, pl_module):
        """Refresh score cache at end of each epoch."""
        if self.skip:
            print(f"[ScoreCache] Skipped (--skip_cache_refresh)")
            return
        if self.strategy == 'random':
            return

        epoch = trainer.current_epoch
        print(f"\n[ScoreCache] Refreshing scores after epoch {epoch}...")

        pl_module.model.eval()
        yes_token_id = pl_module.yes_token_id

        # Get unique users from training data
        all_data = self.train_dataset.data
        user_neg_map = {}
        for sample in all_data:
            uid = sample['user_id']
            if uid not in user_neg_map:
                user_neg_map[uid] = {
                    'neg_ids': sample['neg_item_ids'],
                    'neg_titles': sample['neg_titles'],
                    'neg_image_paths': sample['neg_image_paths'],
                    'user_preference': sample['user_preference'],
                }

        users_to_score = list(user_neg_map.keys())[:self.max_users]
        print(f"[ScoreCache] Scoring {len(users_to_score)} users...")

        cache = {}
        for uid in users_to_score:
            info = user_neg_map[uid]
            neg_ids = info['neg_ids']
            scores = {}

            # Batch score negatives
            for i in range(0, len(neg_ids), self.batch_size):
                batch_ids = neg_ids[i:i + self.batch_size]
                texts = []
                images = []

                for nid in batch_ids:
                    title = info['neg_titles'][str(nid)]
                    prompt = PROMPT_TEMPLATE.format(
                        user_preference=info['user_preference'],
                        title=title,
                    )
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ]
                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    texts.append(text)

                    img_path = os.path.join(self.image_dir, info['neg_image_paths'][str(nid)])
                    if os.path.exists(img_path):
                        try:
                            images.append(Image.open(img_path).convert('RGB'))
                        except Exception:
                            images.append(Image.new('RGB', (336, 336), (0, 0, 0)))
                    else:
                        images.append(Image.new('RGB', (336, 336), (0, 0, 0)))

                batch_inputs = self.processor(
                    text=texts, images=images,
                    padding=True, truncation=True,
                    max_length=512, return_tensors="pt"
                )
                batch_inputs = {k: v.to(pl_module.device) for k, v in batch_inputs.items()
                                if isinstance(v, torch.Tensor)}

                with torch.no_grad():
                    outputs = pl_module.model(**batch_inputs)
                    logits = outputs.logits
                    last_pos = batch_inputs['attention_mask'].sum(dim=1) - 1
                    bs = logits.size(0)
                    last_logits = logits[torch.arange(bs, device=logits.device), last_pos]
                    log_probs = F.log_softmax(last_logits, dim=-1)
                    batch_scores = log_probs[:, yes_token_id].cpu().tolist()

                for nid, score in zip(batch_ids, batch_scores):
                    scores[nid] = score

            cache[uid] = scores

        self.train_dataset.set_score_cache(cache)
        print(f"[ScoreCache] Done: {len(cache)} users cached.")

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
    """Save DPO LoRA adapter every val_check_interval steps and after each epoch."""

    def __init__(self, save_dir, processor):
        self.save_dir = save_dir
        self.processor = processor

    def on_validation_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            step = trainer.global_step
            save_path = os.path.join(self.save_dir, f"step_{step}")
            os.makedirs(save_path, exist_ok=True)
            pl_module.model.save_pretrained(save_path)
            self.processor.save_pretrained(save_path)
            print(f"\n[Saved DPO LoRA to {save_path} (step {step})]")

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.global_rank == 0:
            epoch = trainer.current_epoch
            save_path = os.path.join(self.save_dir, f"epoch_{epoch}")
            os.makedirs(save_path, exist_ok=True)
            pl_module.model.save_pretrained(save_path)
            self.processor.save_pretrained(save_path)
            print(f"\n[Saved DPO LoRA to {save_path}]")


class LogLRCallback(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % 10 == 0:
            lr = trainer.optimizers[0].param_groups[0]['lr']
            pl_module.log('train/lr', lr, sync_dist=False)


def dpo_collate_fn(examples: List[dict]) -> List[dict]:
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file (overrides default.yaml)')
    parser.add_argument('--strategy', type=str, default=None,
                        choices=['random', 'hard', 'top_k'])
    parser.add_argument('--dataset', type=str, default='microlens',
                        choices=['video_games', 'microlens'])
    parser.add_argument('--model_path', type=str, default=None,
                        help='Local path to Qwen2.5-VL model')
    parser.add_argument('--sft_lora_path', type=str, required=True,
                        help='Path to pre-trained SFT LoRA checkpoint')
    parser.add_argument('--beta', type=float, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--devices', type=int, default=8)
    parser.add_argument('--val_check_interval', type=int, default=None,
                        help='Validate every N training steps')
    parser.add_argument('--score_cache_path', type=str, default=None,
                        help='Pre-computed score cache JSON')
    parser.add_argument('--cache_max_users', type=int, default=5000)
    parser.add_argument('--top_k', type=int, default=None)
    parser.add_argument('--skip_cache_refresh', action='store_true')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='Resume training from Lightning checkpoint (.ckpt)')
    parser.add_argument('--tb_version', type=int, default=None,
                        help='TensorBoard version number (for resume)')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
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

    strategy = args.strategy or config.get('strategy', 'top_k')

    # Resolve paths
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    model_id = args.model_path or os.path.join(base_dir, 'models', 'Qwen2.5-VL-7B-Instruct')

    if args.dataset == 'video_games':
        data_path = os.path.join(base_dir, 'data', 'amazon', 'dpo_ready', 'dpo_train.json')
        image_dir = os.path.join(base_dir, 'data', 'amazon', 'images')
        test_path = os.path.join(base_dir, 'data', 'amazon', 'dpo_ready', 'test.json')
    elif args.dataset == 'microlens':
        data_path = os.path.join(base_dir, 'data', 'microlens', 'dpo_ready', 'dpo_train.json')
        image_dir = os.path.join(base_dir, 'data', 'microlens', 'images')
        test_path = os.path.join(base_dir, 'data', 'microlens', 'dpo_ready', 'test.json')

    suffix = f'qwen_dpo_{args.dataset}_{strategy}'
    if strategy == 'top_k' and config['top_k'] != 50:
        suffix += f"_k{config['top_k']}"
    save_dir = os.path.join(base_dir, 'checkpoints', suffix)
    log_dir = os.path.join(base_dir, 'tb_logs')

    print("=" * 60)
    print("Qwen2.5-VL DPO Training Configuration:")
    print(f"  Strategy: {strategy}")
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
    print("\n[1/5] Loading Qwen2.5-VL base model...")
    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,  # Conservative for training memory
    )
    processor.tokenizer.padding_side = "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
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

    with open(data_path, 'r') as f:
        all_dpo_data = json.load(f)
    print(f"  Full DPO data: {len(all_dpo_data)} samples")

    val_size = config['val_num_samples']
    train_data_raw = all_dpo_data[:-val_size]
    val_data_raw = all_dpo_data[-val_size:]
    print(f"  Train split: {len(train_data_raw)}, Val split: {len(val_data_raw)}")

    # Write splits to tmp files (only rank 0 writes, others wait)
    split_suffix = f"{strategy}_k{config['top_k']}" if strategy == 'top_k' else strategy
    data_subdir = args.dataset if args.dataset != 'video_games' else 'amazon'
    train_tmp = os.path.join(base_dir, 'data', data_subdir,
                             'dpo_ready', f'_qwen_dpo_train_split_{split_suffix}.json')
    val_tmp = os.path.join(base_dir, 'data', data_subdir,
                           'dpo_ready', f'_qwen_dpo_val_split_{split_suffix}.json')

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
        negative_strategy=strategy,
        top_k=config['top_k'],
    )
    val_dataset = DPOPreferenceDataset(
        data_path=val_tmp,
        image_dir=image_dir,
        negative_strategy='random',
        top_k=config['top_k'],
    )

    # Load pre-computed score cache if provided
    if args.score_cache_path and os.path.exists(args.score_cache_path):
        with open(args.score_cache_path, 'r') as f:
            raw_cache = json.load(f)
        cache = {
            int(uid): {int(iid): score for iid, score in items.items()}
            for uid, items in raw_cache.items()
        }
        train_dataset.set_score_cache(cache)
        print(f"  Score cache loaded: {len(cache)} users → {strategy} from epoch 0!")
    elif strategy != 'random':
        print(f"  No score cache → epoch 0 will use random negatives (RoDPO paper approach)")

    # Load test data
    with open(test_path, 'r') as f:
        test_data = json.load(f)
    print(f"  Test data: {len(test_data)} users")

    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'],
        shuffle=True, num_workers=4, collate_fn=dpo_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'],
        shuffle=False, num_workers=2, collate_fn=dpo_collate_fn
    )

    # ---- Step 5: Setup trainer ----
    print("[5/5] Setting up trainer...")
    pl_module = QwenDPOModule(
        config=config,
        processor=processor,
        model=model,
        val_dataset=val_dataset,
        test_data=test_data,
        image_dir=image_dir,
    )
    pl_module.set_train_dataloader(train_loader)
    pl_module.set_val_dataloader(val_loader)

    # TensorBoard logger
    tb_kwargs = dict(save_dir=log_dir, name=suffix)
    if args.tb_version is not None:
        tb_kwargs['version'] = args.tb_version
    tb_logger = TensorBoardLogger(**tb_kwargs)

    # Callbacks
    cache_dir = os.path.join(save_dir, 'score_caches')
    ckpt_dir = os.path.join(save_dir, 'lightning_ckpt')
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename='last',
            save_last=True,
            every_n_train_steps=config['val_check_interval'],
        ),
        SaveDPOCallback(save_dir, processor),
        ScoreCacheRefreshCallback(
            train_dataset=train_dataset,
            processor=processor,
            image_dir=image_dir,
            strategy=strategy,
            cache_dir=cache_dir,
            max_users=args.cache_max_users,
            batch_size=8,
            skip=args.skip_cache_refresh,
        ),
        LogLRCallback(),
        EarlyStopping(
            monitor='val/dpo_loss',
            patience=3,
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
        precision="bf16-mixed",
        log_every_n_steps=10,
        val_check_interval=config['val_check_interval'],
        limit_train_batches=0.5,
        num_sanity_val_steps=2,
        callbacks=callbacks,
        logger=tb_logger,
    )

    print(f"\nStarting Qwen2.5-VL DPO training ({strategy})...")
    print(f"  Monitor: tensorboard --logdir {log_dir}")
    trainer.fit(pl_module, ckpt_path=args.ckpt_path)
    print(f"\nTraining complete! Checkpoints saved to: {save_dir}")

    # Clean up tmp files
    for f in [train_tmp, val_tmp]:
        if os.path.exists(f):
            os.remove(f)


if __name__ == '__main__':
    main()
