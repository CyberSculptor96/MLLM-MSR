"""
DPO Training Script for LLaVA-based Multimodal Sequential Recommendation.

Implements RoDPO (Stochastic Top-K Negative Sampling) on MLLM-MSR.
Supports three negative sampling strategies: random, hard (argmax), top_k.

Usage:
    python train_llava_dpo.py --strategy top_k --dataset video_games
    python train_llava_dpo.py --strategy random --dataset video_games
    python train_llava_dpo.py --strategy hard --dataset video_games
"""

import os
import sys
import argparse
import functools
from typing import List

import torch
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader
from PIL import Image

from transformers import AutoProcessor, LlavaNextForConditionalGeneration
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dpo_dataset import DPOPreferenceDataset

# ============ Paths ============
BASE_MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"

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
    """PyTorch Lightning module for DPO training of LLaVA recommender."""

    def __init__(self, config: dict, processor, model):
        super().__init__()
        self.config = config
        self.model = model
        self.processor = processor
        self.beta = config['dpo_beta']

        # Get token IDs
        self.yes_token_id = processor.tokenizer.convert_tokens_to_ids('Yes')
        self.no_token_id = processor.tokenizer.convert_tokens_to_ids('No')
        print(f"Token IDs - Yes: {self.yes_token_id}, No: {self.no_token_id}")

    def extract_log_prob_yes(self, input_ids, attention_mask, pixel_values, image_sizes):
        """
        Extract log P("Yes") at the first response position.

        The input is a prompt ending at [/INST], so the model's prediction
        at the last token position corresponds to the first response token.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
        )
        logits = outputs.logits  # (B, seq_len, vocab_size)

        # Find last non-padding position for each sample
        last_pos = attention_mask.sum(dim=1) - 1  # (B,)
        batch_size = logits.size(0)

        # Extract logits at last position
        last_logits = logits[torch.arange(batch_size, device=logits.device), last_pos]  # (B, V)

        # Compute log P("Yes")
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

    def training_step(self, batch, batch_idx):
        # batch is a list of dicts from DPOPreferenceDataset
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

        # Logging
        self.log('train/dpo_loss', dpo_loss, prog_bar=True, sync_dist=True)
        with torch.no_grad():
            reward_acc = (logits > 0).float().mean()
            margin = logits.mean()
            self.log('train/reward_acc', reward_acc, prog_bar=True, sync_dist=True)
            self.log('train/margin', margin, sync_dist=True)

        return dpo_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config['lr'],
            weight_decay=0.01,
        )
        return optimizer

    def train_dataloader(self):
        return self._train_dataloader

    def set_train_dataloader(self, dataloader):
        self._train_dataloader = dataloader


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


def dpo_collate_fn(examples: List[dict]) -> List[dict]:
    """Simple pass-through collate. Processing happens in training_step."""
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', type=str, default='top_k',
                        choices=['random', 'hard', 'top_k'])
    parser.add_argument('--dataset', type=str, default='video_games',
                        choices=['video_games', 'microlens'])
    parser.add_argument('--sft_lora_path', type=str, required=True,
                        help='Path to pre-trained SFT LoRA checkpoint')
    parser.add_argument('--beta', type=float, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--devices', type=int, default=8)
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    if args.beta:
        config['dpo_beta'] = args.beta
    if args.lr:
        config['lr'] = args.lr
    if args.epochs:
        config['max_epochs'] = args.epochs

    # Resolve data paths
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    if args.dataset == 'video_games':
        data_path = os.path.join(base_dir, 'data', 'amazon', 'dpo_ready', 'dpo_train.json')
        image_dir = os.path.join(base_dir, 'data', 'amazon', 'images')
    elif args.dataset == 'microlens':
        data_path = os.path.join(base_dir, 'data', 'microlens', 'dpo_ready', 'dpo_train.json')
        image_dir = os.path.join(base_dir, 'data', 'microlens', 'images')

    save_dir = os.path.join(base_dir, 'checkpoints', f'dpo_{args.dataset}_{args.strategy}')

    print(f"=" * 60)
    print(f"DPO Training Configuration:")
    print(f"  Strategy: {args.strategy}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Beta: {config['dpo_beta']}")
    print(f"  LR: {config['lr']}")
    print(f"  Epochs: {config['max_epochs']}")
    print(f"  SFT LoRA: {args.sft_lora_path}")
    print(f"  Save dir: {save_dir}")
    print(f"=" * 60)

    # ---- Step 1: Load base model ----
    print("\n[1/4] Loading base model...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        _attn_implementation="flash_attention_2",
    )

    # ---- Step 2: Merge SFT LoRA into base (becomes reference) ----
    print("[2/4] Merging SFT LoRA into base model (reference)...")
    model = PeftModel.from_pretrained(model, args.sft_lora_path)
    model = model.merge_and_unload()

    # ---- Step 3: Add new DPO LoRA adapter ----
    print("[3/4] Adding DPO LoRA adapter...")
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

    # ---- Step 4: Create dataset and dataloader ----
    print("[4/4] Loading DPO dataset...")
    dataset = DPOPreferenceDataset(
        data_path=data_path,
        image_dir=image_dir,
        negative_strategy=args.strategy,
        top_k=config['top_k'],
    )
    print(f"  Dataset size: {len(dataset)} samples")

    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        collate_fn=dpo_collate_fn,
        pin_memory=True,
    )

    # ---- Create Lightning module and train ----
    pl_module = LlavaDPOModule(config, processor, model)
    pl_module.set_train_dataloader(dataloader)

    trainer = L.Trainer(
        accelerator="gpu",
        devices=args.devices,
        strategy="deepspeed_stage_2",
        max_epochs=config['max_epochs'],
        accumulate_grad_batches=config['accumulate_grad_batches'],
        gradient_clip_val=config['gradient_clip_val'],
        precision="16-mixed",
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        callbacks=[SaveDPOCallback(save_dir, processor)],
    )

    print("\nStarting DPO training...")
    trainer.fit(pl_module)
    print(f"\nTraining complete! Checkpoints saved to: {save_dir}")


if __name__ == '__main__':
    main()
