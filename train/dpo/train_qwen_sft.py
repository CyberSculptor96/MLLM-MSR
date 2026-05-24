"""
SFT Training Script for Qwen2.5-VL-based Multimodal Sequential Recommendation.

Trains Qwen2.5-VL with LoRA on binary recommendation task (Yes/No).
Produces SFT LoRA checkpoint that serves as the base for DPO training.

Usage:
    python train_qwen_sft.py --dataset microlens --devices 4
    python train_qwen_sft.py --dataset microlens --devices 8 --epochs 2
"""

import os
import sys
import argparse
import functools
from typing import List

import torch
import lightning as L
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader
from PIL import Image

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dpo_dataset import SFTDataset

# ============ Config ============
DEFAULT_CONFIG = {
    'max_epochs': 3,
    'lr': 2e-5,
    'batch_size': 1,
    'accumulate_grad_batches': 4,
    'max_length': 1024,
    'gradient_clip_val': 1.0,
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
    multimodal_keywords = ['visual', 'multi_modal_projector', 'merger', 'lm_head']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[-1])
    return list(lora_module_names)


class QwenSFTModule(L.LightningModule):
    """PyTorch Lightning module for SFT training of Qwen2.5-VL recommender."""

    def __init__(self, config: dict, processor, model):
        super().__init__()
        self.config = config
        self.model = model
        self.processor = processor

    def training_step(self, batch, batch_idx):
        # batch is a dict from collate_fn
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log("train/loss", loss, prog_bar=True,
                 batch_size=batch['input_ids'].size(0), sync_dist=True)
        return loss

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


class SaveSFTCallback(Callback):
    """Save SFT LoRA adapter after each epoch."""

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
            print(f"\n[Saved SFT LoRA to {save_path}]")


def qwen_sft_collate_fn(examples: List[dict], processor, max_length=1024):
    """Collate function for Qwen2.5-VL SFT training."""
    all_texts = []
    all_images = []

    for ex in examples:
        prompt = PROMPT_TEMPLATE.format(
            user_preference=ex['user_preference'],
            title=ex['item_title'],
        )
        # Build messages with answer for teacher forcing
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": ex['ground_truth']},
                ],
            },
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        all_texts.append(text)
        all_images.append(ex['image'])

    # Process through Qwen processor
    batch = processor(
        text=all_texts,
        images=all_images,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    # Create labels: mask everything except the assistant response
    input_ids = batch["input_ids"]
    labels = input_ids.clone()

    # Mask padding
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Mask user turn (everything before assistant response)
    # Find the assistant header token pattern and mask everything before it
    # For Qwen2.5-VL, the assistant turn starts after "<|im_start|>assistant\n"
    im_start_id = processor.tokenizer.encode('<|im_start|>', add_special_tokens=False)
    assistant_ids = processor.tokenizer.encode('assistant', add_special_tokens=False)

    for i in range(input_ids.size(0)):
        ids = input_ids[i].tolist()
        # Find last occurrence of im_start + assistant pattern
        # This marks the beginning of the assistant response
        assistant_start = -1
        for j in range(len(ids) - len(im_start_id) - len(assistant_ids)):
            if (ids[j:j + len(im_start_id)] == im_start_id and
                    ids[j + len(im_start_id):j + len(im_start_id) + len(assistant_ids)] == assistant_ids):
                # Mask up to after "assistant\n"
                assistant_start = j + len(im_start_id) + len(assistant_ids) + 1  # +1 for \n
        if assistant_start > 0:
            labels[i, :assistant_start] = -100

    batch["labels"] = labels

    # Return as dict (Qwen model expects kwargs)
    return {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='microlens',
                        choices=['video_games', 'microlens'])
    parser.add_argument('--model_path', type=str, default=None,
                        help='Local path to Qwen2.5-VL model')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--devices', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=None)
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    if args.epochs:
        config['max_epochs'] = args.epochs
    if args.lr:
        config['lr'] = args.lr
    if args.batch_size:
        config['batch_size'] = args.batch_size

    # Resolve paths
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    model_id = args.model_path or os.path.join(base_dir, 'models', 'Qwen2.5-VL-7B-Instruct')

    if args.dataset == 'video_games':
        data_path = os.path.join(base_dir, 'data', 'amazon', 'dpo_ready', 'sft_train.json')
        image_dir = os.path.join(base_dir, 'data', 'amazon', 'images')
    elif args.dataset == 'microlens':
        data_path = os.path.join(base_dir, 'data', 'microlens', 'dpo_ready', 'sft_train.json')
        image_dir = os.path.join(base_dir, 'data', 'microlens', 'images')

    save_dir = os.path.join(base_dir, 'checkpoints', f'qwen_sft_{args.dataset}')

    print("=" * 60)
    print("Qwen2.5-VL SFT Training Configuration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Model: {model_id}")
    print(f"  LR: {config['lr']}")
    print(f"  Epochs: {config['max_epochs']}")
    print(f"  Batch size per GPU: {config['batch_size']}")
    print(f"  Accumulate grad: {config['accumulate_grad_batches']}")
    print(f"  Effective batch: {config['batch_size'] * args.devices * config['accumulate_grad_batches']}")
    print(f"  Data: {data_path}")
    print(f"  Images: {image_dir}")
    print(f"  Save dir: {save_dir}")
    print("=" * 60)

    # ---- Load model ----
    print("\n[1/3] Loading Qwen2.5-VL base model...")
    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )
    processor.tokenizer.padding_side = "right"  # SFT uses right padding
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    # ---- Apply LoRA ----
    print("[2/3] Applying LoRA...")
    target_modules = find_all_linear_names(model)
    print(f"  LoRA target modules: {target_modules}")

    lora_config = LoraConfig(
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        target_modules=target_modules,
        init_lora_weights="gaussian",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- Load dataset ----
    print("[3/3] Loading SFT dataset...")
    dataset = SFTDataset(data_path=data_path, image_dir=image_dir)
    print(f"  Dataset size: {len(dataset)} samples")

    collate_fn = functools.partial(
        qwen_sft_collate_fn,
        processor=processor,
        max_length=config['max_length'],
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # ---- Train ----
    pl_module = QwenSFTModule(config, processor, model)
    pl_module.set_train_dataloader(dataloader)

    trainer = L.Trainer(
        accelerator="gpu",
        devices=args.devices,
        strategy="deepspeed_stage_2",
        max_epochs=config['max_epochs'],
        accumulate_grad_batches=config['accumulate_grad_batches'],
        gradient_clip_val=config['gradient_clip_val'],
        precision="bf16-mixed",
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        callbacks=[SaveSFTCallback(save_dir, processor)],
    )

    print("\nStarting Qwen2.5-VL SFT training...")
    trainer.fit(pl_module)
    print(f"\nTraining complete! Checkpoints saved to: {save_dir}")


if __name__ == '__main__':
    main()
