"""
SFT Training Script for LLaVA-based Multimodal Sequential Recommendation.

Trains LLaVA with LoRA on binary recommendation task (Yes/No).
Compatible with both Amazon Video_Games and MicroLens datasets.
Produces SFT LoRA checkpoint that serves as the base for DPO training.

Usage:
    python train_llava_sft.py --dataset video_games --devices 8
    python train_llava_sft.py --dataset microlens --devices 8
"""

import os
import sys
import argparse
from typing import List

import torch
import lightning as L
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader
from PIL import Image, ImageOps

from transformers import AutoProcessor, LlavaNextForConditionalGeneration
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dpo_dataset import SFTDataset

# ============ Config ============
BASE_MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"

DEFAULT_CONFIG = {
    'max_epochs': 4,
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
    """Find all linear layer names for LoRA targeting."""
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['multi_modal_projector', 'vision_model']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
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


class LlavaSFTModule(L.LightningModule):
    """PyTorch Lightning module for SFT training of LLaVA recommender."""

    def __init__(self, config: dict, processor, model):
        super().__init__()
        self.config = config
        self.model = model
        self.processor = processor

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, pixel_values, image_sizes, labels = batch
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            labels=labels,
        )
        loss = outputs.loss
        self.log("train/loss", loss, prog_bar=True, batch_size=input_ids.size(0), sync_dist=True)
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


def sft_collate_fn(examples: List[dict], processor, max_length=1024):
    """Collate function for SFT training."""
    images = []
    texts = []
    for ex in examples:
        prompt = PROMPT_TEMPLATE.format(
            user_preference=ex['user_preference'],
            title=ex['item_title'],
        )
        # Include the answer in the text for teacher forcing
        text = f"[INST] <image>\n{prompt} [/INST] {ex['ground_truth']}"
        texts.append(text)
        images.append(ex['image'])

    images = resize_images_to_uniform(images)
    batch = processor(
        text=texts, images=images,
        padding=True, truncation=True,
        max_length=max_length, return_tensors="pt"
    )

    # Create labels (mask padding tokens)
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels

    return (
        batch["input_ids"],
        batch["attention_mask"],
        batch["pixel_values"],
        batch["image_sizes"],
        batch["labels"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='video_games',
                        choices=['video_games', 'microlens'])
    parser.add_argument('--model_path', type=str, default=None,
                        help='Local path to LLaVA model (overrides BASE_MODEL_ID)')
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
    model_id = args.model_path or BASE_MODEL_ID

    if args.dataset == 'video_games':
        data_path = os.path.join(base_dir, 'data', 'amazon', 'dpo_ready', 'sft_train.json')
        image_dir = os.path.join(base_dir, 'data', 'amazon', 'images')
    elif args.dataset == 'microlens':
        data_path = os.path.join(base_dir, 'data', 'microlens', 'dpo_ready', 'sft_train.json')
        image_dir = os.path.join(base_dir, 'data', 'microlens', 'images')

    save_dir = os.path.join(base_dir, 'checkpoints', f'sft_{args.dataset}')

    print("=" * 60)
    print("SFT Training Configuration:")
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
    print("\n[1/3] Loading base model...")
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "right"
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        _attn_implementation="flash_attention_2",
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
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- Load dataset ----
    print("[3/3] Loading SFT dataset...")
    dataset = SFTDataset(data_path=data_path, image_dir=image_dir)
    print(f"  Dataset size: {len(dataset)} samples")

    import functools
    collate_fn = functools.partial(
        sft_collate_fn,
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
    pl_module = LlavaSFTModule(config, processor, model)
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
        callbacks=[SaveSFTCallback(save_dir, processor)],
    )

    print("\nStarting SFT training...")
    trainer.fit(pl_module)
    print(f"\nTraining complete! Checkpoints saved to: {save_dir}")


if __name__ == '__main__':
    main()
