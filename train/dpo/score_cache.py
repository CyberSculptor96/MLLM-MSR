"""
Score Cache for Top-K and Argmax negative sampling.

Pre-computes P(Yes) scores for candidate negative items,
enabling informed negative selection during DPO training.
"""

import os
import json
import torch
import torch.nn.functional as F
from typing import Dict, List
from tqdm import tqdm
from PIL import Image


PROMPT_TEMPLATE = (
    "Based on the previous interaction history, the user's preference "
    "can be summarized as: {user_preference}\n"
    "Please predict whether this user would interact with the item. "
    "The item's title is '{title}'.\n"
    "Please only response 'yes' or 'no'."
)


class ScoreCache:
    """
    Pre-computes P(Yes) scores for candidate items.
    Used by DPOPreferenceDataset for 'hard' and 'top_k' strategies.
    """

    def __init__(self, model, processor, image_dir: str, device: str = 'cuda'):
        self.model = model
        self.processor = processor
        self.image_dir = image_dir
        self.device = device
        self.cache: Dict[int, Dict[int, float]] = {}

        # Get Yes token ID (with space prefix, matching training)
        self.yes_token_id = processor.tokenizer.encode('Yes', add_special_tokens=False)[0]

    def _load_image(self, image_path: str) -> Image.Image:
        full_path = os.path.join(self.image_dir, image_path)
        if os.path.exists(full_path):
            try:
                return Image.open(full_path).convert('RGB')
            except Exception:
                pass
        return Image.new('RGB', (336, 336), (0, 0, 0))

    @torch.no_grad()
    def _score_batch(self, user_preference: str, items: List[dict]) -> List[float]:
        """Score a batch of items for one user."""
        texts = []
        images = []
        for item in items:
            prompt = PROMPT_TEMPLATE.format(
                user_preference=user_preference,
                title=item['title'],
            )
            text = f"[INST] <image>\n{prompt} [/INST]"
            texts.append(text)
            images.append(self._load_image(item['image_path']))

        batch = self.processor(
            text=texts, images=images,
            padding=True, truncation=True,
            max_length=1024, return_tensors="pt"
        )
        batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        outputs = self.model(**batch)
        logits = outputs.logits  # (B, seq_len, V)

        # Extract logits at last position
        last_pos = batch['attention_mask'].sum(dim=1) - 1
        batch_size = logits.size(0)
        last_logits = logits[torch.arange(batch_size, device=logits.device), last_pos]

        # P(Yes) via softmax over [No, Yes]
        log_probs = F.log_softmax(last_logits, dim=-1)
        scores = log_probs[:, self.yes_token_id].cpu().tolist()
        return scores

    @torch.no_grad()
    def refresh(self, data_path: str, batch_size: int = 4, max_users: int = None):
        """
        Refresh score cache for all users in the dataset.

        Args:
            data_path: path to DPO training JSON
            batch_size: number of items to score per batch
            max_users: limit number of users (for debugging)
        """
        with open(data_path, 'r') as f:
            data = json.load(f)

        if max_users:
            data = data[:max_users]

        self.model.eval()
        self.cache = {}

        for sample in tqdm(data, desc="Computing score cache"):
            user_id = sample['user_id']
            user_pref = sample['user_preference']
            neg_ids = sample['neg_item_ids']

            # Build items list
            items = []
            for nid in neg_ids:
                items.append({
                    'title': sample['neg_titles'][str(nid)],
                    'image_path': sample['neg_image_paths'][str(nid)],
                })

            # Score in batches
            scores = {}
            for i in range(0, len(items), batch_size):
                batch_items = items[i:i + batch_size]
                batch_ids = neg_ids[i:i + batch_size]
                batch_scores = self._score_batch(user_pref, batch_items)
                for nid, score in zip(batch_ids, batch_scores):
                    scores[nid] = score

            self.cache[user_id] = scores

        print(f"Score cache refreshed: {len(self.cache)} users")
        return self.cache

    def save(self, path: str):
        """Save cache to disk."""
        # Convert int keys to str for JSON serialization
        serializable = {
            str(uid): {str(iid): score for iid, score in items.items()}
            for uid, items in self.cache.items()
        }
        with open(path, 'w') as f:
            json.dump(serializable, f)
        print(f"Score cache saved to {path}")

    def load(self, path: str):
        """Load cache from disk."""
        with open(path, 'r') as f:
            data = json.load(f)
        self.cache = {
            int(uid): {int(iid): score for iid, score in items.items()}
            for uid, items in data.items()
        }
        print(f"Score cache loaded: {len(self.cache)} users from {path}")
