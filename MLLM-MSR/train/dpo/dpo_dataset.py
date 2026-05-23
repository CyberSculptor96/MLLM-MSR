"""
DPO Preference Pair Dataset for MLLM-MSR.

Supports three negative sampling strategies:
  - 'random': uniform random from negative pool
  - 'hard': argmax scored item from negative pool
  - 'top_k': uniform random from top-K scored items in pool

Compatible with both Amazon and MicroLens datasets.
"""

import os
import json
import random
from typing import Dict, List, Optional

from PIL import Image
from torch.utils.data import Dataset


class DPOPreferenceDataset(Dataset):
    """
    DPO preference pair dataset.

    Each sample provides:
      - user_preference: str
      - pos_title: str
      - pos_image: PIL.Image
      - neg_title: str (selected by negative sampling strategy)
      - neg_image: PIL.Image
    """

    def __init__(
        self,
        data_path: str,
        image_dir: str,
        negative_strategy: str = 'random',
        top_k: int = 50,
        placeholder_image_size: tuple = (336, 336),
    ):
        """
        Args:
            data_path: path to JSON file with DPO training data
            image_dir: directory containing item images (named {item_id}.jpg)
            negative_strategy: 'random', 'hard', or 'top_k'
            top_k: K value for top-k sampling
            placeholder_image_size: size of placeholder image when file not found
        """
        with open(data_path, 'r') as f:
            self.data = json.load(f)

        self.image_dir = image_dir
        self.strategy = negative_strategy
        self.top_k = top_k
        self.placeholder_size = placeholder_image_size
        self.score_cache: Dict[int, Dict[int, float]] = {}

    def __len__(self):
        return len(self.data)

    def set_score_cache(self, cache: Dict[int, Dict[int, float]]):
        """Update score cache for hard/top_k sampling strategies."""
        self.score_cache = cache

    def _load_image(self, image_path: str) -> Image.Image:
        """Load image from disk, return placeholder if not found."""
        full_path = os.path.join(self.image_dir, image_path)
        if os.path.exists(full_path):
            try:
                return Image.open(full_path).convert('RGB')
            except Exception:
                pass
        # Return a black placeholder image
        return Image.new('RGB', self.placeholder_size, (0, 0, 0))

    def _select_negative(self, user_id: int, neg_item_ids: List[int]) -> int:
        """Select a negative item based on the sampling strategy."""
        if self.strategy == 'random' or not self.score_cache:
            return random.choice(neg_item_ids)

        user_scores = self.score_cache.get(user_id, {})
        if not user_scores:
            return random.choice(neg_item_ids)

        if self.strategy == 'hard':
            # Argmax: pick the highest-scored negative
            return max(neg_item_ids, key=lambda x: user_scores.get(x, 0.0))

        elif self.strategy == 'top_k':
            # Top-K: sort by score, pick randomly from top-K
            scored = [(nid, user_scores.get(nid, 0.0)) for nid in neg_item_ids]
            scored.sort(key=lambda x: -x[1])
            k = min(self.top_k, len(scored))
            top_k_items = scored[:k]
            return random.choice(top_k_items)[0]

        else:
            return random.choice(neg_item_ids)

    def __getitem__(self, idx) -> dict:
        sample = self.data[idx]
        user_id = sample['user_id']

        # Positive item
        pos_title = sample['pos_title']
        pos_image = self._load_image(sample['pos_image_path'])

        # Negative item (strategy-dependent)
        neg_item_ids = sample['neg_item_ids']
        neg_id = self._select_negative(user_id, neg_item_ids)

        neg_title = sample['neg_titles'][str(neg_id)]
        neg_image_path = sample['neg_image_paths'][str(neg_id)]
        neg_image = self._load_image(neg_image_path)

        return {
            'user_preference': sample['user_preference'],
            'pos_title': pos_title,
            'pos_image': pos_image,
            'neg_title': neg_title,
            'neg_image': neg_image,
        }


class SFTDataset(Dataset):
    """
    SFT dataset for LLaVA-based recommendation.

    Each sample provides:
      - user_preference: str
      - item_title: str
      - image: PIL.Image
      - ground_truth: 'Yes' or 'No'
    """

    def __init__(self, data_path: str, image_dir: str, placeholder_image_size=(336, 336)):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.image_dir = image_dir
        self.placeholder_size = placeholder_image_size

    def __len__(self):
        return len(self.data)

    def _load_image(self, image_path: str) -> Image.Image:
        full_path = os.path.join(self.image_dir, image_path)
        if os.path.exists(full_path):
            try:
                return Image.open(full_path).convert('RGB')
            except Exception:
                pass
        return Image.new('RGB', self.placeholder_size, (0, 0, 0))

    def __getitem__(self, idx) -> dict:
        sample = self.data[idx]
        return {
            'user_preference': sample['user_preference'],
            'item_title': sample['item_title'],
            'image': self._load_image(sample['image_path']),
            'ground_truth': sample['ground_truth'],
        }
