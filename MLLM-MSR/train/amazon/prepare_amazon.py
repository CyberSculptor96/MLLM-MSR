"""
Amazon Video_Games Data Preparation for MLLM-MSR DPO Training.

Reads pre-processed TSV/CSV files and builds:
1. SFT training dataset (HuggingFace Dataset format)
2. DPO training dataset (JSON with preference pairs)
3. Test dataset (JSON with 1 pos + 20 neg per user)

User preference: simplified version (concatenating recent item titles).
"""

import os
import json
import random
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

# ============ Config ============
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'amazon', 'processed')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'amazon', 'dpo_ready')
DATASET = 'Video_Games'
MAX_HISTORY = 10       # max recent items for user preference
NUM_NEG_TEST = 20      # negatives per user in test set
NUM_NEG_DPO = 50       # expanded negative pool for DPO (for top-k sampling)
RANDOM_SEED = 42


def load_item_info(data_dir, dataset):
    """Load item descriptions (id -> title, image_url)."""
    path = os.path.join(data_dir, f'{dataset}_item_desc.tsv')
    item_info = {}
    with open(path, 'r', encoding='utf-8') as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split('\t', 2)  # split only first 2 tabs
            if len(parts) < 2:
                continue
            item_id = int(parts[0])
            image_url = parts[1] if len(parts) > 1 else ''
            summary = parts[2] if len(parts) > 2 else ''
            # Extract title (first sentence of summary)
            title = summary.split('.')[0].strip() if summary else f"Item {item_id}"
            item_info[item_id] = {
                'title': title,
                'image_url': image_url,
            }
    return item_info


def load_user_items(data_dir, dataset, split='train'):
    """Load user-item interactions with negatives."""
    if split == 'full':
        path = os.path.join(data_dir, f'{dataset}_user_items_negs.tsv')
        df = pd.read_csv(path, sep='\t', header=0, names=['user_id', 'pos', 'neg'])
    else:
        path = os.path.join(data_dir, f'{dataset}_user_items_negs_{split}.csv')
        df = pd.read_csv(path, sep='\t', header=None, names=['user_id', 'pos', 'neg'])

    users = []
    for _, row in df.iterrows():
        pos_items = [int(x) for x in str(row['pos']).split(',')]
        neg_items = [int(x) for x in str(row['neg']).split(',')]
        users.append({
            'user_id': int(row['user_id']),
            'pos_items': pos_items,
            'neg_items': neg_items,
        })
    return users


def build_user_preference(pos_items, item_info, max_history=10):
    """Build simplified user preference from recent item titles."""
    history_items = pos_items[:-1][-max_history:]  # exclude last (target), take recent N
    titles = []
    for iid in history_items:
        if iid in item_info:
            titles.append(item_info[iid]['title'][:80])  # truncate long titles
        else:
            titles.append(f"Item {iid}")
    if not titles:
        return "The user has limited interaction history."
    return "The user recently interacted with: " + "; ".join(titles)


def expand_negative_pool(user_neg_items, all_item_ids, user_pos_set, target_size=50):
    """Expand the negative pool to target_size by adding random items."""
    neg_set = set(user_neg_items)
    available = list(set(all_item_ids) - user_pos_set - neg_set)
    needed = target_size - len(neg_set)
    if needed > 0 and available:
        extra = random.sample(available, min(needed, len(available)))
        neg_set.update(extra)
    return list(neg_set)


def build_sft_dataset(users, item_info, output_path):
    """Build SFT training data: each sample is (user_pref, item_title, label=Yes/No)."""
    samples = []
    for user in tqdm(users, desc="Building SFT dataset"):
        pos_items = user['pos_items']
        if len(pos_items) < 2:
            continue

        target_item = pos_items[-1]  # last item as positive
        user_pref = build_user_preference(pos_items, item_info, MAX_HISTORY)

        # Positive sample
        if target_item in item_info:
            samples.append({
                'user_id': user['user_id'],
                'user_preference': user_pref,
                'item_id': target_item,
                'item_title': item_info[target_item]['title'],
                'image_path': f"{target_item}.jpg",
                'ground_truth': 'Yes',
            })

        # One negative sample (random from neg pool)
        neg_id = random.choice(user['neg_items'])
        if neg_id in item_info:
            samples.append({
                'user_id': user['user_id'],
                'user_preference': user_pref,
                'item_id': neg_id,
                'item_title': item_info[neg_id]['title'],
                'image_path': f"{neg_id}.jpg",
                'ground_truth': 'No',
            })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(samples, f)
    print(f"SFT dataset: {len(samples)} samples -> {output_path}")
    return samples


def build_dpo_dataset(users, item_info, all_item_ids, output_path):
    """Build DPO training data with expanded negative pools."""
    samples = []
    for user in tqdm(users, desc="Building DPO dataset"):
        pos_items = user['pos_items']
        if len(pos_items) < 2:
            continue

        target_item = pos_items[-1]
        if target_item not in item_info:
            continue

        user_pref = build_user_preference(pos_items, item_info, MAX_HISTORY)
        pos_set = set(pos_items)

        # Expand negative pool for top-k sampling
        neg_pool = expand_negative_pool(
            user['neg_items'], all_item_ids, pos_set, target_size=NUM_NEG_DPO
        )
        # Filter to items with info
        neg_pool = [nid for nid in neg_pool if nid in item_info]

        if not neg_pool:
            continue

        neg_titles = {str(nid): item_info[nid]['title'] for nid in neg_pool}
        neg_image_paths = {str(nid): f"{nid}.jpg" for nid in neg_pool}

        samples.append({
            'user_id': user['user_id'],
            'user_preference': user_pref,
            'pos_item_id': target_item,
            'pos_title': item_info[target_item]['title'],
            'pos_image_path': f"{target_item}.jpg",
            'neg_item_ids': neg_pool,
            'neg_titles': neg_titles,
            'neg_image_paths': neg_image_paths,
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(samples, f)
    print(f"DPO dataset: {len(samples)} samples -> {output_path}")
    return samples


def build_test_dataset(users, item_info, all_item_ids, output_path):
    """Build test data: 1 pos + 20 neg per user for ranking evaluation."""
    samples = []
    for user in tqdm(users, desc="Building test dataset"):
        pos_items = user['pos_items']
        if len(pos_items) < 2:
            continue

        target_item = pos_items[-1]
        if target_item not in item_info:
            continue

        user_pref = build_user_preference(pos_items, item_info, MAX_HISTORY)
        pos_set = set(pos_items)

        # Get 20 negatives
        neg_pool = list(set(user['neg_items']) - pos_set)
        if len(neg_pool) < NUM_NEG_TEST:
            available = list(set(all_item_ids) - pos_set - set(neg_pool))
            extra = random.sample(available, min(NUM_NEG_TEST - len(neg_pool), len(available)))
            neg_pool.extend(extra)
        neg_pool = [nid for nid in neg_pool if nid in item_info][:NUM_NEG_TEST]

        if len(neg_pool) < NUM_NEG_TEST:
            continue

        # Build candidate list: [pos] + [neg * 20]
        candidates = []
        candidates.append({
            'item_id': target_item,
            'title': item_info[target_item]['title'],
            'image_path': f"{target_item}.jpg",
            'label': 1,
        })
        for nid in neg_pool:
            candidates.append({
                'item_id': nid,
                'title': item_info[nid]['title'],
                'image_path': f"{nid}.jpg",
                'label': 0,
            })

        samples.append({
            'user_id': user['user_id'],
            'user_preference': user_pref,
            'candidates': candidates,
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(samples, f)
    print(f"Test dataset: {len(samples)} users -> {output_path}")
    return samples


def main():
    random.seed(RANDOM_SEED)

    print(f"Loading item info for {DATASET}...")
    item_info = load_item_info(DATA_DIR, DATASET)
    all_item_ids = list(item_info.keys())
    print(f"  {len(item_info)} items loaded")

    print(f"Loading train users...")
    train_users = load_user_items(DATA_DIR, DATASET, 'train')
    print(f"  {len(train_users)} train users")

    print(f"Loading test users...")
    test_users = load_user_items(DATA_DIR, DATASET, 'test')
    print(f"  {len(test_users)} test users")

    # Build datasets
    build_sft_dataset(train_users, item_info,
                      os.path.join(OUTPUT_DIR, 'sft_train.json'))

    build_dpo_dataset(train_users, item_info, all_item_ids,
                      os.path.join(OUTPUT_DIR, 'dpo_train.json'))

    build_test_dataset(test_users, item_info, all_item_ids,
                       os.path.join(OUTPUT_DIR, 'test.json'))

    # Save image URL mapping for download
    image_urls = {f"{iid}.jpg": info['image_url']
                  for iid, info in item_info.items() if info['image_url']}
    with open(os.path.join(OUTPUT_DIR, 'image_urls.json'), 'w') as f:
        json.dump(image_urls, f)
    print(f"Image URL mapping: {len(image_urls)} items -> {OUTPUT_DIR}/image_urls.json")

    print("\nDone! Next step: download images using image_urls.json")


if __name__ == '__main__':
    main()
