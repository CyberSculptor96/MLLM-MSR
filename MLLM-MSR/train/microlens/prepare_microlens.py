"""
MicroLens-50k Data Preparation for MLLM-MSR DPO Training.

Reads raw MicroLens CSV files and builds:
1. SFT training dataset (JSON)
2. DPO training dataset (JSON with expanded negative pools)
3. Test dataset (JSON with 1 pos + 20 neg per user)

User preference: simplified version (concatenating recent video titles).
"""

import os
import json
import random
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

# ============ Config ============
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'microlens')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'microlens', 'dpo_ready')
IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'microlens', 'images')
MAX_HISTORY = 10       # max recent items for user preference
MIN_INTERACTIONS = 5   # minimum interactions per user
NUM_NEG_TEST = 20      # negatives per user in test set
NUM_NEG_DPO = 50       # expanded negative pool for DPO
RANDOM_SEED = 42


def load_data():
    """Load and process raw MicroLens data."""
    # Load interaction pairs
    pairs_path = os.path.join(DATA_DIR, 'MicroLens-50k_pairs.csv')
    pairs_df = pd.read_csv(pairs_path)
    print(f"  Raw interactions: {len(pairs_df)}")

    # Load titles
    titles_path = os.path.join(DATA_DIR, 'MicroLens-50k_titles.csv')
    titles_df = pd.read_csv(titles_path)
    item_titles = {int(k): str(v) for k, v in zip(titles_df['item'], titles_df['title'])}
    print(f"  Items with titles: {len(item_titles)}")

    return pairs_df, item_titles


def build_user_sequences(pairs_df, min_interactions=5):
    """Build time-sorted user interaction sequences."""
    # Sort by user and timestamp
    pairs_df = pairs_df.sort_values(['user', 'timestamp'])

    user_sequences = defaultdict(list)
    for _, row in pairs_df.iterrows():
        user_sequences[int(row['user'])].append(int(row['item']))

    # Filter users with minimum interactions
    filtered = {uid: items for uid, items in user_sequences.items()
                if len(items) >= min_interactions}
    print(f"  Users with >= {min_interactions} interactions: {len(filtered)}")
    return filtered


def leave_one_out_split(user_sequences):
    """
    Leave-one-out split:
      - test: last item
      - val: second-to-last
      - train: rest
    """
    train_data = {}
    val_data = {}
    test_data = {}

    for uid, items in user_sequences.items():
        if len(items) < 3:
            continue
        train_data[uid] = items[:-2]
        val_data[uid] = items[-2]
        test_data[uid] = items[-1]

    return train_data, val_data, test_data


def sample_negatives(user_items_set, all_item_ids, num_neg):
    """Sample random negatives avoiding user's positive items."""
    available = list(set(all_item_ids) - user_items_set)
    if len(available) < num_neg:
        return available
    return random.sample(available, num_neg)


def build_user_preference(train_items, item_titles, max_history=10):
    """Build simplified user preference from recent video titles."""
    recent = train_items[-max_history:]
    titles = []
    for iid in recent:
        title = item_titles.get(iid, f"Video {iid}")
        titles.append(str(title)[:80])
    if not titles:
        return "The user has limited interaction history."
    return "The user recently watched: " + "; ".join(titles)


def main():
    random.seed(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading MicroLens data...")
    pairs_df, item_titles = load_data()
    all_item_ids = list(item_titles.keys())

    print("\nBuilding user sequences...")
    user_sequences = build_user_sequences(pairs_df, MIN_INTERACTIONS)

    print("\nPerforming leave-one-out split...")
    train_data, val_data, test_data = leave_one_out_split(user_sequences)
    print(f"  Train users: {len(train_data)}")
    print(f"  Test users: {len(test_data)}")

    # ---- Build SFT Dataset ----
    print("\nBuilding SFT dataset...")
    sft_samples = []
    for uid, train_items in tqdm(train_data.items(), desc="SFT"):
        if not train_items:
            continue
        target_item = train_items[-1]  # predict last train item
        user_pref = build_user_preference(train_items[:-1], item_titles, MAX_HISTORY)

        # Positive
        sft_samples.append({
            'user_id': uid,
            'user_preference': user_pref,
            'item_id': target_item,
            'item_title': str(item_titles.get(target_item, f"Video {target_item}")),
            'image_path': f"{target_item}.jpg",
            'ground_truth': 'Yes',
        })

        # One negative
        user_pos_set = set(user_sequences[uid])
        neg = sample_negatives(user_pos_set, all_item_ids, 1)
        if neg:
            sft_samples.append({
                'user_id': uid,
                'user_preference': user_pref,
                'item_id': neg[0],
                'item_title': str(item_titles.get(neg[0], f"Video {neg[0]}")),
                'image_path': f"{neg[0]}.jpg",
                'ground_truth': 'No',
            })

    with open(os.path.join(OUTPUT_DIR, 'sft_train.json'), 'w') as f:
        json.dump(sft_samples, f)
    print(f"  SFT dataset: {len(sft_samples)} samples")

    # ---- Build DPO Dataset ----
    print("\nBuilding DPO dataset...")
    dpo_samples = []
    for uid, train_items in tqdm(train_data.items(), desc="DPO"):
        if not train_items:
            continue
        target_item = test_data[uid]  # DPO targets the test item
        user_pref = build_user_preference(train_items, item_titles, MAX_HISTORY)
        user_pos_set = set(user_sequences[uid])

        # Expanded negative pool
        neg_pool = sample_negatives(user_pos_set, all_item_ids, NUM_NEG_DPO)
        if not neg_pool:
            continue

        neg_titles = {str(nid): str(item_titles.get(nid, f"Video {nid}")) for nid in neg_pool}
        neg_image_paths = {str(nid): f"{nid}.jpg" for nid in neg_pool}

        dpo_samples.append({
            'user_id': uid,
            'user_preference': user_pref,
            'pos_item_id': target_item,
            'pos_title': str(item_titles.get(target_item, f"Video {target_item}")),
            'pos_image_path': f"{target_item}.jpg",
            'neg_item_ids': neg_pool,
            'neg_titles': neg_titles,
            'neg_image_paths': neg_image_paths,
        })

    with open(os.path.join(OUTPUT_DIR, 'dpo_train.json'), 'w') as f:
        json.dump(dpo_samples, f)
    print(f"  DPO dataset: {len(dpo_samples)} samples")

    # ---- Build Test Dataset ----
    print("\nBuilding test dataset...")
    test_samples = []
    for uid in tqdm(test_data.keys(), desc="Test"):
        target_item = test_data[uid]
        train_items = train_data[uid]
        user_pref = build_user_preference(train_items, item_titles, MAX_HISTORY)
        user_pos_set = set(user_sequences[uid])

        neg_pool = sample_negatives(user_pos_set, all_item_ids, NUM_NEG_TEST)
        if len(neg_pool) < NUM_NEG_TEST:
            continue

        candidates = []
        candidates.append({
            'item_id': target_item,
            'title': str(item_titles.get(target_item, f"Video {target_item}")),
            'image_path': f"{target_item}.jpg",
            'label': 1,
        })
        for nid in neg_pool:
            candidates.append({
                'item_id': nid,
                'title': str(item_titles.get(nid, f"Video {nid}")),
                'image_path': f"{nid}.jpg",
                'label': 0,
            })

        test_samples.append({
            'user_id': uid,
            'user_preference': user_pref,
            'candidates': candidates,
        })

    with open(os.path.join(OUTPUT_DIR, 'test.json'), 'w') as f:
        json.dump(test_samples, f)
    print(f"  Test dataset: {len(test_samples)} users")

    # ---- Summary ----
    print("\n" + "=" * 50)
    print("MicroLens Data Preparation Complete!")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  SFT samples: {len(sft_samples)}")
    print(f"  DPO samples: {len(dpo_samples)}")
    print(f"  Test users: {len(test_samples)}")
    print(f"\n  NOTE: Cover images needed in {IMAGE_DIR}")
    print(f"  Download from: https://github.com/westlake-repl/MicroLens")
    print("=" * 50)


if __name__ == '__main__':
    main()
