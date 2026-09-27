#!/usr/bin/env python3
"""Cheap, targeted diagnostic (a few hundred pairs, seconds of compute):
does the multilingual embedding model actually separate true cross-source
matches from random noise? If cosine similarity for true matches isn't
meaningfully higher than for random pairs, a full-scale (hours-long)
embedding-based blocking run isn't worth attempting -- same logic as the
earlier MinHash/LSH check that correctly predicted it wouldn't help."""
import os
import sys
import random
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from offline_eval import DATA_DIR, DELIM

from sentence_transformers import SentenceTransformer
import numpy as np

random.seed(42)

# Load a sample of ground truth pairs (S1 id -> matched ids)
gt_path = os.path.join(DATA_DIR, "train", "train_ground_truth.tsv")
pairs = []
with open(gt_path, encoding="utf-8") as f:
    f.readline()
    for line in f:
        p = line.rstrip("\n").split(DELIM)
        if len(p) > 1 and p[1]:
            matches = p[1].split(",")
            if matches:
                pairs.append((p[0], matches[0]))
random.shuffle(pairs)
sample_pairs = pairs[:400]
s1_needed = {p[0] for p in sample_pairs}
match_needed = {p[1] for p in sample_pairs}

print(f"Sampled {len(sample_pairs)} true-match ground-truth pairs")

# Look up names for S1 side
s1_names = {}
with open(os.path.join(DATA_DIR, "train", "train_source1.tsv"), encoding="utf-8") as f:
    f.readline()
    for line in f:
        p = line.rstrip("\n").split(DELIM)
        if p and p[0] in s1_needed:
            s1_names[p[0]] = p[1] if len(p) > 1 else ""

# Look up names for matched side (could be in source2 or source3)
match_names = {}
for src in ["2", "3"]:
    with open(os.path.join(DATA_DIR, "train", f"train_source{src}.tsv"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\n").split(DELIM)
            if p and p[0] in match_needed:
                match_names[p[0]] = p[1] if len(p) > 1 else ""

valid_pairs = [(a, b) for a, b in sample_pairs if a in s1_names and b in match_names and s1_names[a] and match_names[b]]
print(f"Valid pairs with both names present: {len(valid_pairs)}")

model = SentenceTransformer('/root/.cache/mini_lm_multilingual', device='cpu')

s1_texts = [s1_names[a] for a, b in valid_pairs]
match_texts = [match_names[b] for a, b in valid_pairs]

emb_s1 = model.encode(s1_texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
emb_match = model.encode(match_texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)

# True-match cosine similarity (diagonal)
true_sims = np.sum(emb_s1 * emb_match, axis=1)

# Random (mismatched) pair similarity -- shuffle the match side
rng = np.random.RandomState(0)
shuffled_idx = rng.permutation(len(emb_match))
random_sims = np.sum(emb_s1 * emb_match[shuffled_idx], axis=1)

print(f"\nTrue-match cosine similarity:   mean={true_sims.mean():.3f}  median={np.median(true_sims):.3f}  p10={np.percentile(true_sims,10):.3f}")
print(f"Random-pair cosine similarity:  mean={random_sims.mean():.3f}  median={np.median(random_sims):.3f}  p90={np.percentile(random_sims,90):.3f}")
print(f"\nSeparation (true p10 vs random p90): {np.percentile(true_sims,10):.3f} vs {np.percentile(random_sims,90):.3f}")
frac_true_above_random_p90 = (true_sims > np.percentile(random_sims, 90)).mean()
print(f"Fraction of true matches scoring above the random-pair 90th percentile: {frac_true_above_random_p90*100:.1f}%")

print("\n--- Sample true-match pairs and their similarity ---")
order = np.argsort(true_sims)
for i in list(order[:5]) + list(order[-5:]):
    a, b = valid_pairs[i]
    print(f"  sim={true_sims[i]:.3f}  '{s1_texts[i][:40]}' <-> '{match_texts[i][:40]}'")
