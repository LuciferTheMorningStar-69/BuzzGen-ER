#!/usr/bin/env python3
"""Same diagnostic as embed_diagnostic.py, but with LaBSE (a much heavier,
purpose-built cross-lingual bitext-alignment model) instead of the small
MiniLM model, which tested poorly on exactly the hard cross-script cases
this dataset needs (e.g. 0.056 cosine similarity for a real Gujarati/
English match pair). LaBSE is too slow to run at full scale on the cloud
container's 4 CPU cores, but may be practical on faster local hardware
(e.g. Apple Silicon with MPS acceleration) -- this script auto-detects
the best available device (mps > cuda > cpu).

Run from the repo root: python3 code/embed_diagnostic_labse.py
Requires: pip install sentence-transformers torch
(Downloads ~1.8GB for LaBSE on first run.)
"""
import os
import sys
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from offline_eval import DATA_DIR, DELIM

import torch
from sentence_transformers import SentenceTransformer
import numpy as np

random.seed(42)

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")

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

s1_names = {}
with open(os.path.join(DATA_DIR, "train", "train_source1.tsv"), encoding="utf-8") as f:
    f.readline()
    for line in f:
        p = line.rstrip("\n").split(DELIM)
        if p and p[0] in s1_needed:
            s1_names[p[0]] = p[1] if len(p) > 1 else ""

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

model = SentenceTransformer('sentence-transformers/LaBSE', device=device)

s1_texts = [s1_names[a] for a, b in valid_pairs]
match_texts = [match_names[b] for a, b in valid_pairs]

import time
t0 = time.time()
emb_s1 = model.encode(s1_texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
emb_match = model.encode(match_texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
elapsed = time.time() - t0
rate = 2 * len(s1_texts) / elapsed
print(f"Encoded {2*len(s1_texts)} texts in {elapsed:.1f}s ({rate:.1f} sentences/sec)")
total_candidates_estimate = 4_133_346 + 6_186_873
print(f"Projected time for full India+US candidate pool at this rate: {total_candidates_estimate/rate/60:.1f} min ({total_candidates_estimate/rate/3600:.2f} hrs)")

true_sims = np.sum(emb_s1 * emb_match, axis=1)
rng = np.random.RandomState(0)
shuffled_idx = rng.permutation(len(emb_match))
random_sims = np.sum(emb_s1 * emb_match[shuffled_idx], axis=1)

print(f"\nTrue-match cosine similarity:   mean={true_sims.mean():.3f}  median={np.median(true_sims):.3f}  p10={np.percentile(true_sims,10):.3f}")
print(f"Random-pair cosine similarity:  mean={random_sims.mean():.3f}  median={np.median(random_sims):.3f}  p90={np.percentile(random_sims,90):.3f}")
frac_true_above_random_p90 = (true_sims > np.percentile(random_sims, 90)).mean()
print(f"Fraction of true matches scoring above the random-pair 90th percentile: {frac_true_above_random_p90*100:.1f}%")

print("\n--- Specifically checking the hard Indic cross-script cases (lowest-similarity true matches) ---")
order = np.argsort(true_sims)
for i in list(order[:10]):
    a, b = valid_pairs[i]
    print(f"  sim={true_sims[i]:.3f}  '{s1_texts[i][:40]}' <-> '{match_texts[i][:40]}'")
