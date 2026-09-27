#!/usr/bin/env python3
"""Quick throughput benchmark for CPU-only sentence embedding encoding,
before committing to full-scale (millions of records) encoding. Uses a
small, memory-bounded sample so it can run safely alongside another
heavy job without risking OOM."""
import time
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from offline_eval import DATA_DIR, DELIM

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('/root/.cache/mini_lm_multilingual', device='cpu')

names = []
path = os.path.join(DATA_DIR, "train", "train_source2.tsv")
with open(path, encoding="utf-8") as f:
    f.readline()
    for i, line in enumerate(f):
        if i >= 5000:
            break
        p = line.rstrip("\n").split(DELIM)
        if p and len(p) > 1:
            names.append(p[1])

print(f"Encoding {len(names):,} raw business names (no anyascii transliteration)...")
t0 = time.time()
embs = model.encode(names, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
elapsed = time.time() - t0
rate = len(names) / elapsed
print(f"Done in {elapsed:.1f}s -- {rate:.1f} sentences/sec")
print(f"Embedding shape: {embs.shape}")

total_candidates_estimate = 4_133_346 + 6_186_873  # India + US candidate pools
est_time = total_candidates_estimate / rate
print(f"\nProjected time to encode full India+US candidate pool ({total_candidates_estimate:,} records): {est_time:.0f}s ({est_time/60:.1f} min)")
