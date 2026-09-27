#!/usr/bin/env python3
"""
Offline validation harness. Holds out a slice of train_source1 that was never
touched by any existing training run (rows beyond the first 100k of the file,
which is the max any existing script reads for training/hard-negatives),
builds the FULL blocking index over train_source2+3, generates candidates,
and scores multiple candidate matching strategies against train_ground_truth
using the exact official macro F_0.5 formula.
"""
import os
import sys
import time
import random
import pickle
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from normalization import get_core_name, get_compact_name, normalize_address, extract_address_digits
from blocking import CandidateGenerator
from features import compute_pair_features

DELIM = "\t"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".eval_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

TOTAL_S1_ROWS = 2_206_821  # data rows in train_source1.tsv, excluding header (wc -l - 1)
HOLDOUT_TAIL_ROWS = 200_000  # fixed reserved tail -- stays valid regardless of training size,
                             # so training can scale up (currently up to HOLDOUT_START rows)
                             # without ever invalidating this holdout split
HOLDOUT_START = TOTAL_S1_ROWS - HOLDOUT_TAIL_ROWS  # = 2,006,821
HOLDOUT_SAMPLE_SIZE = 20_000
SEED = 1234


def prep_record(rec):
    eid, name, addr, ctry = rec
    cn = get_core_name(name)
    comp = get_compact_name(cn)
    norm_a = normalize_address(addr, ctry)
    dig = extract_address_digits(norm_a)
    return (eid, name, addr, ctry, cn, comp, dig, norm_a)


def load_holdout_s1():
    path = os.path.join(DATA_DIR, "train", "train_source1.tsv")
    rows = []
    with open(path, encoding="utf-8") as f:
        f.readline()
        for i, line in enumerate(f):
            if i < HOLDOUT_START:
                continue
            p = line.rstrip("\n").split(DELIM)
            if p and p[0]:
                rows.append((p[0], p[1] if len(p) > 1 else "", p[2] if len(p) > 2 else "", p[3] if len(p) > 3 else ""))
    random.Random(SEED).shuffle(rows)
    sample = rows[:HOLDOUT_SAMPLE_SIZE]
    print(f"Holdout pool available: {len(rows):,} rows (fixed reserved tail, row {HOLDOUT_START:,}+); sampled {len(sample):,}")
    ctry_counts = defaultdict(int)
    for r in sample:
        ctry_counts[r[3]] += 1
    print(f"  Sample country breakdown: {dict(ctry_counts)}")
    return sample


def load_ground_truth(needed_ids):
    path = os.path.join(DATA_DIR, "train", "train_ground_truth.tsv")
    gt = {}
    with open(path, encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\n").split(DELIM)
            if p[0] in needed_ids:
                gt[p[0]] = set(p[1].split(",")) if (len(p) > 1 and p[1]) else set()
    return gt


def build_country_generator(country, max_candidates=15):
    """Build a candidate index scoped to ONE country's pool (mirrors how the
    real inference pipeline partitions by country) to keep peak memory bounded."""
    print(f"Building candidate generator for country={country} (train_source2+3, filtered)...")
    t0 = time.time()
    cand_raw = []
    for src in ["2", "3"]:
        path = os.path.join(DATA_DIR, "train", f"train_source{src}.tsv")
        with open(path, encoding="utf-8") as f:
            f.readline()
            for line in f:
                p = line.rstrip("\n").split(DELIM)
                if p and p[0] and len(p) > 3 and p[3] == country:
                    cand_raw.append((p[0], p[1] if len(p) > 1 else "", p[2] if len(p) > 2 else "", p[3]))
    print(f"  Loaded {len(cand_raw):,} candidate records in {time.time()-t0:.1f}s")

    t0 = time.time()
    gen = CandidateGenerator(max_candidates=max_candidates)
    gen.fit_candidates(cand_raw)
    print(f"  Fitted index in {time.time()-t0:.1f}s")
    return gen


def macro_f05(pred_map, gt_map, ids):
    """pred_map: id -> set(predicted match ids). gt_map: id -> set(true match ids)."""
    scores = []
    tp_tot = fp_tot = fn_tot = 0
    for sid in ids:
        pred = pred_map.get(sid, set())
        true = gt_map.get(sid, set())
        tp = len(pred & true)
        fp = len(pred - true)
        fn = len(true - pred)
        tp_tot += tp; fp_tot += fp; fn_tot += fn
        if not true:
            scores.append(1.0 if not pred else 0.0)
        else:
            if not pred:
                scores.append(0.0)
            else:
                p = tp / len(pred)
                r = tp / len(true)
                denom = 0.25 * p + r
                scores.append((1.25 * p * r / denom) if denom > 0 else 0.0)
    precision = tp_tot / max(1, tp_tot + fp_tot)
    recall = tp_tot / max(1, tp_tot + fn_tot)
    return sum(scores) / len(scores), precision, recall


if __name__ == "__main__":
    print("This module provides shared utilities; run compare_strategies.py to execute the evaluation.")
