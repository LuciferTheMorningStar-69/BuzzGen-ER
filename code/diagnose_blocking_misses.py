#!/usr/bin/env python3
"""Finds true matches missed by blocking (not in the generated candidate set)
and prints the S1/missed-candidate pairs for manual inspection, plus quick
heuristics (shared tokens? shared digits?) to spot patterns."""
import os
import sys
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from normalization import get_core_name, extract_address_digits, normalize_address
from offline_eval import load_holdout_s1, load_ground_truth

CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", ".eval_cache", "holdout_cands_feats.pkl")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
DELIM = "\t"

s1_sample = load_holdout_s1()
s1_by_id = {r[0]: r for r in s1_sample}
ids = list(s1_by_id.keys())
gt_map = load_ground_truth(set(ids))

with open(CACHE_PATH, "rb") as f:
    all_cands, all_feats, recall_hits, recall_total, total_cands = pickle.load(f)

missed = []  # (sid, missed_cid)
for sid in ids:
    true = gt_map.get(sid, set())
    if not true:
        continue
    got = {c[0] for c in all_cands[sid]}
    for cid in (true - got):
        missed.append((sid, cid))

print(f"Total missed true-match pairs: {len(missed):,} (out of {recall_total:,} total true matches)")

import random
random.seed(3)
sample = random.sample(missed, min(30, len(missed)))
needed_cids = {cid for _, cid in sample}

cand_rec = {}
for src in ["2", "3"]:
    path = os.path.join(DATA_DIR, "train", f"train_source{src}.tsv")
    with open(path, encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\n").split(DELIM)
            if p[0] in needed_cids:
                cand_rec[p[0]] = (p[1] if len(p) > 1 else "", p[2] if len(p) > 2 else "", p[3] if len(p) > 3 else "")

print("\n--- Sample of missed true matches ---\n")
for sid, cid in sample:
    _, n1, a1, c1 = s1_by_id[sid]
    n2, a2, c2 = cand_rec.get(cid, ("?", "?", "?"))
    cn1, cn2 = get_core_name(n1), get_core_name(n2)
    t1, t2 = set(cn1.split()), set(cn2.split())
    shared_tok = t1 & t2
    dig1 = extract_address_digits(normalize_address(a1, c1))
    dig2 = extract_address_digits(normalize_address(a2, c2))
    print(f"S1 {sid}: \"{n1}\" | {a1} | {c1}")
    print(f"  MISSED -> {cid}: \"{n2}\" | {a2} | {c2}")
    print(f"  shared_core_tokens={shared_tok or 'NONE'}  shared_digits={dig1&dig2 or 'NONE'} (s1_digits={dig1}, cand_digits={dig2})")
    print()
