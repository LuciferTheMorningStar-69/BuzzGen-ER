#!/usr/bin/env python3
"""Tests whether raising the blocking candidate cap (currently 15) recovers
meaningful recall by itself, before investing in new indexing logic.
Builds the index ONCE with a large cap (200) and measures the recall
ceiling and full LightGBM F0.5 at multiple truncation points (5..200)."""
import os
import sys
import time
import gc
from collections import defaultdict
import numpy as np
import lightgbm as lgb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from features import compute_pair_features
from offline_eval import prep_record, load_holdout_s1, load_ground_truth, build_country_generator, macro_f05

CAPS = [5, 10, 15, 20, 30, 50, 75, 100, 150, 200]
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "output", "champion_lgb_model_v2.txt")
THRESHOLD = 0.60

s1_sample = load_holdout_s1()
ids = [r[0] for r in s1_sample]
gt_map = load_ground_truth(set(ids))
s1_prep = [prep_record(r) for r in s1_sample]
s1_by_country = defaultdict(list)
for s_rec in s1_prep:
    s1_by_country[s_rec[3]].append(s_rec)

# sid -> list of (cid, feats) sorted by blocking score descending, up to max(CAPS)
all_ranked = {}
recall_hits_at = {c: 0 for c in CAPS}
recall_total = 0

t0 = time.time()
for country, recs in s1_by_country.items():
    print(f"Building generator for {country} with max_candidates={max(CAPS)}...")
    gen = build_country_generator(country, max_candidates=max(CAPS))
    for s_rec in recs:
        sid = s_rec[0]
        cands = gen.generate_candidates_for_s1(s_rec)  # already sorted desc by score
        ranked = []
        for sc, c_idx in cands:
            c_rec = gen.cand_records[c_idx]
            ranked.append((c_rec[0], compute_pair_features(s_rec, c_rec)))
        all_ranked[sid] = ranked

        true = gt_map.get(sid, set())
        if true:
            recall_total += len(true)
            cid_order = [cid for cid, _ in ranked]
            for cap in CAPS:
                found = set(cid_order[:cap]) & true
                recall_hits_at[cap] += len(found)
    del gen
    gc.collect()
print(f"Done in {time.time()-t0:.1f}s\n")

print("=== Recall ceiling vs candidate cap ===")
for cap in CAPS:
    print(f"  cap={cap:4d}: recall_ceiling={recall_hits_at[cap]/recall_total*100:.2f}%  "
          f"({recall_hits_at[cap]}/{recall_total})")

print("\n=== Full F0.5 vs candidate cap (using trained model + its threshold) ===")
booster = lgb.Booster(model_file=MODEL_PATH)
for cap in CAPS:
    sid_cid_feat = []
    for sid in ids:
        for cid, feats in all_ranked[sid][:cap]:
            sid_cid_feat.append((sid, cid, feats))
    X = np.array([f for _, _, f in sid_cid_feat], dtype=np.float32)
    probs = booster.predict(X, num_threads=8) if len(X) else np.array([])
    pred_map = {}
    for (sid, cid, _), p in zip(sid_cid_feat, probs):
        if p >= THRESHOLD:
            pred_map.setdefault(sid, set()).add(cid)
    f05, prec, rec = macro_f05(pred_map, gt_map, ids)
    avg_cands = sum(len(v[:cap]) for v in all_ranked.values()) / len(ids)
    print(f"  cap={cap:4d}: F0.5={f05:.5f}  P={prec*100:.2f}%  R={rec*100:.2f}%  avg_candidates/entity={avg_cands:.2f}")
