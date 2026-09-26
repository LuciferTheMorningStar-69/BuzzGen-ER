#!/usr/bin/env python3
"""
Retrains the LightGBM matcher on a much larger S1 sample (up to the first
100,000 rows of train_source1.tsv -- guaranteed disjoint from the offline
eval holdout, which is sampled only from rows beyond 100,000) using the
FULL per-country candidate pool (not truncated to the first 100k rows of
source2/3, which starved the model of realistic hard negatives). Threshold
is calibrated via extended macro F_0.5 grid search and actually used
(the existing champion_ml_pipeline.py computed a calibrated threshold but
then discarded it in favor of a hardcoded 0.80 default -- fixed here).
"""
import os
import sys
import time
import gc
import argparse
from collections import defaultdict
import numpy as np
import lightgbm as lgb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from features import FEATURE_NAMES, compute_pair_features
from offline_eval import prep_record, build_country_generator, macro_f05

DELIM = "\t"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")


def load_train_s1(max_rows):
    path = os.path.join(DATA_DIR, "train", "train_source1.tsv")
    rows = []
    with open(path, encoding="utf-8") as f:
        f.readline()
        for i, line in enumerate(f):
            if i >= max_rows:
                break
            p = line.rstrip("\n").split(DELIM)
            if p and p[0]:
                rows.append((p[0], p[1] if len(p) > 1 else "", p[2] if len(p) > 2 else "", p[3] if len(p) > 3 else ""))
    return rows


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-train-s1", type=int, default=100_000)
    ap.add_argument("--model-out", default=os.path.join(os.path.dirname(__file__), "..", "output", "champion_lgb_model_v2.txt"))
    args = ap.parse_args()

    t_start = time.time()
    s1_raw = load_train_s1(args.num_train_s1)
    s1_ids = {r[0] for r in s1_raw}
    print(f"Loaded {len(s1_raw):,} S1 training entities (rows 0-{args.num_train_s1:,})")

    gt_map = load_ground_truth(s1_ids)
    n_with = sum(1 for v in gt_map.values() if v)
    print(f"  {n_with:,} have >=1 true match, {len(s1_ids)-n_with:,} true singletons")

    s1_prep = [prep_record(r) for r in s1_raw]
    s1_by_country = defaultdict(list)
    for s_rec in s1_prep:
        s1_by_country[s_rec[3]].append(s_rec)

    X_list, y_list, group_list = [], [], []
    actual_counts = {}
    group_idx = 0

    for country, recs in s1_by_country.items():
        print(f"\nCountry={country}: building full candidate pool...")
        gen = build_country_generator(country, max_candidates=15)
        t0 = time.time()
        for s_rec in recs:
            sid = s_rec[0]
            true_set = gt_map.get(sid, set())
            actual_counts[group_idx] = len(true_set)
            cands = gen.generate_candidates_for_s1(s_rec)
            for sc, c_idx in cands:
                c_rec = gen.cand_records[c_idx]
                cid = c_rec[0]
                feats = compute_pair_features(s_rec, c_rec)
                X_list.append(feats)
                y_list.append(1 if cid in true_set else 0)
                group_list.append(group_idx)
            group_idx += 1
        print(f"  Generated pairs for {len(recs):,} entities in {time.time()-t0:.1f}s")
        del gen
        gc.collect()

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    groups = np.array(group_list)
    print(f"\nTotal pairs: {len(y):,} (positives: {int(y.sum()):,}, negatives: {len(y)-int(y.sum()):,})")

    unique_groups = np.unique(groups)
    rng = np.random.RandomState(42)
    rng.shuffle(unique_groups)
    split_pt = int(len(unique_groups) * 0.85)
    train_grps = set(unique_groups[:split_pt].tolist())
    val_grps = set(unique_groups[split_pt:].tolist())
    tr_mask = np.isin(groups, list(train_grps))
    val_mask = np.isin(groups, list(val_grps))

    print(f"Train pairs: {tr_mask.sum():,}  Val pairs: {val_mask.sum():,}")

    print("\nTraining LightGBM...")
    t0 = time.time()
    dtrain = lgb.Dataset(X[tr_mask], label=y[tr_mask], feature_name=FEATURE_NAMES)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 127,
        'learning_rate': 0.06,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 1,
        'min_child_samples': 25,
        'n_jobs': 8,
        'verbose': -1,
    }
    booster = lgb.train(params, dtrain, num_boost_round=250)
    print(f"  Trained in {time.time()-t0:.1f}s")

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    booster.save_model(args.model_out)
    print(f"  Saved model to {args.model_out}")

    print("\nCalibrating threshold on internal validation split...")
    val_preds = booster.predict(X[val_mask], num_threads=8)
    val_groups_arr = groups[val_mask]
    val_y_arr = y[val_mask]
    val_grp_list = sorted(val_grps)

    best = (-1.0, None)
    for thresh in np.arange(0.50, 0.995, 0.01):
        scores = []
        tp_tot = fp_tot = fn_tot = 0
        for g in val_grp_list:
            g_mask = (val_groups_arr == g)
            gp = val_preds[g_mask]
            gy = val_y_arr[g_mask]
            pos = gp >= thresh
            tp = int(np.sum(pos & (gy == 1)))
            fp = int(np.sum(pos & (gy == 0)))
            fn = int(np.sum((~pos) & (gy == 1)))
            tp_tot += tp; fp_tot += fp; fn_tot += fn
            true_cnt = actual_counts.get(g, 0)
            if true_cnt == 0:
                scores.append(1.0 if np.sum(pos) == 0 else 0.0)
            else:
                if np.sum(pos) == 0:
                    scores.append(0.0)
                else:
                    p = tp / max(1, tp + fp)
                    r = tp / true_cnt
                    denom = 0.25 * p + r
                    scores.append((1.25 * p * r / denom) if denom > 0 else 0.0)
        mean_f05 = float(np.mean(scores))
        prec = tp_tot / max(1, tp_tot + fp_tot)
        rec = tp_tot / max(1, tp_tot + fn_tot)
        print(f"  thresh={thresh:.2f}  F0.5={mean_f05:.5f}  P={prec*100:.2f}%  R={rec*100:.2f}%")
        if mean_f05 > best[0]:
            best = (mean_f05, float(thresh))

    print(f"\n>>> Selected threshold={best[1]:.2f}  Internal Val F0.5={best[0]:.5f} <<<")
    print(f"Total time: {time.time()-t_start:.1f}s")

    with open(args.model_out + ".threshold", "w") as f:
        f.write(str(best[1]))


if __name__ == "__main__":
    main()
