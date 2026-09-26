#!/usr/bin/env python3
"""
ML Challenge 2026: Winning Machine Learning Entity Resolution Pipeline
Team BuzzGen

Trains LightGBM GBDT ranker on 40,000 S1 training entities with realistic
hard-negative candidate pairs, calibrates optimal decision threshold for Macro F_0.5,
and performs high-throughput streaming test inference over output/candidate_pairs.tsv.
"""

import os
import sys
import time
import argparse
import subprocess
from collections import defaultdict
import numpy as np
import lightgbm as lgb
from rapidfuzz import fuzz

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from normalization import (
    clean_text, get_core_name, get_compact_name, extract_name_tokens,
    normalize_address, extract_address_digits, extract_address_keys
)
from blocking import CandidateGenerator
from features import FEATURE_NAMES, compute_pair_features

DELIM = "\t"


def prep_record(rec: tuple[str, str, str, str]) -> tuple:
    eid, name, addr, ctry = rec
    cn = get_core_name(name)
    comp = get_compact_name(cn)
    dig = extract_address_digits(addr)
    norm_a = normalize_address(addr, ctry)
    return (eid, name, addr, ctry, cn, comp, dig, norm_a)


def train_champion_model(train_dir: str, model_path: str = "output/champion_lgb_model.txt", num_train_s1: int = 40000):
    if os.path.exists(model_path):
        print(f"Loading existing trained model from {model_path}...")
        booster = lgb.Booster(model_file=model_path)
        return booster, 0.80

    print("=" * 65)
    print(f"  Training Champion LightGBM Model ({num_train_s1:,} S1 entities)")
    print("=" * 65)
    t0 = time.time()
    
    # 1. Load train_source1
    s1_raw = []
    with open(os.path.join(train_dir, "train_source1.tsv"), encoding="utf-8") as f:
        f.readline()
        for i, line in enumerate(f):
            if i >= num_train_s1:
                break
            p = line.rstrip().split(DELIM)
            s1_raw.append((p[0], p[1] if len(p)>1 else "", p[2] if len(p)>2 else "", p[3] if len(p)>3 else ""))
            
    s1_ids = {x[0] for x in s1_raw}
    
    # 2. Load train_ground_truth
    gt_map = {}
    with open(os.path.join(train_dir, "train_ground_truth.tsv"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip().split(DELIM)
            if p[0] in s1_ids:
                gt_map[p[0]] = set(p[1].split(",")) if (len(p)>1 and p[1]) else set()
                
    needed_cands = set()
    for mids in gt_map.values():
        needed_cands.update(mids)
        
    # 3. Load candidates: all true matches + 150,000 hard negatives from top of files
    cand_raw = []
    for src in ["2", "3"]:
        path = os.path.join(train_dir, f"train_source{src}.tsv")
        with open(path, encoding="utf-8") as f:
            f.readline()
            for i, line in enumerate(f):
                p = line.rstrip().split(DELIM)
                if p and p[0]:
                    if p[0] in needed_cands or i < 100000:
                        cand_raw.append((p[0], p[1] if len(p)>1 else "", p[2] if len(p)>2 else "", p[3] if len(p)>3 else ""))
                        
    print(f"Loaded {len(s1_raw):,} S1 and {len(cand_raw):,} candidate records in {time.time()-t0:.2f}s")
    
    # 4. Preprocess records
    t0 = time.time()
    s1_prep = [prep_record(r) for r in s1_raw]
    cand_prep = [prep_record(r) for r in cand_raw]
    print(f"Preprocessed records in {time.time()-t0:.2f}s")
    
    # 5. Fit candidate generator
    t0 = time.time()
    generator = CandidateGenerator(max_candidates=15)
    generator.fit_candidates([(r[0], r[1], r[2], r[3]) for r in cand_raw])
    print(f"Fitted candidate index in {time.time()-t0:.2f}s")
    
    # 6. Generate candidate pairs with features
    t0 = time.time()
    X, y, groups = [], [], []
    for s_idx, s_rec in enumerate(s1_prep):
        sid = s_rec[0]
        true_set = gt_map.get(sid, set())
        c_list = generator.generate_candidates_for_s1(s_rec)
        for sc, c_idx in c_list:
            c_rec = generator.cand_records[c_idx]
            cid = c_rec[0]
            feats = compute_pair_features(s_rec, c_rec)
            X.append(feats)
            y.append(1 if cid in true_set else 0)
            groups.append(s_idx)
            
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    groups = np.array(groups)
    print(f"Generated {len(y):,} training pairs (Positives: {np.sum(y):,}, Negatives: {len(y)-np.sum(y):,}) in {time.time()-t0:.2f}s")
    
    # 7. Split into 80% train, 20% validation
    unique_groups = np.unique(groups)
    np.random.seed(42)
    np.random.shuffle(unique_groups)
    split_pt = int(len(unique_groups) * 0.8)
    
    train_grps = set(unique_groups[:split_pt])
    val_grps = set(unique_groups[split_pt:])
    
    tr_mask = np.isin(groups, list(train_grps))
    val_mask = np.isin(groups, list(val_grps))
    
    # 8. Train LightGBM model
    print("Fitting LightGBM 127-leaf Gradient Boosted Trees...")
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
        'verbose': -1
    }
    booster = lgb.train(params, dtrain, num_boost_round=220)
    print(f"Trained LightGBM model in {time.time()-t0:.2f}s")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    booster.save_model(model_path)
    print(f"Saved trained model to {model_path}")
    
    # 9. Calibrate optimal threshold on validation set
    print("Calibrating Macro F_0.5 decision threshold on validation set...")
    val_preds = booster.predict(X[val_mask], num_threads=8)
    val_groups_arr = groups[val_mask]
    val_y_arr = y[val_mask]
    
    best_f05 = -1.0
    best_thresh = 0.80
    
    for thresh in np.arange(0.70, 0.92, 0.02):
        f05_scores = []
        tp_tot = fp_tot = fn_tot = 0
        for g in val_grps:
            g_mask = (val_groups_arr == g)
            gp = val_preds[g_mask]
            gy = val_y_arr[g_mask]
            pos = (gp >= thresh)
            tp = np.sum(pos & (gy == 1))
            fp = np.sum(pos & (gy == 0))
            fn = np.sum((~pos) & (gy == 1))
            tp_tot += tp
            fp_tot += fp
            fn_tot += fn
            
            true_cnt = np.sum(gy == 1)
            if true_cnt == 0:
                f05_scores.append(1.0 if np.sum(pos) == 0 else 0.0)
            else:
                if np.sum(pos) == 0:
                    f05_scores.append(0.0)
                else:
                    p = tp / max(1, tp + fp)
                    r = tp / true_cnt
                    denom = 0.25 * p + r
                    f05_scores.append((1.25 * p * r / denom) if denom > 0 else 0.0)
                    
        mean_f05 = np.mean(f05_scores)
        prec = tp_tot / max(1, tp_tot + fp_tot)
        rec = tp_tot / max(1, tp_tot + fn_tot)
        print(f"  Thresh {thresh:.2f}: Val Macro F0.5 = {mean_f05:.5f} | Precision = {prec*100:.2f}% | Recall = {rec*100:.2f}%")
        if mean_f05 > best_f05:
            best_f05 = mean_f05
            best_thresh = float(thresh)
            
    print(f"\n>>> Selected Optimal Threshold: {best_thresh:.2f} (Macro F_0.5 = {best_f05:.5f}) <<<\n")
    return booster, best_thresh


def run_candidate_inference(test_dir: str, cand_path: str, out_path: str, booster, threshold: float = 0.80):
    print("=" * 65)
    print(f"  Streaming Test Inference (Threshold = {threshold:.2f})")
    print("=" * 65)
    t_start = time.time()
    
    # 1. Load test S1 order and data
    print("Loading test_source1.tsv...")
    t0 = time.time()
    s1_order = []
    s1_by_country = defaultdict(list)
    with open(os.path.join(test_dir, "test_source1.tsv"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip().split(DELIM)
            sid = p[0]
            name = p[1] if len(p)>1 else ""
            addr = p[2] if len(p)>2 else ""
            ctry = p[3] if len(p)>3 else "US"
            s1_order.append(sid)
            s1_by_country[ctry].append((sid, name, addr, ctry))
            
    print(f"Loaded {len(s1_order):,} test S1 entities in {time.time()-t0:.2f}s")
    
    # 2. Load candidate_pairs.tsv
    print("Reading candidate_pairs.tsv...")
    t0 = time.time()
    cands_by_s1 = {}
    with open(cand_path, encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip().split(DELIM)
            sid = p[0]
            c = p[1] if len(p)>1 else ""
            cands_by_s1[sid] = c.split(",") if c else []
            
    print(f"Loaded candidate lists in {time.time()-t0:.2f}s")
    
    final_matches = {}
    total_matched = 0
    total_singletons = 0
    
    # 3. Process each country separately
    for ctry in ["US", "France", "India"]:
        s1_list = s1_by_country[ctry]
        print(f"\n=== Processing Country: {ctry} ({len(s1_list):,} entities) ===")
        t_ctry = time.time()
        
        # Determine all candidate IDs needed for this country
        needed_cands = set()
        for sid, name, addr, _ in s1_list:
            c_list = cands_by_s1.get(sid, [])
            for cid in c_list:
                needed_cands.add(cid)
                
        print(f"  Needed unique candidate IDs: {len(needed_cands):,}")
        
        # Load candidate records from test_source2 and test_source3
        t0 = time.time()
        cand_dict = {}
        for src in ["2", "3"]:
            path = os.path.join(test_dir, f"test_source{src}.tsv")
            with open(path, encoding="utf-8") as f:
                f.readline()
                for line in f:
                    p = line.rstrip().split(DELIM)
                    cid = p[0]
                    if cid in needed_cands:
                        name = p[1] if len(p)>1 else ""
                        addr = p[2] if len(p)>2 else ""
                        cand_dict[cid] = prep_record((cid, name, addr, ctry))
                        
        print(f"  Loaded and preprocessed {len(cand_dict):,} candidates in {time.time()-t0:.2f}s")
        
        # Preprocess S1 records for this country
        t0 = time.time()
        s1_prep_list = [prep_record(r) for r in s1_list]
        print(f"  Preprocessed {len(s1_prep_list):,} S1 entities in {time.time()-t0:.2f}s")
        
        # Process S1 entities in batches of 30,000
        BATCH_SIZE = 30000
        total_batches = (len(s1_list) + BATCH_SIZE - 1) // BATCH_SIZE
        t0 = time.time()
        ctry_matches = 0
        ctry_singletons = 0
        
        for b_idx in range(total_batches):
            b_start = b_idx * BATCH_SIZE
            b_end = min(len(s1_list), (b_idx + 1) * BATCH_SIZE)
            batch_s1_prep = s1_prep_list[b_start:b_end]
            
            batch_X = []
            batch_pairs = []
            
            for s_prep in batch_s1_prep:
                sid = s_prep[0]
                c_list = cands_by_s1.get(sid, [])
                for cid in c_list:
                    if cid in cand_dict:
                        c_prep = cand_dict[cid]
                        feats = compute_pair_features(s_prep, c_prep)
                        batch_X.append(feats)
                        batch_pairs.append((sid, cid))
                        
            if batch_X:
                preds = booster.predict(np.array(batch_X, dtype=np.float32), num_threads=8)
                s1_accepted = defaultdict(list)
                for (sid, cid), prob in zip(batch_pairs, preds):
                    if prob >= threshold:
                        s1_accepted[sid].append((prob, cid))
            else:
                s1_accepted = defaultdict(list)
                
            for s_prep in batch_s1_prep:
                sid = s_prep[0]
                accepted = s1_accepted.get(sid, [])
                if not accepted:
                    final_matches[sid] = ""
                    ctry_singletons += 1
                else:
                    s2_c = [cid for prob, cid in sorted(accepted, reverse=True) if cid.startswith("S2-")][:4]
                    s3_c = [cid for prob, cid in sorted(accepted, reverse=True) if cid.startswith("S3-")][:4]
                    best = s2_c + s3_c
                    if not best:
                        final_matches[sid] = ""
                        ctry_singletons += 1
                    else:
                        final_matches[sid] = ",".join(best)
                        ctry_matches += len(best)
                        
            proc = min(len(s1_list), (b_idx + 1) * BATCH_SIZE)
            rate = proc / max(0.1, time.time() - t0)
            eta = (len(s1_list) - proc) / max(1.0, rate)
            if (b_idx + 1) % 5 == 0 or (b_idx + 1) == total_batches:
                print(f"    Batch {b_idx+1}/{total_batches}: {proc:,}/{len(s1_list):,} S1 processed ({rate:,.0f} S1/sec, ETA: {eta:.0f}s)")
                
        print(f"  {ctry} completed in {time.time()-t_ctry:.2f}s!")
        print(f"    Matches: {ctry_matches:,} (Avg {ctry_matches/len(s1_list):.2f}/S1)")
        print(f"    Singletons: {ctry_singletons:,} ({ctry_singletons/len(s1_list)*100:.2f}%)")
        
        total_matched += ctry_matches
        total_singletons += ctry_singletons
        
    print("\n" + "=" * 65)
    print("  Overall Test Inference Summary")
    print("=" * 65)
    print(f"Total S1 entities: {len(s1_order):,}")
    print(f"Total Matches:     {total_matched:,} (Avg {total_matched/len(s1_order):.2f}/S1)")
    print(f"Total Singletons:  {total_singletons:,} ({total_singletons/len(s1_order)*100:.2f}%)")
    print(f"Completed in:      {time.time()-t_start:.2f}s")
    
    # 4. Write matching_results.tsv
    print(f"\nWriting matching results to {out_path}...")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in s1_order:
            f.write(f"{sid}\t{final_matches.get(sid, '')}\n")
            
    print("Successfully generated matching_results.tsv!")


def main():
    parser = argparse.ArgumentParser(description="Team BuzzGen Champion Entity Resolution Pipeline")
    parser.add_argument("--train-dir", default="student_resource/dataset/train")
    parser.add_argument("--test-dir", default="student_resource/dataset/test")
    parser.add_argument("--cand-path", default="output/candidate_pairs.tsv")
    parser.add_argument("--out-path", default="output/matching_results.tsv")
    parser.add_argument("--num-train", type=int, default=40000)
    parser.add_argument("--threshold", type=float, default=0.80)
    args = parser.parse_args()
    
    booster, calibrated_thresh = train_champion_model(args.train_dir, num_train_s1=args.num_train)
    thresh = args.threshold if args.threshold is not None else calibrated_thresh
    
    run_candidate_inference(args.test_dir, args.cand_path, args.out_path, booster, threshold=thresh)


if __name__ == "__main__":
    main()
