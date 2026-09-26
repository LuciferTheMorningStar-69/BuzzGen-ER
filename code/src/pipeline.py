"""
ML Challenge 2026: Business Entity Resolution
End-to-End Master Pipeline
"""

import os
import sys
import time
import argparse
import subprocess
import numpy as np
from collections import defaultdict

from normalization import (
    clean_text, get_core_name, get_compact_name, extract_name_tokens,
    normalize_address, extract_address_digits, extract_address_keys
)
from blocking import CandidateGenerator
from features import FEATURE_NAMES, compute_pair_features
from model import EntityResolutionModel

DELIM = "\t"


def prep_record(rec: tuple[str, str, str, str]) -> tuple:
    eid, name, addr, ctry = rec
    cn = get_core_name(name)
    comp = get_compact_name(name)
    dig = extract_address_digits(addr)
    norm_a = normalize_address(addr, ctry)
    return (eid, name, addr, ctry, cn, comp, dig, norm_a)


def load_train_data(train_dir: str, max_s1: int = 50000):
    print(f"Loading training data from {train_dir} (up to {max_s1:,} S1 entities)...")
    t0 = time.time()
    
    s1_raw = []
    with open(os.path.join(train_dir, "train_source1.tsv"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip().split(DELIM)
            if p and p[0]:
                s1_raw.append((p[0], p[1] if len(p)>1 else "", p[2] if len(p)>2 else "", p[3] if len(p)>3 else ""))
                if len(s1_raw) >= max_s1:
                    break
                    
    s1_ids = {x[0] for x in s1_raw}
    gt_map = {}
    with open(os.path.join(train_dir, "train_ground_truth.tsv"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip().split(DELIM)
            if p and p[0] in s1_ids:
                gt_map[p[0]] = set(p[1].split(",")) if (len(p)>1 and p[1]) else set()
                
    needed_cands = set()
    for mids in gt_map.values():
        needed_cands.update(mids)
        
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
                        
    print(f"Loaded {len(s1_raw):,} S1 and {len(cand_raw):,} Cands in {time.time()-t0:.2f}s")
    return s1_raw, cand_raw, gt_map


def train_matching_model(train_dir: str, num_s1: int = 50000) -> EntityResolutionModel:
    s1_raw, cand_raw, gt_map = load_train_data(train_dir, max_s1=num_s1)
    
    t0 = time.time()
    s1_prep = [prep_record(r) for r in s1_raw]
    cand_prep = [prep_record(r) for r in cand_raw]
    print(f"Preprocessed training records in {time.time()-t0:.2f}s")
    
    generator = CandidateGenerator(max_candidates=15)
    generator.fit_candidates([(r[0], r[1], r[2], r[3]) for r in cand_raw])
    
    t0 = time.time()
    X = []
    y = []
    groups = []
    actual_counts = {}
    
    for s_idx, s_rec in enumerate(s1_prep):
        sid = s_rec[0]
        true_cands = gt_map.get(sid, set())
        actual_counts[s_idx] = len(true_cands)
        
        cands = generator.generate_candidates_for_s1(s_rec)
        for sc, c_idx in cands:
            c_rec = generator.cand_records[c_idx]
            cid = c_rec[0]
            label = 1 if cid in true_cands else 0
            feats = compute_pair_features(s_rec, c_rec)
            X.append(feats)
            y.append(label)
            groups.append(s_idx)
            
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    groups = np.array(groups)
    print(f"Constructed {len(y):,} training candidate pairs in {time.time()-t0:.2f}s")
    
    unique_groups = np.unique(groups)
    np.random.seed(42)
    np.random.shuffle(unique_groups)
    split_pt = int(len(unique_groups) * 0.8)
    
    train_grps = set(unique_groups[:split_pt])
    val_grps = set(unique_groups[split_pt:])
    
    tr_mask = np.isin(groups, list(train_grps))
    val_mask = np.isin(groups, list(val_grps))
    
    model = EntityResolutionModel()
    print("Fitting LightGBM model...")
    model.train(X[tr_mask], y[tr_mask], num_trees=180)
    
    print("Optimizing Macro F_0.5 decision threshold...")
    model.optimize_threshold(X[val_mask], y[val_mask], groups[val_mask], actual_counts)
    return model


def run_test_inference(test_dir: str, output_dir: str, model: EntityResolutionModel, max_s1_per_country: int = None):
    os.makedirs(output_dir, exist_ok=True)
    matching_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")
    
    print(f"Starting test inference on {test_dir}...")
    t_start = time.time()
    
    print("Reading test_source1.tsv...")
    s1_order = []
    country_s1 = defaultdict(list)
    with open(os.path.join(test_dir, "test_source1.tsv"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip().split(DELIM)
            if p and p[0]:
                sid = p[0]
                name = p[1] if len(p)>1 else ""
                addr = p[2] if len(p)>2 else ""
                ctry = p[3] if len(p)>3 else ""
                s1_order.append(sid)
                country_s1[ctry].append((sid, name, addr, ctry))
                
    print(f"Total test S1 entities: {len(s1_order):,}")
    for ctry, items in country_s1.items():
        print(f"  Country {ctry}: {len(items):,} S1 entities")
        
    results = {}
    
    for ctry, s1_list in country_s1.items():
        if max_s1_per_country:
            s1_list = s1_list[:max_s1_per_country]
            
        print(f"\n=== Processing Country: {ctry} ({len(s1_list):,} S1 entities) ===")
        t0 = time.time()
        
        cands_ctry = []
        for src in ["2", "3"]:
            path = os.path.join(test_dir, f"test_source{src}.tsv")
            print(f"  Reading {path} for country {ctry}...")
            with open(path, encoding="utf-8") as f:
                f.readline()
                for line in f:
                    p = line.rstrip().split(DELIM)
                    if p and p[0] and len(p) > 3 and p[3] == ctry:
                        cands_ctry.append((p[0], p[1] if len(p)>1 else "", p[2] if len(p)>2 else "", p[3]))
                        
        print(f"  Loaded {len(cands_ctry):,} candidate records for country {ctry} in {time.time()-t0:.2f}s")
        
        t0 = time.time()
        generator = CandidateGenerator(max_candidates=15)
        generator.fit_candidates(cands_ctry)
        print(f"  Built candidate generator index in {time.time()-t0:.2f}s")
        
        BATCH_SIZE = 10000
        total_batches = (len(s1_list) + BATCH_SIZE - 1) // BATCH_SIZE
        
        print(f"  Generating candidates and scoring in {total_batches} batches...")
        t0 = time.time()
        processed_count = 0
        
        for b_idx in range(total_batches):
            b_start = b_idx * BATCH_SIZE
            b_end = min(len(s1_list), (b_idx + 1) * BATCH_SIZE)
            batch_s1 = s1_list[b_start:b_end]
            
            batch_s1_prep = [prep_record(r) for r in batch_s1]
            
            batch_X = []
            batch_pair_map = []
            s1_cands_map = defaultdict(list)
            
            for s_idx, s_rec in enumerate(batch_s1_prep):
                cands = generator.generate_candidates_for_s1(s_rec)
                cand_ids = []
                for sc, c_idx in cands:
                    c_rec = generator.cand_records[c_idx]
                    cid = c_rec[0]
                    cand_ids.append(cid)
                    feats = compute_pair_features(s_rec, c_rec)
                    batch_X.append(feats)
                    batch_pair_map.append((s_idx, cid))
                s1_cands_map[s_idx] = cand_ids
                
            if batch_X:
                batch_preds = model.predict_pairs(np.array(batch_X, dtype=np.float32))
                s1_matches = defaultdict(list)
                for (s_idx, cid), prob in zip(batch_pair_map, batch_preds):
                    if prob >= model.threshold:
                        s1_matches[s_idx].append(cid)
            else:
                s1_matches = defaultdict(list)
                
            for s_idx, s_rec in enumerate(batch_s1_prep):
                sid = s_rec[0]
                c_ids = s1_cands_map[s_idx]
                m_ids = s1_matches[s_idx]
                results[sid] = (",".join(c_ids), ",".join(m_ids))
                
            processed_count += len(batch_s1)
            if (b_idx + 1) % 5 == 0 or (b_idx + 1) == total_batches:
                rate = processed_count / max(0.1, time.time() - t0)
                print(f"    Batch {b_idx+1}/{total_batches}: {processed_count:,}/{len(s1_list):,} S1 processed ({rate:,.0f} S1/sec)")
                
    print(f"\nWriting submission files to {output_dir}...")
    with open(matching_path, "w", encoding="utf-8") as f_match, open(candidate_path, "w", encoding="utf-8") as f_cand:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        
        empty_matches = 0
        total_matched_ids = 0
        total_candidate_ids = 0
        
        for sid in s1_order:
            c_str, m_str = results.get(sid, ("", ""))
            f_match.write(f"{sid}\t{m_str}\n")
            f_cand.write(f"{sid}\t{c_str}\n")
            
            if not m_str:
                empty_matches += 1
            else:
                total_matched_ids += len(m_str.split(","))
            if c_str:
                total_candidate_ids += len(c_str.split(","))
                
    print(f"Done in {time.time()-t_start:.2f}s!")
    print(f"Total S1 rows written: {len(s1_order):,}")
    print(f"Singletons (empty matches): {empty_matches:,} ({empty_matches/len(s1_order)*100:.2f}%)")
    print(f"Total matched IDs: {total_matched_ids:,} (Avg {total_matched_ids/len(s1_order):.2f}/S1)")
    print(f"Total candidate IDs: {total_candidate_ids:,} (Avg {total_candidate_ids/len(s1_order):.2f}/S1)")


def main():
    parser = argparse.ArgumentParser(description="ML Challenge 2026: Business Entity Resolution Pipeline")
    parser.add_argument("--train-dir", default="student_resource/dataset/train", help="Path to train directory")
    parser.add_argument("--test-dir", default="student_resource/dataset/test", help="Path to test directory")
    parser.add_argument("--output-dir", default="output", help="Path to output directory")
    parser.add_argument("--num-train-s1", type=int, default=50000, help="Number of S1 train records to use")
    parser.add_argument("--max-test-s1", type=int, default=None, help="Limit test S1 records per country (for dry-run testing)")
    args = parser.parse_args()
    
    print("=================================================================")
    print("  ML Challenge 2026: Winning Business Entity Resolution Pipeline")
    print("=================================================================")
    
    model = train_matching_model(args.train_dir, num_s1=args.num_train_s1)
    run_test_inference(args.test_dir, args.output_dir, model, max_s1_per_country=args.max_test_s1)
    
    validator_path = "student_resource/utils/validate_submission.py"
    if os.path.exists(validator_path):
        print("\nRunning submission validator...")
        subprocess.run([
            sys.executable, validator_path,
            "--matching", os.path.join(args.output_dir, "matching_results.tsv"),
            "--candidate", os.path.join(args.output_dir, "candidate_pairs.tsv"),
            "--test-dir", args.test_dir
        ])


if __name__ == "__main__":
    main()
