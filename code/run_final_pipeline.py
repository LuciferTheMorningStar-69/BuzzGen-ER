#!/usr/bin/env python3
"""
Final, single, reproducible pipeline for the real test set:
  1. Blocking: generate candidate_pairs.tsv via CandidateGenerator (per-country
     index, max_candidates=15 -- matches the setup validated offline).
  2. Matching: score every candidate pair with the existing trained
     LightGBM model (output/champion_lgb_model.txt) and keep pairs at or
     above --threshold (offline-validated optimum ~0.97-0.99; the
     hardcoded 0.80 default and all rule-based "precision calibrator"
     post-processing scripts were measured to perform worse and are
     intentionally NOT used here).
No retraining happens here; see train_full_model.py for that experiment.
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
from blocking import CandidateGenerator
from features import compute_pair_features
from offline_eval import prep_record

DELIM = "\t"


def load_s1(test_dir):
    s1_order = []
    s1_by_country = defaultdict(list)
    with open(os.path.join(test_dir, "test_source1.tsv"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\n").split(DELIM)
            if not p or not p[0]:
                continue
            sid = p[0]
            name = p[1] if len(p) > 1 else ""
            addr = p[2] if len(p) > 2 else ""
            ctry = p[3] if len(p) > 3 else ""
            s1_order.append(sid)
            s1_by_country[ctry].append((sid, name, addr, ctry))
    return s1_order, s1_by_country


def build_country_generator(test_dir, country, max_candidates):
    cand_raw = []
    for src in ["2", "3"]:
        path = os.path.join(test_dir, f"test_source{src}.tsv")
        with open(path, encoding="utf-8") as f:
            f.readline()
            for line in f:
                p = line.rstrip("\n").split(DELIM)
                if p and p[0] and len(p) > 3 and p[3] == country:
                    cand_raw.append((p[0], p[1] if len(p) > 1 else "", p[2] if len(p) > 2 else "", p[3]))
    gen = CandidateGenerator(max_candidates=max_candidates)
    gen.fit_candidates(cand_raw)
    return gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default=os.path.join(os.path.dirname(__file__), "..", "dataset", "test"))
    ap.add_argument("--output-dir", default=os.path.join(os.path.dirname(__file__), "..", "output"))
    ap.add_argument("--model-path", default=os.path.join(os.path.dirname(__file__), "..", "output", "champion_lgb_model.txt"))
    ap.add_argument("--threshold", type=float, default=0.98)
    ap.add_argument("--max-candidates", type=int, default=15)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cand_path = os.path.join(args.output_dir, "candidate_pairs.tsv")
    match_path = os.path.join(args.output_dir, "matching_results.tsv")

    t_start = time.time()
    print("Loading test_source1.tsv...")
    s1_order, s1_by_country = load_s1(args.test_dir)
    print(f"  {len(s1_order):,} total S1 entities")
    for c, lst in s1_by_country.items():
        print(f"    {c}: {len(lst):,}")

    print(f"\nLoading matcher model from {args.model_path} (threshold={args.threshold})...")
    booster = lgb.Booster(model_file=args.model_path)

    cand_results = {}   # sid -> comma-joined candidate ids
    match_results = {}  # sid -> comma-joined matched ids

    for country, s1_list in s1_by_country.items():
        print(f"\n=== Country: {country} ({len(s1_list):,} entities) ===")
        t0 = time.time()
        gen = build_country_generator(args.test_dir, country, args.max_candidates)
        print(f"  Built candidate index ({len(gen.cand_records):,} records) in {time.time()-t0:.1f}s")

        t0 = time.time()
        BATCH = 20000
        n_batches = (len(s1_list) + BATCH - 1) // BATCH
        for b in range(n_batches):
            batch = s1_list[b*BATCH:(b+1)*BATCH]
            batch_prep = [prep_record(r) for r in batch]

            batch_X = []
            batch_pairs = []
            s1_cand_ids = {}
            for s_rec in batch_prep:
                sid = s_rec[0]
                cands = gen.generate_candidates_for_s1(s_rec)
                cids = []
                for sc, c_idx in cands:
                    c_rec = gen.cand_records[c_idx]
                    cids.append(c_rec[0])
                    batch_X.append(compute_pair_features(s_rec, c_rec))
                    batch_pairs.append((sid, c_rec[0]))
                s1_cand_ids[sid] = cids

            if batch_X:
                probs = booster.predict(np.array(batch_X, dtype=np.float32), num_threads=8)
                accepted = defaultdict(list)
                for (sid, cid), p in zip(batch_pairs, probs):
                    if p >= args.threshold:
                        accepted[sid].append((p, cid))
            else:
                accepted = defaultdict(list)

            for s_rec in batch_prep:
                sid = s_rec[0]
                cand_results[sid] = ",".join(s1_cand_ids[sid])
                acc = accepted.get(sid, [])
                s2 = [c for _, c in sorted(acc, reverse=True) if c.startswith("S2-")]
                s3 = [c for _, c in sorted(acc, reverse=True) if c.startswith("S3-")]
                match_results[sid] = ",".join(s2 + s3)

            if (b+1) % 5 == 0 or (b+1) == n_batches:
                done = min(len(s1_list), (b+1)*BATCH)
                rate = done / max(0.1, time.time()-t0)
                print(f"    {done:,}/{len(s1_list):,} ({rate:,.0f} S1/sec)")

        del gen
        gc.collect()
        print(f"  {country} done in {time.time()-t0:.1f}s")

    print(f"\nWriting {cand_path} and {match_path}...")
    with open(cand_path, "w", encoding="utf-8") as fc, open(match_path, "w", encoding="utf-8") as fm:
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        n_singleton = 0
        n_matches = 0
        for sid in s1_order:
            fc.write(f"{sid}\t{cand_results.get(sid,'')}\n")
            m = match_results.get(sid, "")
            fm.write(f"{sid}\t{m}\n")
            if not m:
                n_singleton += 1
            else:
                n_matches += len(m.split(","))

    print(f"\nTotal S1: {len(s1_order):,}")
    print(f"Singletons: {n_singleton:,} ({n_singleton/len(s1_order)*100:.2f}%)")
    print(f"Total matched ids: {n_matches:,} (avg {n_matches/len(s1_order):.2f}/S1)")
    print(f"Total time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
