#!/usr/bin/env python3
"""
Runs the offline eval harness: generates candidates for a held-out sample of
train_source1 entities, then scores several matching strategies against
train_ground_truth using the official macro F_0.5 formula.
"""
import os
import sys
import time
import numpy as np
import lightgbm as lgb
from rapidfuzz import fuzz

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from normalization import get_core_name, get_compact_name, normalize_address, extract_address_digits, LEGAL_TERMS
from features import compute_pair_features

import gc
from collections import defaultdict
from offline_eval import (
    prep_record, load_holdout_s1, load_ground_truth, build_country_generator, macro_f05
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "output", "champion_lgb_model_v2.txt")

# --- v5's modified core-name stoplist (removes business-identity-bearing terms) ---
V5_BAD_TERMS = {'ecole', 'services', 'service', 'centre', 'center', 'club', 'trust',
                'society', 'foundation', 'group', 'groupe', 'solutions', 'technologies', 'technology'}
V5_LEGAL_TERMS = LEGAL_TERMS - V5_BAD_TERMS


def get_core_name_custom(text, legal_terms):
    from normalization import clean_text
    cleaned = clean_text(text)
    tokens = [w for w in cleaned.split() if w not in legal_terms]
    return " ".join(tokens) if tokens else cleaned


def match_digits(dig1, dig2):
    if not dig1 or not dig2:
        return True
    if dig1 & dig2:
        return True
    d1_clean = {d.lstrip('0') for d in dig1 if d.lstrip('0')}
    d2_clean = {d.lstrip('0') for d in dig2 if d.lstrip('0')}
    if d1_clean & d2_clean:
        return True
    for a in d1_clean:
        for b in d2_clean:
            if len(a) >= 2 and (a in b or b in a):
                return True
            if len(a) == 1 and (b.endswith(a) or b.startswith(a)):
                return True
    return False


def champion_resolver_is_true_match(cn1, comp1, norm_a1, dig1, ft1, cn2, comp2, norm_a2, dig2, ft2):
    if dig1 and dig2 and not match_digits(dig1, dig2):
        return False, 0.0
    has_addr = bool(norm_a1 and norm_a2 and norm_a1 != 'nan' and norm_a2 != 'nan')
    as_ = fuzz.token_set_ratio(norm_a1, norm_a2) if has_addr else 0
    ns_sort = fuzz.token_sort_ratio(cn1, cn2)
    ns_set = fuzz.token_set_ratio(cn1, cn2)
    ns_comp = fuzz.ratio(comp1, comp2)
    ftm = bool(ft1 and ft2 and ft1 == ft2)
    score = 0.5 * max(ns_sort, ns_set) + 0.5 * as_ + (20.0 if ftm else 0.0)
    if has_addr and as_ >= 80:
        if ns_sort >= 35 or ns_comp >= 40 or ftm:
            return True, score + 50.0
    if ns_sort >= 75 or ns_comp >= 75:
        if has_addr and as_ < 25:
            return False, 0.0
        return True, score + 40.0
    if comp1 and comp2 and (comp1 in comp2 or comp2 in comp1) and min(len(comp1), len(comp2)) >= 5:
        if has_addr and as_ < 25:
            return False, 0.0
        return True, score + 45.0
    if has_addr and as_ >= 45 and (ns_sort >= 50 or ns_comp >= 55 or (ftm and (ns_sort >= 40 or as_ >= 60))):
        return True, score + 30.0
    return False, 0.0


def refine_precision_keep(cn1, comp1, norm_a1, dig1, ft1, cn2, comp2, norm_a2, dig2, ft2):
    if not match_digits(dig1, dig2):
        return False
    if norm_a1 and norm_a2:
        as_ = fuzz.token_set_ratio(norm_a1, norm_a2)
        if as_ < 40:
            return False
    ns_sort = fuzz.token_sort_ratio(cn1, cn2)
    ns_comp = fuzz.ratio(comp1, comp2)
    ftm = bool(ft1 and ft2 and ft1 == ft2)
    if ns_sort < 50 and ns_comp < 55 and not ftm:
        return False
    return True


def v5_keep(cn1, comp1, norm_a1, dig1, ft1, cn2, comp2, norm_a2, dig2, ft2):
    if dig1 and dig2 and not match_digits(dig1, dig2):
        return False
    has_addr = bool(norm_a1 and norm_a2)
    as_ = fuzz.token_set_ratio(norm_a1, norm_a2) if has_addr else 0
    ns_sort = fuzz.token_sort_ratio(cn1, cn2)
    ns_comp = fuzz.ratio(comp1, comp2)
    ftm = bool(ft1 and ft2 and ft1 == ft2)
    if not norm_a2:
        if ns_sort < 70 and ns_comp < 75 and not (ftm and ns_sort >= 60):
            return False
    else:
        if as_ < 40:
            return False
        if ns_sort < 50 and ns_comp < 55 and not ftm:
            return False
    return True


CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", ".eval_cache", "holdout_cands_feats.pkl")


def main():
    t_start = time.time()
    s1_sample = load_holdout_s1()
    ids = [r[0] for r in s1_sample]
    gt_map = load_ground_truth(set(ids))

    n_with_matches = sum(1 for i in ids if gt_map.get(i))
    print(f"Holdout: {len(ids):,} entities, {n_with_matches:,} have >=1 true match, "
          f"{len(ids)-n_with_matches:,} true singletons\n")

    s1_prep = [prep_record(r) for r in s1_sample]

    if os.path.exists(CACHE_PATH):
        print(f"Loading cached candidates+features from {CACHE_PATH}...")
        import pickle
        with open(CACHE_PATH, "rb") as f:
            all_cands, all_feats, recall_hits, recall_total, total_cands = pickle.load(f)
        print(f"  Loaded. Avg candidates/entity: {total_cands/len(ids):.2f}")
    else:
        print("\nGenerating candidates + features for holdout sample (one country index at a time)...")
        t0 = time.time()
        s1_by_country = defaultdict(list)
        for s_rec in s1_prep:
            s1_by_country[s_rec[3]].append(s_rec)

        all_cands = {}          # sid -> list of (cid, cand_prep)
        all_feats = {}          # sid -> list of (cid, feats)
        recall_hits = 0
        recall_total = 0
        total_cands = 0

        for country, recs in s1_by_country.items():
            gen = build_country_generator(country, max_candidates=15)
            for s_rec in recs:
                sid = s_rec[0]
                cands = gen.generate_candidates_for_s1(s_rec)
                clist = []
                flist = []
                for sc, c_idx in cands:
                    c_rec = gen.cand_records[c_idx]
                    clist.append((c_rec[0], c_rec))
                    flist.append((c_rec[0], compute_pair_features(s_rec, c_rec)))
                all_cands[sid] = clist
                all_feats[sid] = flist
                total_cands += len(clist)

                true = gt_map.get(sid, set())
                if true:
                    found = {c[0] for c in clist} & true
                    recall_hits += len(found)
                    recall_total += len(true)
            del gen
            gc.collect()

        print(f"  Done in {time.time()-t0:.1f}s. Avg candidates/entity: {total_cands/len(ids):.2f}")

        import pickle
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "wb") as f:
            pickle.dump((all_cands, all_feats, recall_hits, recall_total, total_cands), f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Cached to {CACHE_PATH}")

    print(f"  BLOCKING RECALL CEILING: {recall_hits}/{recall_total} = {recall_hits/max(1,recall_total)*100:.2f}% "
          f"of true matches are reachable via candidates\n")

    # ---- Strategy: LightGBM at various thresholds ----
    print("=" * 70)
    print("Strategy: LightGBM (champion_lgb_model.txt) threshold sweep")
    print("=" * 70)
    if os.path.exists(MODEL_PATH):
        booster = lgb.Booster(model_file=MODEL_PATH)
        sid_cid_prob = []
        for sid in ids:
            for cid, feats in all_feats[sid]:
                sid_cid_prob.append((sid, cid, feats))
        X = np.array([f for _, _, f in sid_cid_prob], dtype=np.float32)
        probs = booster.predict(X, num_threads=8) if len(X) else np.array([])

        by_sid_prob = {}
        for (sid, cid, _), p in zip(sid_cid_prob, probs):
            by_sid_prob.setdefault(sid, []).append((cid, p))

        print(f"  Prob percentiles: {np.percentile(probs, [50,90,95,99,99.5,99.9])}")
        print(f"  Fraction of pairs with prob>=0.99: {np.mean(probs>=0.99)*100:.3f}%  "
              f">=0.995: {np.mean(probs>=0.995)*100:.3f}%  >=0.999: {np.mean(probs>=0.999)*100:.3f}%\n")

        thresh_grid = list(np.arange(0.50, 0.99, 0.02)) + list(np.arange(0.99, 0.9999, 0.001))
        best = (-1, None)
        for thresh in thresh_grid:
            pred_map = {}
            for sid in ids:
                pred_map[sid] = {cid for cid, p in by_sid_prob.get(sid, []) if p >= thresh}
            f05, prec, rec = macro_f05(pred_map, gt_map, ids)
            marker = ""
            if thresh > 0.795 and thresh < 0.805:
                marker = "  <-- currently hardcoded default in champion_ml_pipeline.py"
            print(f"  thresh={thresh:.2f}  F0.5={f05:.5f}  P={prec*100:.2f}%  R={rec*100:.2f}%{marker}")
            if f05 > best[0]:
                best = (f05, thresh)
        print(f"\n  >>> BEST: thresh={best[1]:.2f}  F0.5={best[0]:.5f} <<<\n")
    else:
        print(f"  Model not found at {MODEL_PATH}, skipping.\n")

    # ---- Strategy: champion_resolver.py rules ----
    def eval_rule_strategy(name, keep_fn, cap_s2=2, cap_s3=2, use_score=False, custom_legal_terms=None):
        print("=" * 70)
        print(f"Strategy: {name}")
        print("=" * 70)
        pred_map = {}
        for s_rec in s1_prep:
            sid, n1, a1, c1, cn1, comp1, dig1, norm_a1 = s_rec
            if custom_legal_terms is not None:
                cn1 = get_core_name_custom(n1, custom_legal_terms)
                comp1 = get_compact_name(cn1)
            t1 = cn1.split()
            ft1 = t1[0] if t1 else ""
            kept_s2, kept_s3 = [], []
            for cid, c_rec in all_cands[sid]:
                _, n2, a2, c2, cn2, comp2, dig2, norm_a2 = c_rec
                if custom_legal_terms is not None:
                    cn2 = get_core_name_custom(n2, custom_legal_terms)
                    comp2 = get_compact_name(cn2)
                t2 = cn2.split()
                ft2 = t2[0] if t2 else ""
                if use_score:
                    ok, sc = keep_fn(cn1, comp1, norm_a1, dig1, ft1, cn2, comp2, norm_a2, dig2, ft2)
                else:
                    ok = keep_fn(cn1, comp1, norm_a1, dig1, ft1, cn2, comp2, norm_a2, dig2, ft2)
                    sc = fuzz.token_set_ratio(cn1, cn2)
                if ok:
                    (kept_s2 if cid.startswith("S2-") else kept_s3).append((sc, cid))
            kept_s2.sort(reverse=True)
            kept_s3.sort(reverse=True)
            best = [c for _, c in kept_s2[:cap_s2]] + [c for _, c in kept_s3[:cap_s3]]
            pred_map[sid] = set(best)
        f05, prec, rec = macro_f05(pred_map, gt_map, ids)
        avg_matches = sum(len(v) for v in pred_map.values()) / len(ids)
        n_singleton = sum(1 for v in pred_map.values() if not v)
        print(f"  F0.5={f05:.5f}  P={prec*100:.2f}%  R={rec*100:.2f}%  "
              f"avg_matches/entity={avg_matches:.2f}  singleton_rate={n_singleton/len(ids)*100:.2f}%\n")
        return f05

    eval_rule_strategy("champion_resolver.py (original, cap 2+2)", champion_resolver_is_true_match, 2, 2, use_score=True)
    eval_rule_strategy("refine_precision.py (cap 3+3)", refine_precision_keep, 3, 3)
    eval_rule_strategy("enhance_precision_v5.py (cap 2+2, modified LEGAL_TERMS)", v5_keep, 2, 2, custom_legal_terms=V5_LEGAL_TERMS)

    print(f"\nTotal eval time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
