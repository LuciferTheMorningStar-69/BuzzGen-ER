#!/usr/bin/env python3
"""Tests whether TF-IDF character n-gram cosine similarity (via sparse
matrix retrieval, sklearn NearestNeighbors) can retrieve true matches that
the current token/address/digit inverted-index blocking misses. This is a
mechanically different retrieval signal (structural character overlap
instead of exact/near-exact token overlap), so it's not just another
variation of what's already been tried and validated as flat around 91%
recall ceiling.

Reports TF-IDF-alone recall ceiling per country, for direct comparison
against the known current-blocking ceiling (~90.99%).
"""
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from normalization import get_core_name
from offline_eval import load_holdout_s1, load_ground_truth, DATA_DIR, DELIM

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

TOP_K = 30
MAX_FEATURES = 100_000


def load_candidate_core_names(country):
    ids, names = [], []
    for src in ["2", "3"]:
        path = os.path.join(DATA_DIR, "train", f"train_source{src}.tsv")
        with open(path, encoding="utf-8") as f:
            f.readline()
            for line in f:
                p = line.rstrip("\n").split(DELIM)
                if p and p[0] and len(p) > 3 and p[3] == country:
                    ids.append(p[0])
                    names.append(get_core_name(p[1] if len(p) > 1 else ""))
    return ids, names


def main():
    s1_sample = load_holdout_s1()
    ids_all = [r[0] for r in s1_sample]
    gt_map = load_ground_truth(set(ids_all))

    s1_by_country = defaultdict(list)
    for r in s1_sample:
        s1_by_country[r[3]].append(r)

    overall_hits = 0
    overall_total = 0

    for country, recs in s1_by_country.items():
        print(f"\n=== {country} ===")
        t0 = time.time()
        cand_ids, cand_names = load_candidate_core_names(country)
        print(f"  Loaded {len(cand_ids):,} candidate records in {time.time()-t0:.1f}s")

        t0 = time.time()
        vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), max_features=MAX_FEATURES, min_df=1)
        cand_matrix = vec.fit_transform(cand_names)
        print(f"  Fit+transform TF-IDF ({cand_matrix.shape}) in {time.time()-t0:.1f}s")

        t0 = time.time()
        nn = NearestNeighbors(metric='cosine', algorithm='brute', n_jobs=-1)
        nn.fit(cand_matrix)
        print(f"  Fit NearestNeighbors in {time.time()-t0:.1f}s")

        query_names = [get_core_name(r[1]) for r in recs]
        query_ids = [r[0] for r in recs]
        query_matrix = vec.transform(query_names)

        t0 = time.time()
        BATCH = 500
        hits = 0
        total = 0
        for start in range(0, query_matrix.shape[0], BATCH):
            batch = query_matrix[start:start + BATCH]
            dist, idx = nn.kneighbors(batch, n_neighbors=TOP_K)
            for row_i, sid in enumerate(query_ids[start:start + BATCH]):
                true = gt_map.get(sid, set())
                if not true:
                    continue
                total += 1
                retrieved_ids = {cand_ids[j] for j in idx[row_i]}
                if true & retrieved_ids:
                    hits += 1
            if start % 5000 == 0:
                print(f"    ...{start}/{query_matrix.shape[0]} queries, elapsed {time.time()-t0:.1f}s")
        print(f"  Query phase done in {time.time()-t0:.1f}s")
        print(f"  {country} TF-IDF recall ceiling (entity-level, >=1 match found): {hits}/{total} = {100*hits/max(1,total):.2f}%")
        overall_hits += hits
        overall_total += total

    print(f"\n>>> OVERALL TF-IDF recall ceiling: {overall_hits}/{overall_total} = {100*overall_hits/max(1,overall_total):.2f}% <<<")
    print("(compare against current token/address/digit blocking ceiling: 90.99%, at entity level this metric differs slightly from the true-match-count ceiling reported elsewhere -- treat as directionally comparable)")


if __name__ == "__main__":
    main()
