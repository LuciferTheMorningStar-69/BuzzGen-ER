#!/usr/bin/env python3
"""Validates matching_results.tsv and candidate_pairs.tsv against the
platform's stated submission rules (no external deps)."""
import sys
import os

DELIM = "\t"


def load_test_s1_ids(test_dir):
    ids = set()
    with open(os.path.join(test_dir, "test_source1.tsv"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\n").split(DELIM)
            if p and p[0]:
                ids.add(p[0])
    return ids


def load_valid_cand_ids(test_dir):
    ids = set()
    for src in ["2", "3"]:
        with open(os.path.join(test_dir, f"test_source{src}.tsv"), encoding="utf-8") as f:
            f.readline()
            for line in f:
                p = line.rstrip("\n").split(DELIM)
                if p and p[0]:
                    ids.add(p[0])
    return ids


def load_tsv_map(path):
    m = {}
    dupes = 0
    with open(path, encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            p = line.rstrip("\n").split(DELIM)
            sid = p[0]
            val = p[1] if len(p) > 1 else ""
            if sid in m:
                dupes += 1
            m[sid] = val
    return m, dupes


def main():
    test_dir = sys.argv[1] if len(sys.argv) > 1 else "dataset/test"
    match_path = sys.argv[2] if len(sys.argv) > 2 else "output/matching_results.tsv"
    cand_path = sys.argv[3] if len(sys.argv) > 3 else "output/candidate_pairs.tsv"

    issues = []

    print("Loading test_source1 ids...")
    s1_ids = load_test_s1_ids(test_dir)
    print(f"  {len(s1_ids):,} S1 entities")

    print("Loading valid S2/S3 candidate ids...")
    valid_cand_ids = load_valid_cand_ids(test_dir)
    print(f"  {len(valid_cand_ids):,} valid S2/S3 ids")

    print(f"Loading {match_path}...")
    match_map, match_dupes = load_tsv_map(match_path)
    print(f"Loading {cand_path}...")
    cand_map, cand_dupes = load_tsv_map(cand_path)

    if match_dupes:
        issues.append(f"matching_results.tsv has {match_dupes} duplicate source1_entity_id rows")
    if cand_dupes:
        issues.append(f"candidate_pairs.tsv has {cand_dupes} duplicate source1_entity_id rows")

    missing_in_match = s1_ids - set(match_map.keys())
    if missing_in_match:
        issues.append(f"{len(missing_in_match)} S1 entities missing from matching_results.tsv (e.g. {list(missing_in_match)[:3]})")

    extra_in_match = set(match_map.keys()) - s1_ids
    if extra_in_match:
        issues.append(f"{len(extra_in_match)} rows in matching_results.tsv reference S1 ids not in test set")

    n_bad_ref = 0
    n_dup_within_row = 0
    n_not_subset_of_cand = 0
    n_self_match = 0
    example_bad = []
    total_matched = 0
    total_singleton = 0

    for sid, m in match_map.items():
        if not m:
            total_singleton += 1
            continue
        parts = m.split(",")
        total_matched += len(parts)
        if len(parts) != len(set(parts)):
            n_dup_within_row += 1
        bad = [c for c in parts if c not in valid_cand_ids]
        if bad:
            n_bad_ref += 1
            if len(example_bad) < 3:
                example_bad.append((sid, bad[:3]))
        if any(c == sid for c in parts):
            n_self_match += 1
        cand_list = set(cand_map.get(sid, "").split(",")) if cand_map.get(sid, "") else set()
        if not set(parts).issubset(cand_list):
            n_not_subset_of_cand += 1

    if n_bad_ref:
        issues.append(f"{n_bad_ref} rows reference candidate ids that don't exist in test S2/S3 (e.g. {example_bad})")
    if n_dup_within_row:
        issues.append(f"{n_dup_within_row} rows have duplicate ids within their own matched_entity_ids list")
    if n_self_match:
        issues.append(f"{n_self_match} rows reference an S1 id as a match (self-match)")
    if n_not_subset_of_cand:
        issues.append(f"{n_not_subset_of_cand} rows have matched ids that are NOT a subset of that row's candidate_pairs.tsv list")

    print(f"\nTotal S1: {len(s1_ids):,}")
    print(f"Singletons: {total_singleton:,} ({total_singleton/len(s1_ids)*100:.2f}%)")
    print(f"Total matched ids: {total_matched:,} (avg {total_matched/len(s1_ids):.2f}/S1)")

    print("\n" + "=" * 60)
    if issues:
        print(f"FAIL — {len(issues)} issue(s) found:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        sys.exit(1)
    else:
        print("PASS — no issues found. Safe to submit.")
        sys.exit(0)


if __name__ == "__main__":
    main()
