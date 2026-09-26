"""
ML Challenge 2026: Business Entity Resolution
Feature Engineering Module
"""

import numpy as np
from rapidfuzz import fuzz

FEATURE_NAMES = [
    'ns_raw', 'ns_core_set', 'ns_core_sort', 'ns_core_ratio', 'ns_comp_ratio',
    'first_tok_match', 'exact_name', 'exact_addr', 'len_diff', 'len_ratio',
    'has_a2', 'as_set', 'as_sort', 'as_ratio',
    'inter_dig', 'dig_jaccard', 'dig_mismatch',
    'max_sim', 'mean_sim', 'min_sim', 'is_s3'
]

def compute_pair_features(s1_tuple, cand_tuple) -> list[float]:
    """
    Compute 21 pairwise lexical, structural, and numerical features.
    s1_tuple: (sid, name, addr, ctry, cn, comp, dig, norm_a)
    cand_tuple: (cid, name, addr, ctry, cn, comp, dig, norm_a)
    """
    _, n1, a1, c1, cn1, comp1, dig1, norm_a1 = s1_tuple
    cid, n2, a2, c2, cn2, comp2, dig2, norm_a2 = cand_tuple

    # Name features
    ns_raw = float(fuzz.token_set_ratio(n1, n2))
    ns_core_set = float(fuzz.token_set_ratio(cn1, cn2))
    ns_core_sort = float(fuzz.token_sort_ratio(cn1, cn2))
    ns_core_ratio = float(fuzz.ratio(cn1, cn2))
    ns_comp_ratio = float(fuzz.ratio(comp1, comp2))

    t1 = cn1.split()
    t2 = cn2.split()
    first_tok_match = 1.0 if (t1 and t2 and t1[0] == t2[0]) else 0.0

    exact_name = 1.0 if (cn1 and cn2 and cn1 == cn2) else 0.0
    exact_addr = 1.0 if (norm_a1 and norm_a2 and norm_a1 == norm_a2) else 0.0

    len1 = len(cn1)
    len2 = len(cn2)
    len_diff = float(abs(len1 - len2))
    len_ratio = float(min(len1, len2)) / float(max(1, max(len1, len2)))

    # Address features
    has_a2 = 1.0 if norm_a2 else 0.0
    if norm_a1 and norm_a2:
        as_set = float(fuzz.token_set_ratio(norm_a1, norm_a2))
        as_sort = float(fuzz.token_sort_ratio(norm_a1, norm_a2))
        as_ratio = float(fuzz.ratio(norm_a1, norm_a2))
    else:
        as_set = as_sort = as_ratio = 0.0

    # Digits and street numbers
    inter_dig = float(len(dig1 & dig2))
    union_dig = float(len(dig1 | dig2))
    dig_jaccard = (inter_dig / union_dig) if union_dig > 0 else 0.0
    dig_mismatch = 1.0 if (len(dig1) > 0 and len(dig2) > 0 and inter_dig == 0) else 0.0

    max_sim = max(ns_core_set, as_set)
    mean_sim = (ns_core_set + as_set) / 2.0
    min_sim = min(ns_core_set, as_set) if has_a2 else ns_core_set
    is_s3 = 1.0 if cid.startswith('S3-') else 0.0

    return [
        ns_raw, ns_core_set, ns_core_sort, ns_core_ratio, ns_comp_ratio,
        first_tok_match, exact_name, exact_addr, len_diff, len_ratio,
        has_a2, as_set, as_sort, as_ratio,
        inter_dig, dig_jaccard, dig_mismatch,
        max_sim, mean_sim, min_sim, is_s3
    ]