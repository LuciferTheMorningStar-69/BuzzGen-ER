"""
ML Challenge 2026: Business Entity Resolution
High-Precision, Scalable Candidate Generation (Blocking) Module
"""

import time
from collections import defaultdict
from rapidfuzz import fuzz
from normalization import (
    clean_text, get_core_name, get_compact_name, extract_name_tokens,
    normalize_address, extract_address_digits, extract_address_keys
)

class CandidateGenerator:
    """Multi-tier inverted index for ultra-lean candidate generation."""

    def __init__(self, max_candidates: int = 15):
        self.max_candidates = max_candidates
        self.tok_idx = defaultdict(list)
        self.comp_idx = defaultdict(list)
        self.comp_prefix_idx = defaultdict(list)
        self.addr_idx = defaultdict(list)
        self.cand_records = []

    def fit_candidates(self, candidates: list[tuple[str, str, str, str]]):
        """
        Build inverted indices on candidate pool (S2 + S3).
        Each candidate is (entity_id, business_name, business_address, country).
        """
        self.cand_records = []
        for idx, (cid, name, addr, ctry) in enumerate(candidates):
            cn = get_core_name(name)
            comp = get_compact_name(cn)
            norm_a = normalize_address(addr, ctry)
            dig = extract_address_digits(norm_a)
            self.cand_records.append((cid, name, addr, ctry, cn, comp, dig, norm_a))

            for t in extract_name_tokens(cn):
                self.tok_idx[t].append(idx)
            if len(comp) >= 4:
                self.comp_idx[comp].append(idx)
                if len(comp) >= 6:
                    self.comp_prefix_idx[comp[:5]].append(idx)
            for ak in extract_address_keys(norm_a):
                self.addr_idx[ak].append(idx)

    def generate_candidates_for_s1(self, s1_prep: tuple) -> list[tuple[float, int]]:
        """
        Retrieve and rank top candidates for a preprocessed S1 record.
        s1_prep: (sid, name, addr, ctry, cn, comp, dig, norm_a)
        Returns list of (score, candidate_index) sorted descending by score.
        """
        sid, name, addr, ctry, cn, comp, dig, norm_a = s1_prep

        cand_indices = set()
        tokens = extract_name_tokens(cn)
        tok_postings = [(len(self.tok_idx.get(t, [])), t) for t in tokens if t in self.tok_idx]
        tok_postings.sort()

        for i, (count, t) in enumerate(tok_postings):
            if i < 2 or count <= 300:
                if count <= 800:
                    cand_indices.update(self.tok_idx[t])

        if len(comp) >= 4:
            cand_indices.update(self.comp_idx.get(comp, []))
            if len(comp) >= 6:
                p = self.comp_prefix_idx.get(comp[:5], [])
                if len(p) <= 100:
                    cand_indices.update(p)

        for ak in extract_address_keys(norm_a):
            p = self.addr_idx.get(ak, [])
            if len(p) <= 250:
                cand_indices.update(p)

        t1 = cn.split()
        scored = []
        for c_idx in cand_indices:
            c_rec = self.cand_records[c_idx]
            ns = fuzz.token_set_ratio(cn, c_rec[4])
            has_a = bool(norm_a and c_rec[7])
            as_ = fuzz.token_set_ratio(norm_a, c_rec[7]) if has_a else 0.0

            t2 = c_rec[4].split()
            ftm = (t1 and t2 and t1[0] == t2[0])
            inter_dig = len(dig & c_rec[6])

            if has_a:
                sc = 0.5 * ns + 0.5 * as_ + (20.0 if inter_dig else 0.0) + (15.0 if ftm else 0.0)
            else:
                sc = ns + (15.0 if ftm else 0.0)

            scored.append((sc, c_idx))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:self.max_candidates]
