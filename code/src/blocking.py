"""
ML Challenge 2026: Business Entity Resolution
High-Precision, Scalable Candidate Generation (Blocking) Module
"""

import time
from collections import defaultdict
from rapidfuzz import fuzz
import jellyfish
from normalization import (
    clean_text, get_core_name, get_compact_name, extract_name_tokens,
    normalize_address, extract_address_digits, extract_address_keys
)

def _metaphone_codes(tokens: list[str]) -> set[str]:
    """Phonetic codes for a name's tokens. Bridges transliteration drift that
    exact/fuzzy string matching misses (e.g. anyascii's 'epeks imphotek' for
    'apex infotech' has raw Jaccard ~0.02 but metaphone+Jaro-Winkler ~0.8)."""
    codes = set()
    for t in tokens:
        try:
            c = jellyfish.metaphone(t)
        except Exception:
            c = ""
        if c:
            codes.add(c)
    return codes


class CandidateGenerator:
    """Multi-tier inverted index for ultra-lean candidate generation."""

    def __init__(self, max_candidates: int = 15):
        self.max_candidates = max_candidates
        self.tok_idx = defaultdict(list)
        self.comp_idx = defaultdict(list)
        self.comp_prefix_idx = defaultdict(list)
        self.addr_idx = defaultdict(list)
        self.digit_idx = defaultdict(list)
        self.phon_idx = defaultdict(list)
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
            for code in _metaphone_codes(extract_name_tokens(cn)):
                self.phon_idx[code].append(idx)
            if len(comp) >= 4:
                self.comp_idx[comp].append(idx)
                if len(comp) >= 6:
                    self.comp_prefix_idx[comp[:5]].append(idx)
            for ak in extract_address_keys(norm_a):
                self.addr_idx[ak].append(idx)
            for d in dig:
                if len(d) >= 3:
                    self.digit_idx[d].append(idx)

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

        rescue_tokens = []
        for i, (count, t) in enumerate(tok_postings):
            if i < 2 or count <= 300:
                if count <= 800:
                    cand_indices.update(self.tok_idx[t])
            elif count <= 8000:
                # Too common to trust blindly (e.g. "red"), but worth a
                # corroborated rescue pass below rather than dropping outright.
                rescue_tokens.append(t)

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

        # Standalone digit-only key: catches same-location matches when name
        # tokens don't overlap at all (e.g. native-script vs transliterated
        # business names), which word-paired address keys miss entirely.
        for d in dig:
            if len(d) >= 3:
                p = self.digit_idx.get(d, [])
                if len(p) <= 300:
                    cand_indices.update(p)

        # Phonetic (metaphone) key: catches transliteration-drifted names
        # that share no exact token/n-gram at all (e.g. "epeks imphotek" for
        # "apex infotech") but sound alike. Rare codes are trusted directly;
        # common codes go through the same corroboration check as the
        # rescue pass below, to avoid the "several weak signals outrank one
        # strong rare one" failure mode from an earlier, less careful attempt.
        phon_rescue_codes = []
        for code in _metaphone_codes(tokens):
            p = self.phon_idx.get(code, [])
            if not p:
                continue
            if len(p) <= 150:
                cand_indices.update(p)
            elif len(p) <= 3000:
                phon_rescue_codes.append(code)

        # Rescue pass for common tokens (e.g. "red") that were correctly
        # matched but excluded above to avoid blindly trusting a huge
        # postings list. Cheaply pre-score candidates by ACTUAL shared-token/
        # digit overlap (not just "matched via some index") and only rescue
        # the most corroborated few -- this is what a prior naive
        # signal-count-only triage got wrong: it let several weak common
        # signals outrank one strong rare one. Here nothing is admitted
        # without direct, cheap-to-verify textual evidence.
        if rescue_tokens or phon_rescue_codes:
            t1_set = set(cn.split())
            s1_codes = _metaphone_codes(tokens) if phon_rescue_codes else set()
            pre_scored = []
            seen = set()
            for t in rescue_tokens:
                for c_idx in self.tok_idx[t]:
                    if c_idx in cand_indices or c_idx in seen:
                        continue
                    seen.add(c_idx)
                    c_rec = self.cand_records[c_idx]
                    shared_tok = len(t1_set & set(c_rec[4].split()))
                    shared_dig = len(dig & c_rec[6])
                    pre_score = 2 * shared_tok + shared_dig
                    if pre_score > 0:
                        pre_scored.append((pre_score, c_idx))
            for code in phon_rescue_codes:
                for c_idx in self.phon_idx[code]:
                    if c_idx in cand_indices or c_idx in seen:
                        continue
                    seen.add(c_idx)
                    c_rec = self.cand_records[c_idx]
                    # Corroboration for phonetic-only matches: how many OTHER
                    # metaphone codes also overlap (multiple phonetically
                    # matching words = much stronger evidence than one), plus
                    # digit overlap. Literal token overlap doesn't apply here
                    # by construction (that's why it needed the phonetic path).
                    shared_phon = len(s1_codes & _metaphone_codes(c_rec[4].split()))
                    shared_dig = len(dig & c_rec[6])
                    pre_score = 2 * shared_phon + shared_dig
                    if pre_score > 0:
                        pre_scored.append((pre_score, c_idx))
            pre_scored.sort(reverse=True)
            for _, c_idx in pre_scored[:50]:
                cand_indices.add(c_idx)

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
