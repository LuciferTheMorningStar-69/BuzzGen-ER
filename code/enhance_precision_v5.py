#!/usr/bin/env python3
"""
Team BuzzGen: Enhanced Precision Calibrator v5
Eliminates generic-word over-merging, address digit discrepancies,
and restores true singletons to maximize Macro F_0.5.
"""

import os
import sys
import time
from collections import defaultdict
from rapidfuzz import fuzz

sys.path.insert(0, 'code/src')
import normalization
# Prevent stripping of critical business identity words
bad_terms = {'ecole', 'services', 'service', 'centre', 'center', 'club', 'trust', 'society', 'foundation', 'group', 'groupe', 'solutions', 'technologies', 'technology'}
normalization.LEGAL_TERMS = normalization.LEGAL_TERMS - bad_terms

from normalization import get_core_name, get_compact_name, normalize_address, extract_address_digits

def match_digits(dig1, dig2):
    """Check if address numerical sets are compatible (not contradictory)."""
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

def main():
    print("=" * 65)
    print("  Team BuzzGen: Enhanced Precision Calibrator v5")
    print("=" * 65)
    
    test_dir = "student_resource/dataset/test"
    v1_backup_path = "output/matching_results_v1_backup.tsv"
    out_matching_path = "output/matching_results.tsv"
    
    # 1. Read test_source1 to get exact ordering and country mapping
    print("Reading test_source1.tsv...")
    t0 = time.time()
    s1_order = []
    s1_by_country = defaultdict(list)
    with open(os.path.join(test_dir, "test_source1.tsv"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip().split("\t")
            sid = parts[0]
            name = parts[1] if len(parts) > 1 else ""
            addr = parts[2] if len(parts) > 2 else ""
            ctry = parts[3] if len(parts) > 3 else "US"
            s1_order.append(sid)
            s1_by_country[ctry].append((sid, name, addr, ctry))
            
    print(f"Loaded {len(s1_order):,} test S1 entities in {time.time()-t0:.2f}s")
    
    # 2. Read v1 matches
    print(f"\nReading candidate matches from {v1_backup_path}...")
    t0 = time.time()
    v1_matches = {}
    with open(v1_backup_path, encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip().split("\t")
            sid = parts[0]
            m = parts[1] if len(parts) > 1 else ""
            v1_matches[sid] = m
    print(f"Loaded initial matches in {time.time()-t0:.2f}s")
    
    # 3. Process each country
    final_matches = {}
    total_kept_matches = 0
    total_singletons = 0
    
    for ctry in ["US", "France", "India"]:
        s1_list = s1_by_country[ctry]
        print(f"\n=== Processing Country: {ctry} ({len(s1_list):,} entities) ===")
        t_ctry = time.time()
        
        # Collect needed candidates for this country
        needed_cands = set()
        for sid, name, addr, _ in s1_list:
            m = v1_matches.get(sid, "")
            if m:
                for mid in m.split(","):
                    needed_cands.add(mid)
                    
        print(f"  Needed unique candidate records: {len(needed_cands):,}")
        
        # Load candidate records from test_source2 and test_source3
        t0 = time.time()
        cand_dict = {}
        for src in ["2", "3"]:
            p = os.path.join(test_dir, f"test_source{src}.tsv")
            with open(p, encoding="utf-8") as f:
                f.readline()
                for line in f:
                    parts = line.rstrip().split("\t")
                    cid = parts[0]
                    if cid in needed_cands:
                        name = parts[1] if len(parts) > 1 else ""
                        addr = parts[2] if len(parts) > 2 else ""
                        cn = get_core_name(name)
                        comp = get_compact_name(cn)
                        norm_a = normalize_address(addr, ctry)
                        dig = extract_address_digits(norm_a)
                        t = cn.split()
                        ft = t[0] if t else ""
                        cand_dict[cid] = (cn, comp, norm_a, dig, ft)
                        
        print(f"  Loaded and preprocessed candidates in {time.time()-t0:.2f}s")
        
        # Run high-precision filtering
        t0 = time.time()
        ctry_kept = 0
        ctry_singletons = 0
        
        for sid, name, addr, _ in s1_list:
            m = v1_matches.get(sid, "")
            if not m:
                final_matches[sid] = ""
                ctry_singletons += 1
                continue
                
            m_list = m.split(",")
            cn1 = get_core_name(name)
            comp1 = get_compact_name(cn1)
            norm_a1 = normalize_address(addr, ctry)
            dig1 = extract_address_digits(norm_a1)
            t1 = cn1.split()
            ft1 = t1[0] if t1 else ""
            
            kept_s2 = []
            kept_s3 = []
            
            for mid in m_list:
                if mid not in cand_dict:
                    continue
                cn2, comp2, norm_a2, dig2, ft2 = cand_dict[mid]
                
                # 1. Reject digit mismatch
                if dig1 and dig2 and not match_digits(dig1, dig2):
                    continue
                    
                has_addr = bool(norm_a1 and norm_a2)
                as_ = fuzz.token_set_ratio(norm_a1, norm_a2) if has_addr else 0
                ns_sort = fuzz.token_sort_ratio(cn1, cn2)
                ns_comp = fuzz.ratio(comp1, comp2)
                ftm = bool(ft1 and ft2 and ft1 == ft2)
                
                if not norm_a2:
                    # No address: require strong name match
                    if ns_sort < 70 and ns_comp < 75 and not (ftm and ns_sort >= 60):
                        continue
                else:
                    # Address present: require address consistency
                    if as_ < 40:
                        continue
                    if ns_sort < 50 and ns_comp < 55 and not ftm:
                        continue
                        
                sc = max(ns_sort, ns_comp) + 0.5 * as_ + (20 if ftm else 0)
                if mid.startswith("S2-"):
                    kept_s2.append((sc, mid))
                else:
                    kept_s3.append((sc, mid))
                    
            kept_s2.sort(reverse=True)
            kept_s3.sort(reverse=True)
            
            # Select top-2 from S2 and top-2 from S3
            best = [x[1] for x in kept_s2[:2]] + [x[1] for x in kept_s3[:2]]
            
            if not best:
                final_matches[sid] = ""
                ctry_singletons += 1
            else:
                final_matches[sid] = ",".join(best)
                ctry_kept += len(best)
                
        print(f"  {ctry} completed in {time.time()-t_ctry:.2f}s!")
        print(f"    Kept matches: {ctry_kept:,} (Avg {ctry_kept/len(s1_list):.2f}/S1)")
        print(f"    Singletons:   {ctry_singletons:,} ({ctry_singletons/len(s1_list)*100:.2f}%)")
        
        total_kept_matches += ctry_kept
        total_singletons += ctry_singletons
        
    print("\n" + "=" * 65)
    print("  Overall Enhanced Precision Summary")
    print("=" * 65)
    print(f"Total S1 entities: {len(s1_order):,}")
    print(f"Total Matches:     {total_kept_matches:,} (Avg {total_kept_matches/len(s1_order):.2f}/S1)")
    print(f"Total Singletons:  {total_singletons:,} ({total_singletons/len(s1_order)*100:.2f}%)")
    
    print(f"\nWriting refined submission to {out_matching_path}...")
    with open(out_matching_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in s1_order:
            f.write(f"{sid}\t{final_matches.get(sid, '')}\n")
            
    print("Done writing matching_results.tsv!")

if __name__ == "__main__":
    main()
