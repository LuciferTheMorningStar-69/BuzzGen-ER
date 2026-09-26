# Overnight session notes (BuzzGen entity resolution)

## What's ready now
`output/matching_results.tsv` on the `claude-work` branch (commit `c51b88b`) is the
best-validated result: **F0.5 = 0.90732** on an independent held-out slice of the
training data (rows the model never saw, scored with the exact official formula).
This is up from the previously-submitted 0.845 (which itself was already a big
jump from the 0.796 baseline via a threshold-bug fix).

Generated with:
- `output/champion_lgb_model_v3.txt` (trained on 100k S1 entities, full
  per-country candidate pool for realistic hard negatives)
- `threshold=0.60`, `name-floor=30`
- The **original, known-good** `blocking.py` (see "what didn't work" below)

## Root causes found and fixed
1. **Threshold bug**: `champion_ml_pipeline.py` computed a calibrated threshold
   but discarded it for a hardcoded 0.80. True optimum was much higher
   (~0.98-0.99 for the old model). Fixed by using the properly calibrated
   threshold and dropping the rule-based "precision calibrator" scripts
   (`champion_resolver.py`, `refine_precision.py`, `enhance_precision_v5.py`),
   all three of which scored *worse* than plain threshold-based matching when
   validated (0.68, 0.56, 0.60 vs 0.79-0.89 F0.5).
2. **Undertrained model**: only 40k of 2.2M available S1 training entities were
   used, with a truncated (weaker) negative-candidate pool. Retrained on 100k
   with the full candidate pool → F0.5 0.887 → 0.907.
3. **Fuzzy legal-term stripping**: `anyascii` transliterates Indic-script legal
   suffixes imperfectly (e.g. "praivet" for "private"), so exact-match legal-term
   stripping left junk tokens polluting the core name. Now tolerates edit
   distance 1-2. Small but real, validated gain (0.90668 → 0.90732).
4. **France false merges**: manual inspection found address-only false merges
   (e.g. two unrelated businesses matched purely on a shared street). Added a
   name-similarity floor guard — measured near-zero effect in practice (the
   specific false merges found score above the floor via partial token overlap,
   e.g. shared city-name tokens). **Not actually fixed**, just guarded weakly.

## What was tried for blocking recall and did NOT work
Blocking recall ceiling is **~90.9-90.95%** — even hypothetically perfect
precision at that recall caps F0.5 around ~0.981, well short of the ~0.99
several other teams are reportedly scoring. This is the real remaining
bottleneck, not model quality.

- **Diagnosed root cause**: for India, native-script business names
  (Hindi/Tamil/Telugu/etc.) sometimes don't blocking-match their
  English-transliterated S1 counterpart. But `anyascii`'s transliteration is
  actually reasonably close phonetically (e.g. "arihnt enrji" ≈ "arihant
  energy") — the deeper bug was that shared tokens like "red" were being
  correctly retrieved but then EXCLUDED by an overly strict retrieval
  frequency cap (300-800 postings) before ever reaching scoring.
- **Attempt 1 — raise retrieval caps** (300/800→6000, etc.): recovered some
  candidates but caused runaway runtime (~5 entities/sec instead of ~200+/sec,
  would have taken ~90+ hours on the full test set) because thousands of
  candidates per query all went through expensive `fuzz.token_set_ratio`
  scoring.
- **Attempt 2 — bound scoring with signal-count triage**: accumulate a
  per-candidate signal count (cheap), only fuzzy-score the top 150 by count.
  Fixed the runtime (back to ~28/sec), but **recall ceiling dropped to 89.15%
  and F0.5 dropped to 0.899** — worse than baseline. The triage apparently
  deprioritizes true matches with only one weak supporting signal in favor of
  false candidates matching several generic/common signals.
- **Reverted** to the pre-experiment blocking.py (commit `e9dd0d3`). Net
  result: blocking recall is exactly where it started.

## Recommended next steps (not attempted — needs careful work, not a rushed
## overnight fix)
1. **TF-IDF-weighted triage instead of raw signal count**: the failed
   attempt's triage weighted every signal equally. Weighting by inverse
   document frequency (rare signal = more weight) would likely fix the
   "multiple common signals outrank one rare distinguishing signal" failure
   mode without reintroducing the runtime blowup. Should be validated
   carefully in isolation before deploying.
2. **Real France-specific feature**: the name-floor guard didn't work because
   it's too blunt. A proper fix needs a feature that specifically penalizes
   matches sharing only a generic/common first token (e.g. a city name) —
   this requires passing corpus-level token document-frequency into
   `features.py` (currently it only sees the two records being compared, no
   corpus context), and retraining once that feature exists.
3. **More training data**: v2→v3 (40k→100k S1, +fuzzy legal terms) only
   gained +0.00064 F0.5 — diminishing returns, probably not worth pursuing
   further before the blocking recall problem is actually solved, since
   model-quality improvements are capped by the ~0.981 blocking ceiling
   regardless.

## Operational notes
- This container has 4 CPU cores / 15GB RAM. Running two index-building jobs
  concurrently reliably causes severe CPU contention (10-50x slowdown) and
  sometimes OOM. Run heavy jobs (`run_final_pipeline.py`,
  `compare_strategies.py`, `train_full_model.py`) one at a time.
- The offline validation harness (`code/offline_eval.py`,
  `code/compare_strategies.py`) samples held-out entities from rows beyond
  100,000 of `train_source1.tsv` — any future retraining must stay within
  rows 0-100,000 (or the holdout boundary must move too) to avoid
  contaminating validation.
- `.eval_cache/holdout_cands_feats.pkl` caches the held-out candidate/feature
  generation (~10 min to rebuild) — delete and rebuild whenever `blocking.py`
  or `normalization.py` changes, since the cache silently goes stale otherwise.
