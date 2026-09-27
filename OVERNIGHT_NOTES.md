# Overnight session notes (BuzzGen entity resolution)

## What's ready now
`output/matching_results.tsv` on the `claude-work` branch (commit `b838119`) is the
best-validated result: **F0.5 = 0.90742** on an independent held-out slice of the
training data (rows the model never saw, scored with the exact official formula).
This is up from the previously-submitted 0.845 (which itself was already a big
jump from the 0.796 baseline via a threshold-bug fix).

Generated with:
- `output/champion_lgb_model_v3.txt` (trained on 100k S1 entities, full
  per-country candidate pool for realistic hard negatives)
- `threshold=0.60`, `name-floor=30`
- `blocking.py` with the corroborated-rescue pass (see "what worked" below) --
  a safe, validated improvement over the plain reverted baseline
  (recall ceiling 90.95%->90.99%, F0.5 0.90732->0.90742, no regression
  anywhere in the threshold sweep). Runtime cost: ~1.4-1.8x slower blocking
  (total run took ~92 min instead of ~75 min on the full 1.73M-entity test
  set) -- worth it for a genuine, non-regressive gain, but noted in case
  future work needs to trade it off against time budget.

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
- **Attempt 3 — corroborated rescue pass (WORKED, deployed)**: kept the
  original retrieval caps completely unchanged as the baseline (zero
  regression risk there), then added a *separate*, bounded rescue pass for
  tokens excluded for being too common (300-8000 postings). Unlike attempt 2,
  nothing gets rescued on index membership alone -- each candidate needs
  actual cheap-to-verify evidence (shared token count + digit overlap via
  set intersection, not fuzzy string scoring) and only the top 50 by that
  score get admitted per query. Validated: recall ceiling 90.95%->90.99%,
  F0.5 0.90732->0.90742, no regression across the full threshold sweep.
  This is a small win, not the breakthrough needed to approach 0.99, but it
  is a genuine, safe one. Commit `69e31b9`.

## Recommended next steps (not attempted — needs careful work, not a rushed
## overnight fix)
1. **Extend the corroborated-rescue idea further**: the rescue-pass margin
   (top 50 by shared-token/digit corroboration) was chosen conservatively
   for a first safe attempt. It's plausible a larger rescue pool, or adding
   the 4-gram signal (implemented then reverted in attempt 2) as a *cheap
   corroboration score* rather than a blind retrieval expansion, recovers
   more of the remaining ~9% recall gap without repeating attempt 2's
   mistake. Should be validated in isolation, and the runtime cost
   (attempt 3 already added ~1.4-1.8x to blocking time) needs watching --
   don't stack more rescue passes without re-checking full-scale runtime.
2. **TF-IDF-weighted triage instead of raw signal count**: if a future
   attempt still needs a triage step (e.g. because the rescue pool must
   grow beyond what stays cheap), weighting by inverse document frequency
   (rare signal = more weight) would likely fix the "multiple common
   signals outrank one rare distinguishing signal" failure mode from
   attempt 2 without reintroducing the runtime blowup. Should be validated
   carefully in isolation before deploying.
3. **Real France-specific feature**: the name-floor guard didn't work because
   it's too blunt. A proper fix needs a feature that specifically penalizes
   matches sharing only a generic/common first token (e.g. a city name) —
   this requires passing corpus-level token document-frequency into
   `features.py` (currently it only sees the two records being compared, no
   corpus context), and retraining once that feature exists.
4. **More training data**: v2→v3 (40k→100k S1, +fuzzy legal terms) only
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
