# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** BuzzGen  
**Team Members:** Nagavarapu Mourya (+3 Members)  
**Date:** September 2026  
**Competition Track:** ML Challenge | 72-Hour Hackathon | Business Entity Resolution  

---

## 1. Executive Summary
We present an end-to-end, high-precision, and highly scalable machine learning solution for large-scale Business Entity Resolution (ER) across three independent, noisy data sources ($S_1$, $S_2$, $S_3$). Our solution leverages an **open-set multilingual normalization pipeline** (incorporating Indic script transliteration via `anyascii`, French accent/legal parsing, and address standardization), an **ultra-lean multi-tier candidate generation index** that achieves a measured ~91% recall ceiling (validated on a 20,000-entity held-out split, disjoint from all training data) while generating an average of ~14-15 candidates per $S_1$ entity, and a **LightGBM gradient-boosted decision tree ranker** with a calibrated decision boundary tailored specifically to maximize the precision-heavy macro-averaged $F_{0.5}$ metric and reward singleton detection.

---

## 2. Methodology

### 2.1 Problem Analysis & Exploratory Data Analysis
During our initial data exploration across the ~12 million records in the train and test partitions, we identified several fundamental patterns and noise dynamics:
1. **Strict Country Partitioning**:
   - Analysis of all 7,638,365 ground truth matching pairs revealed that **0.0000% of true matches cross country boundaries**.
   - While the training set contains records from the US and India, the test set introduces a third country, **France** (comprising 259,452 $S_1$ entities, 703,378 $S_2$ records, and 731,615 $S_3$ records).
   - Country partitioning allows decomposing the resolution space into completely independent country blocks, reducing the Cartesian search space by $>10\times$ without losing any recall.
2. **Noise and Variation Patterns**:
   - **Indic Script Transliteration**: Indian business names frequently appear in native scripts (Devanagari, Tamil, Telugu, Bengali, Kannada) in $S_2$/$S_3$ while $S_1$ is written in Latin English. Standard string metrics on raw scripts yielded similarities below 10%. By introducing phonetically grounded ASCII transliteration via `anyascii`, similarities for matching pairs increased from $<10\%$ to $>85\%$.
   - **French Corporate Forms & Accents**: French test records feature distinct corporate identifiers (`SARL`, `SAS`, `SASU`, `SA`, `SCI`, `EURL`) and street prefixes (`Rue`, `Bd`, `Allée`, `Av`, `bis`). Accents (e.g., `é, è, ê, à, î, ô, û, ç`) required NFC/ASCII mapping to ensure stem alignment.
   - **URL/Domain Names as Business Names**: Many $S_2$/$S_3$ records substitute company names with websites or domain names (e.g., `maurewilliamscolombier.com` vs `Maure Williams Colombier Inc`). Stripping URL protocols, `www.`, and TLD suffixes restored stem identity.
   - **Street Number Invariance**: Across address variations, numerical tokens (house numbers, suite numbers, PIN/ZIP codes) remain preserved.
3. **Singleton Dynamics**:
   - In ground truth, ~5.58% of $S_1$ records have no matches in $S_2$ or $S_3$ (singletons).
   - Under the macro $F_{0.5}$ metric, predicting an empty list for a singleton scores **1.0**, whereas even a single false match drops the score to **0.0**. Therefore, high confidence gating on singletons is paramount.

### 2.2 Solution Strategy
- **Approach Type**: Multi-Tier Lean Inverted Index Blocking + 22-Dimensional Pairwise Feature Engineering + LightGBM Gradient Boosted Decision Tree + Macro $F_{0.5}$ Singleton Calibrated Thresholding.
- **Core Innovations**:
  1. *Phonetic Open-Set Canonicalization*: Unifies multilingual scripts into standardized core stems while preserving discriminative brand identifiers.
  2. *Ultra-Lean Multi-Tier Blocking*: Uses rare token inverted indices, compact string prefixes, address spatial keys, and a corroborated common-token rescue pass to maximize candidate recall ceiling (measured ~91%) while minimizing candidate set size (averaging ~14-15 per entity) to excel on the blocking evaluation criterion.
  3. *Metric-Aligned Threshold Optimization*: Grid search over out-of-fold validation predictions to identify the global decision threshold $\tau^*$ that directly maximizes the macro-averaged $F_{0.5}$ score.

---

## 3. Candidate Generation (Blocking)

Candidate generation is evaluated directly as part of final rankings, favoring pipelines that generate smaller candidate sets per $S_1$ entity while preserving high recall.

- **Blocking Keys Used**:
  1. **Core Name Inverted Index**: Index of rare discriminative name tokens (stop words and legal suffixes removed; postings capped to filter ubiquitous generic terms).
  2. **Compact Alphanumeric Index**: Normalized string without spaces (e.g., `esparzatrust`, `znbclub`) and 5-character prefix matching to capture concatenated website names and DBA abbreviations.
  3. **Spatial Address Keys**: Compound `number_street` keys (e.g., `85_wayne`, `5_pierre`, `684_nandgram`) ensuring matches between entities with name variations that share physical locations.
- **Candidate Set Statistics** (measured on a 20,000-entity held-out split, disjoint from training data, over the full US+India candidate pool of ~10.3M records):
  - Average ~14.6 candidates per $S_1$ entity.
  - True match recall ceiling in candidate set: **~91%** (63,228 of 69,487 true match instances reachable).
  - Reduction ratio: $>99.999\%$ reduction in comparison space compared to Cartesian product.
- **Ensuring True Matches Were Not Lost**:
  - Composite candidate scoring: Each candidate retrieved by any key is ranked using an additive composite score:
    $$\text{Score} = 0.5 \cdot \text{NameSim} + 0.5 \cdot \text{AddrSim} + 20 \cdot \mathbb{I}(\text{DigitMatch}) + 15 \cdot \mathbb{I}(\text{FirstTokenMatch})$$
  - Only top-15 candidates per $S_1$ entity are retained, ensuring high true-match ranking while maintaining an ultra-lean candidate footprint.

---

## 4. Matching Model

### 4.1 Feature Engineering (22 Features)
For each $(S_1, S_{\text{cand}})$ candidate pair, we extract 22 high-signal features:
- **Lexical Name Features**:
  - `ns_raw`: Token set ratio of raw names.
  - `ns_core_set`: Token set ratio of stripped core names.
  - `ns_core_sort`: Token sort ratio of stripped core names.
  - `ns_core_ratio`: Levenshtein ratio of stripped core names.
  - `ns_comp_ratio`: Character ratio of compact names (concatenated).
  - `first_tok_match`: Binary indicator of exact brand/first-token match.
  - `first_tok_generic`: Binary flag that discounts `first_tok_match` when the shared first token is just the address's city name repeated in the business name (e.g. a business named after its own city) rather than a distinguishing brand word -- catches false merges between unrelated businesses sharing a city-name prefix. Validated in isolation: F0.5 0.90401 -> 0.90531 on the held-out set, improving precision and recall together.
  - `exact_name`: Binary indicator of exact normalized core name match.
  - `len_diff`: Absolute difference in core name length.
  - `len_ratio`: Ratio of shorter to longer core name length.
- **Address & Numerical Features**:
  - `has_a2`: Indicator if candidate has a non-empty address.
  - `as_set`: Token set ratio of normalized addresses.
  - `as_sort`: Token sort ratio of normalized addresses.
  - `as_ratio`: Levenshtein ratio of normalized addresses.
  - `exact_addr`: Binary indicator of exact normalized address match.
  - `inter_dig`: Count of overlapping numerical digits (house numbers, postal codes).
  - `dig_jaccard`: Jaccard similarity of extracted numerical digit sets.
  - `dig_mismatch`: Binary flag when both records contain numbers but have zero overlap (identifying adjacent but distinct street numbers like 5952 vs 5953).
- **Composite & Contextual Features**:
  - `max_sim`: $\max(\text{ns\_core\_set}, \text{as\_set})$.
  - `mean_sim`: Arithmetic mean of name and address similarities.
  - `min_sim`: Minimum of name and address similarities.
  - `is_s3`: Source indicator (1 if candidate is from $S_3$, 0 if $S_2$).

### 4.2 Model Architecture & Training
- **Model Type**: LightGBM Gradient Boosted Decision Tree (`GBDT`).
  - `num_leaves`: 127
  - `learning_rate`: 0.06
  - `feature_fraction`: 0.85
  - `bagging_fraction`: 0.85, `bagging_freq`: 1
  - `min_child_samples`: 25
  - `num_boost_round`: 250
- **Validation Scheme**: Single held-out split (85%/15%) grouped by $S_1$ entity ID, preventing entity leakage between training and internal validation. Final offline scoring uses a separate, fixed 20,000-entity holdout reserved from the tail of the training data and never touched during training.
- **Threshold Selection**:
  - Grid search over $\tau \in [0.50, 0.995)$ with step 0.01 on the internal validation split.
  - Calibrated directly on the macro $F_{0.5}$ metric.
  - Selected threshold $\tau^* = 0.58$.

---

## 5. Results & Error Analysis

- **Macro $F_{0.5}$ Validation Score**: **0.90531** on a 20,000-entity held-out split (US + India, disjoint from all training data), at the selected threshold. Independently re-confirmed at 0.90507 on a second machine/environment (cross-validated against a separate run of this same pipeline).
- **Candidate Recall Ceiling**: **~91%** of true match instances reachable within top-15 candidates (measured, not estimated).
- **Note on France**: the training data contains no France records at all, so the France portion of the test set cannot be validated offline; actual leaderboard performance (which blends all three countries) has historically run several points below the offline US/India number, reflecting this gap. The `first_tok_generic` feature specifically targets a diagnosed France false-merge pattern, so its real-world benefit there is plausibly larger than the offline US/India number shows, but this cannot be confirmed without a submission.
- **Common False Positives (Wrong Merges)**:
  - Businesses sharing generic brand stems (e.g., "Star Developers") located on nearby streets where address digits differ by small offsets (e.g., 37 vs 41). Penalized via the `dig_mismatch` feature.
  - Entities whose first name token is also a place name repeated in the address (e.g., a business named after its city), which weakens the `first_tok_match` signal -- now corrected via the `first_tok_generic` feature (Section 4.1).
- **Common False Negatives (Missed Matches)**:
  - The dominant cause of the ~9% recall gap: pairs sharing essentially no common name tokens, digits, or address structure after normalization (e.g. heavy transliteration drift), which no text-similarity-based blocking method tested (token/address/digit indices, n-gram, phonetic/metaphone) was able to retrieve.

---

## 6. Conclusion
Our pipeline demonstrates that entity resolution at massive scale (11.7 million records) does not require slow, resource-heavy neural models. By combining linguistically informed open-set preprocessing, ultra-lean inverted index blocking, and a precision-calibrated LightGBM model, we achieve a validated macro $F_{0.5}$ of **0.90531** on held-out US/India data (recall ceiling ~91%, the binding constraint on further gains), a lean average candidate footprint (~14-15 per entity), and reproducibility from raw data in under 30 minutes on commodity 4-core hardware.

---

## Appendix

### A. Code Artifacts & Reproducibility
The runnable pipeline is at `code/business_entity_resolution/src/` (mirrored, shared-module-only, at `code/src/`):
- `normalization.py`: Normalization routines, Indic transliteration, regex-compiled state/address parsing.
- `blocking.py`: `CandidateGenerator` inverted index and candidate ranking.
- `features.py`: 22-dimensional pairwise feature extraction.
- `train_full_model.py`: trains the LightGBM model on `train_source1.tsv` and calibrates the decision threshold via macro-$F_{0.5}$ grid search.
- `run_final_pipeline.py`: runs the trained model against the real test set to produce `matching_results.tsv` and `candidate_pairs.tsv`.
- `offline_eval.py` / `compare_strategies.py`: the held-out validation harness used to produce every number reported in Section 5.
- `validate_submission.py`: validates output files against the platform's submission rules.
- `requirements.txt` (at `code/business_entity_resolution/`): Pinned dependencies (`lightgbm`, `rapidfuzz`, `anyascii`, `scikit-learn`, `numpy`, `pandas`, `scipy`).

Entry point to reproduce results (run from the repository root):
```bash
python3 code/business_entity_resolution/src/train_full_model.py --num-train-s1 100000 --model-out output/champion_lgb_model.txt
python3 code/business_entity_resolution/src/run_final_pipeline.py \
    --model-path output/champion_lgb_model.txt \
    --threshold 0.58 \
    --test-dir dataset/test \
    --output-dir output
```
