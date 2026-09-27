# ML Challenge 2026: Business Entity Resolution Pipeline

High-performance, scalable machine learning pipeline for Business Entity Resolution across three heterogeneous sources (`Source 1`, `Source 2`, and `Source 3`) covering multiple countries (`US`, `India`, and `France`).

## 1. Overview of Approach
- **Open-Set Multilingual Normalization**: Zero-dependency unicode & Indic script transliteration via `anyascii`, legal suffix canonicalization (US, Indian, and French legal structures), domain name extraction with guarded word-segmentation of concatenated slugs, and address street/digit normalization.
- **Ultra-Lean Candidate Generation (Blocking)**: High-speed inverted index on discriminative core tokens, compact alphanumeric stems, address spatial keys (`number_street`), and a corroborated common-token rescue pass. Measured recall ceiling ~91% (on a 20,000-entity held-out split, disjoint from training) while maintaining an average of ~14-15 candidates per S1 entity.
- **21-Dimensional Pairwise Feature Engineering**: Captures token sort/set ratios, character n-gram similarities, street digit Jaccard overlap, numerical mismatch penalties, first-word brand matches, and source indicators.
- **Gradient Boosted Ranking with Singleton Calibration**: LightGBM binary classifier trained on hard negatives and positive pairs, with threshold grid search calibrated specifically for the competition's macro $F_{0.5}$ metric.

## 2. Directory Structure
```
code/
├── src/
│   ├── normalization.py       # Text cleaning, transliteration, domain-slug segmentation, legal & address parsing
│   ├── blocking.py            # High-speed inverted index candidate generator
│   └── features.py            # Pairwise feature extraction
├── train_full_model.py        # Trains the LightGBM model and calibrates the decision threshold
├── run_final_pipeline.py      # Runs the trained model against the test set to produce submission files
├── offline_eval.py            # Shared utilities for the held-out validation harness
├── compare_strategies.py      # Held-out validation harness (produces the numbers in Documentation_template.md)
├── validate_submission.py     # Validates output files against the platform's submission rules
├── requirements.txt           # Pinned dependencies
└── README.md                  # This file
```

## 3. Environment Setup
```bash
pip install -r code/requirements.txt
```

## 4. End-to-End Execution
Train the model:
```bash
python3 code/train_full_model.py \
    --num-train-s1 100000 \
    --model-out output/champion_lgb_model.txt
```

Generate the submission files (`matching_results.tsv` and `candidate_pairs.tsv`) for the test set:
```bash
python3 code/run_final_pipeline.py \
    --model-path output/champion_lgb_model.txt \
    --threshold 0.60 \
    --test-dir dataset/test \
    --output-dir output
```

## 5. Validation
Verify the generated files against the platform's submission rules:
```bash
python3 code/validate_submission.py dataset/test output/matching_results.tsv output/candidate_pairs.tsv
```
