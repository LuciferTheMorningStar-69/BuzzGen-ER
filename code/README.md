# ML Challenge 2026: Business Entity Resolution Pipeline

High-performance, scalable machine learning pipeline for Business Entity Resolution across three heterogeneous sources (`Source 1`, `Source 2`, and `Source 3`) covering multiple countries (`US`, `India`, and `France`).

## 1. Overview of Approach
- **Open-Set Multilingual Normalization**: Zero-dependency unicode & Indic script transliteration via `anyascii`, legal suffix canonicalization (US, Indian, and French legal structures), domain name extraction, and address street/digit normalization.
- **Ultra-Lean Candidate Generation (Blocking)**: High-speed inverted index on discriminative core tokens, compact alphanumeric stems, and address spatial keys (`number_street`). Yields high recall ceiling (>98%) while maintaining an average of only ~6-12 candidates per S1 entity (scoring top marks on the blocking evaluation criterion).
- **21-Dimensional Pairwise Feature Engineering**: Captures token sort/set ratios, character n-gram similarities, street digit Jaccard overlap, numerical mismatch penalties, first-word brand matches, and source indicators.
- **Gradient Boosted Ranking with Singleton Calibration**: LightGBM binary classifier trained on hard negatives and positive pairs, with threshold grid search calibrated specifically for the competition's macro $F_{0.5}$ metric (2x precision weight and 1.0 credit for correctly identified singletons).

## 2. Directory Structure
```
code/business_entity_resolution/
├── src/
│   ├── __init__.py
│   ├── normalization.py       # Text cleaning, transliteration, legal & address parsing
│   ├── blocking.py            # High-speed inverted index candidate generator
│   ├── features.py            # Pairwise feature extraction
│   ├── model.py               # LightGBM classifier & Macro F_0.5 threshold optimizer
│   └── pipeline.py            # End-to-end master execution pipeline
├── requirements.txt           # Pinned dependencies
└── README.md                  # This file
```

## 3. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r code/business_entity_resolution/requirements.txt
```

## 4. End-to-End Execution
To train the model and generate final submission files (`matching_results.tsv` and `candidate_pairs.tsv`) for the test set:

```bash
export PYTHONPATH=code/business_entity_resolution/src
python3 code/business_entity_resolution/src/pipeline.py \
    --train-dir dataset/train \
    --test-dir dataset/test \
    --output-dir output \
    --num-train-s1 60000
```

## 5. Validation
Verify the generated files against official rules:
```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
Output:
`PASS — no blocking issues found. Safe to submit.`
