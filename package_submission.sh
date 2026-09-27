#!/bin/bash
set -e

echo "=== Packaging Submission for Team BuzzGen ==="
ZIP_NAME="BuzzGen_submission.zip"
rm -f "$ZIP_NAME" Team_BuzzGen_submission.zip Team_AlphaResolve_submission.zip

# Check output files exist
if [ ! -f "output/matching_results.tsv" ] || [ ! -f "output/candidate_pairs.tsv" ]; then
    echo "Error: Output files missing in output/ directory."
    exit 1
fi

# Create clean zip structure -- only the code that actually produced the
# delivered output/ files. Deliberately excludes code/run_pipeline.sh
# (points at a stale, pre-fix src/pipeline.py with outdated hyperparameters
# and a discarded-threshold bug) and code/champion_resolver.py (an
# abandoned rule-based approach, never used for the delivered results,
# whose docstring falsely claims "Macro F_0.5 > 0.97 - 0.99").
zip -q -r "$ZIP_NAME" \
    output/matching_results.tsv \
    output/candidate_pairs.tsv \
    code/src \
    code/train_full_model.py \
    code/run_final_pipeline.py \
    code/offline_eval.py \
    code/compare_strategies.py \
    code/validate_submission.py \
    code/requirements.txt \
    code/README.md \
    Documentation_template.md

cp "$ZIP_NAME" Team_BuzzGen_submission.zip

echo "Submission package created: $ZIP_NAME and Team_BuzzGen_submission.zip"
ls -lh "$ZIP_NAME"
unzip -l "$ZIP_NAME"
