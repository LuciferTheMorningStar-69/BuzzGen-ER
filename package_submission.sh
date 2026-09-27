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

# Final submission package structure per the official contest rules:
# output/ (both files) + code/business_entity_resolution/ (runnable
# pipeline) + Documentation_template.md. This directory's scripts are
# the same ones that actually produced the delivered output/ files
# (train_full_model.py, run_final_pipeline.py), not the previously-stale
# src/pipeline.py + src/model.py that were replaced earlier tonight.
zip -q -r "$ZIP_NAME" \
    output/matching_results.tsv \
    output/candidate_pairs.tsv \
    code/business_entity_resolution \
    Documentation_template.md \
    -x "*/__pycache__/*"

cp "$ZIP_NAME" Team_BuzzGen_submission.zip

echo "Submission package created: $ZIP_NAME and Team_BuzzGen_submission.zip"
ls -lh "$ZIP_NAME"
unzip -l "$ZIP_NAME"
