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

# Create clean zip structure
zip -q -r "$ZIP_NAME" \
    output/matching_results.tsv \
    output/candidate_pairs.tsv \
    code/src \
    code/requirements.txt \
    code/README.md \
    code/run_pipeline.sh \
    code/business_entity_resolution \
    code/champion_resolver.py \
    Documentation_template.md

cp "$ZIP_NAME" Team_BuzzGen_submission.zip

echo "Submission package created: $ZIP_NAME and Team_BuzzGen_submission.zip"
ls -lh "$ZIP_NAME"
unzip -l "$ZIP_NAME"
