#!/bin/bash
set -e
echo "=== Unpacking BuzzGen-ER Dataset ==="

# 1. Reassemble split parts
for part_a in dataset/*/*.tsv.gz.part_aa; do
  if [ -f "$part_a" ]; then
    base="${part_a%.part_aa}"
    echo "Reassembling $(basename "$base")..."
    cat "${base}".part_* > "$base"
    rm -f "${base}".part_*
  fi
done

# 2. Decompress all .tsv.gz files into .tsv
for gz in dataset/*/*.tsv.gz; do
  if [ -f "$gz" ]; then
    echo "Decompressing $(basename "$gz")..."
    gzip -d -f "$gz"
  fi
done

echo "=== Dataset unpack complete! ==="
ls -lh dataset/train/ dataset/test/
