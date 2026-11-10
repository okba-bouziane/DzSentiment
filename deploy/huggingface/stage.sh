#!/usr/bin/env bash
# Copy what the Space needs into one folder, ready for `upload_folder`.
set -euo pipefail
cd "$(dirname "$0")/../.."
out="${1:-build/space}"
rm -rf "$out" && mkdir -p "$out"
cp deploy/huggingface/Dockerfile deploy/huggingface/README.md "$out/"
cp -r dzsent app "$out/"
find "$out" -name __pycache__ -prune -exec rm -rf {} +
echo "staged in $out"
