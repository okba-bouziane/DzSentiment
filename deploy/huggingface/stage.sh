#!/usr/bin/env bash
# Copy what a Space needs into one folder, ready for `upload_folder`.
#   deploy/huggingface/stage.sh static   # free static Space: the page runs the 8-bit ONNX model in the browser
#   deploy/huggingface/stage.sh docker   # Docker Space: FastAPI server with the full-precision model
set -euo pipefail
cd "$(dirname "$0")/../.."
kind="${1:-static}"
out="build/space-$kind"
rm -rf "$out" && mkdir -p "$out"
if [ "$kind" = static ]; then
  cp deploy/huggingface-static/README.md app/static/index.html "$out/"
else
  cp deploy/huggingface/Dockerfile deploy/huggingface/README.md "$out/"
  cp -r dzsent app "$out/"
  find "$out" -name __pycache__ -prune -exec rm -rf {} +
  rm -f "$out/app/static/_check.json"
fi
echo "staged in $out"
