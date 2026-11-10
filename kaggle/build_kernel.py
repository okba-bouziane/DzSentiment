"""Build a self-contained Kaggle kernel that prepares the data and fine-tunes both models on a GPU.

The kernel is one script: it unpacks this repo's `dzsent/` and `scripts/` (embedded as base64),
downloads the pinned datasets from Hugging Face, trains, and leaves predictions + models in
/kaggle/working for `kaggle kernels output`.

  python kaggle/build_kernel.py --user <kaggle-username>
  kaggle kernels push -p build/kaggle-kernel
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RUNNER = r'''
import base64, io, os, subprocess, sys, tarfile, time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # one T4, so the effective batch stays 32
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.chdir("/kaggle/working")
tarfile.open(fileobj=io.BytesIO(base64.b64decode(BUNDLE))).extractall(".")

def run(cmd):
    print("$", cmd, flush=True); t0 = time.time()
    subprocess.run(cmd, shell=True, check=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)

run('pip install -q "transformers>=4.45,<5" "accelerate>=1.0" "datasets>=3.0"')
run("nvidia-smi --query-gpu=name,memory.total --format=csv")
run("python scripts/prepare_data.py")
for model, name, extra in MODELS:
    run(f"python scripts/train.py --model {model} --name {name} --batch 32 --accum 1 {extra}")
run("rm -rf data/raw")
'''


def bundle() -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for folder in ("dzsent", "scripts"):
            for f in sorted((ROOT / folder).rglob("*.py")):
                tar.add(f, arcname=str(f.relative_to(ROOT)))
    return base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--slug", default="dzsentiment-train")
    ap.add_argument("--out", default="build/kaggle-kernel")
    ap.add_argument("--models", default=(
        "FacebookAI/xlm-roberta-base:xlmr-base:,"
        "alger-ia/dziribert:dziribert:,"
        "FacebookAI/xlm-roberta-base:xlmr-base-os4:--oversample 4"
    ), help="model:name:extra train.py args, comma separated")
    args = ap.parse_args()
    models = [tuple(m.split(":", 2)) for m in args.models.split(",")]
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.py").write_text(f"BUNDLE = {bundle()!r}\nMODELS = {models!r}\n{RUNNER}")
    (out / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{args.user}/{args.slug}",
        "title": args.slug,
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
    }, indent=2))
    print(f"kernel written to {out}")


if __name__ == "__main__":
    main()
