"""CPU latency and size of each saved model: one comment at a time, and a batch of 32.

Runs on CPU with a fixed thread count so numbers are comparable to a small server.
Writes results/latency.json.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dzsent.data import load_split  # noqa: E402


def dir_mb(path: Path) -> float:
    return round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    import platform
    import subprocess

    import joblib
    import torch

    from dzsent.classifier import SentimentClassifier

    torch.set_num_threads(args.threads)
    texts = [r["text"] for r in load_split("test")][: args.n]
    try:  # macOS gives the chip name here; elsewhere fall back to the architecture
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    except OSError:
        cpu = ""
    out = {"threads": args.threads, "machine": f"{cpu or platform.processor()} ({platform.machine()})",
           "comments": len(texts), "models": {}}

    def measure(fn) -> dict:
        fn(texts[:8])  # warm-up
        single = []
        for t in texts:
            t0 = time.perf_counter(); fn([t]); single.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        for i in range(0, len(texts), 32):
            fn(texts[i:i + 32])
        per_comment_batched = (time.perf_counter() - t0) * 1000 / len(texts)
        single.sort()
        return {"single_p50_ms": round(statistics.median(single), 1),
                "single_p95_ms": round(single[int(0.95 * len(single)) - 1], 1),
                "batch32_ms_per_comment": round(per_comment_batched, 2)}

    tfidf = Path("models/tfidf-logreg.joblib")
    if tfidf.exists():
        pipe = joblib.load(tfidf)
        out["models"]["tfidf-logreg"] = {**measure(pipe.predict), "size_mb": round(tfidf.stat().st_size / 1e6, 1)}
    for d in sorted(p for p in Path("models").iterdir() if (p / "config.json").exists()):
        clf = SentimentClassifier(d, threads=args.threads)
        out["models"][d.name] = {**measure(clf.predict), "size_mb": dir_mb(d)}
        print(d.name, out["models"][d.name], flush=True)
    Path("results/latency.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
