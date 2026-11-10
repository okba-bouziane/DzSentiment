"""Download the pinned source files and write data/processed/{train,validation,test}.jsonl.

Darija: the source has one file, so it is split 80/10/10 per label with seed 42.
Arabic, French, English tweets: the official train/validation/test splits are kept.
Each row: {"id", "text", "label" (0 neg, 1 neu, 2 pos), "lang", "script", "source"}.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dzsent import data as D  # noqa: E402


def fetch(url: str, cache: Path) -> bytes:
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return cache.read_bytes()
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    cache.write_bytes(r.content)
    return r.content


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    raw, out = Path(args.raw), Path(args.out)
    stats: dict = {"sources": {}}

    # Darija YouTube comments
    blob = fetch(D.hf_url(D.DZ_REPO, D.DZ_REV, D.DZ_FILE), raw / "dz" / D.DZ_FILE)
    df = pd.read_csv(io.BytesIO(blob))
    rows = [
        {"text": D.clean(t), "label": int(l), "lang": "darija", "source": D.DZ_REPO}
        for t, l in zip(df["text"], df["label"])
    ]
    rows, dups, conflicts = D.dedupe(rows)
    stats["sources"]["darija"] = {"raw_rows": len(df), "duplicates_removed": dups, "conflicting_texts_removed": conflicts}
    splits = D.stratified_split(rows, seed=args.seed)
    stats["sources"]["darija"]["cross_split_removed"] = D.remove_cross_split_duplicates(splits)

    # Tweets: official splits per language
    for lang in D.TWEET_LANGS:
        tw_splits = {}
        for s in D.SPLITS:
            path = f"data/{lang}/{s}.jsonl"
            blob = fetch(D.hf_url(D.TWEETS_REPO, D.TWEETS_REV, path), raw / "tweets" / lang / f"{s}.jsonl")
            tw_splits[s] = [
                {"text": D.clean(j["text"]), "label": int(j["label"]), "lang": lang, "source": D.TWEETS_REPO}
                for j in map(json.loads, blob.decode("utf-8").splitlines()) if j
            ]
        n_raw = sum(len(v) for v in tw_splits.values())
        removed = D.remove_cross_split_duplicates(tw_splits)
        stats["sources"][lang] = {"raw_rows": n_raw, "cross_split_removed": removed}
        for s in D.SPLITS:
            splits[s] += tw_splits[s]

    for s in D.SPLITS:
        for i, r in enumerate(splits[s]):
            r["id"] = f"{s}-{i:05d}"
            r["script"] = D.script_of(r["text"])
        splits[s] = [{k: r[k] for k in ("id", "text", "label", "lang", "script", "source")} for r in splits[s]]
        D.write_jsonl(out / f"{s}.jsonl", splits[s])

    import hashlib
    stats["sha256"] = {s: hashlib.sha256((out / f"{s}.jsonl").read_bytes()).hexdigest() for s in D.SPLITS}
    stats["splits"] = {
        s: {
            "rows": len(splits[s]),
            "by_lang": dict(Counter(r["lang"] for r in splits[s])),
            "by_label": dict(Counter(r["label"] for r in splits[s])),
            "darija_script": dict(Counter(r["script"] for r in splits[s] if r["lang"] == "darija")),
        }
        for s in D.SPLITS
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/data_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats["splits"], indent=2))


if __name__ == "__main__":
    main()
