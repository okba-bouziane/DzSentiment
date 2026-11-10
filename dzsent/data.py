"""Loading, cleaning and splitting the two source datasets.

Sources are pinned to a dataset commit so every run sees the same rows.
"""

from __future__ import annotations

import json
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

DZ_REPO = "Abdou/dz-sentiment-yt-comments"
DZ_REV = "062b3027196abaa728f62d8dff65dc5ce1b4863d"
DZ_FILE = "ADArabic-3labels-50016.csv"

TWEETS_REPO = "cardiffnlp/tweet_sentiment_multilingual"
TWEETS_REV = "14b13edfbc4046892f6011d114c29c0f83170589"
TWEET_LANGS = {"arabic": "arabic", "french": "french", "english": "english"}

SPLITS = ("train", "validation", "test")

_ARABIC = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿ]")
_LATIN = re.compile(r"[A-Za-zÀ-ɏ]")
_SPACES = re.compile(r"\s+")


def hf_url(repo: str, rev: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}"


def clean(text: str) -> str:
    """Light cleaning only: Unicode NFC and collapsed whitespace. Emoji and spelling stay."""
    return _SPACES.sub(" ", unicodedata.normalize("NFC", str(text))).strip()


def dedup_key(text: str) -> str:
    """Key used to find duplicates: case-folded, whitespace removed."""
    return _SPACES.sub("", clean(text).casefold())


def script_of(text: str) -> str:
    has_ar = bool(_ARABIC.search(text))
    has_lat = bool(_LATIN.search(text))
    if has_ar and has_lat:
        return "mixed"
    if has_ar:
        return "arabic"
    if has_lat:
        return "latin"
    return "other"


def stratified_split(rows: list[dict], fractions=(0.8, 0.1, 0.1), seed: int = 42) -> dict[str, list[dict]]:
    """Split rows per label so each split keeps the label balance. Deterministic for a seed."""
    assert abs(sum(fractions) - 1) < 1e-9
    by_label: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)
    rng = random.Random(seed)
    out = {s: [] for s in SPLITS}
    for label in sorted(by_label):
        group = by_label[label][:]
        rng.shuffle(group)
        n = len(group)
        n_train = round(n * fractions[0])
        n_val = round(n * fractions[1])
        out["train"] += group[:n_train]
        out["validation"] += group[n_train:n_train + n_val]
        out["test"] += group[n_train + n_val:]
    for s in SPLITS:
        rng.shuffle(out[s])
    return out


def remove_cross_split_duplicates(splits: dict[str, list[dict]]) -> dict[str, int]:
    """Drop rows from validation/test whose text also appears in train (or val, for test).

    Returns how many rows were dropped per split. Mutates `splits`.
    """
    seen = {dedup_key(r["text"]) for r in splits["train"]}
    dropped = {}
    for s in ("validation", "test"):
        keep = [r for r in splits[s] if dedup_key(r["text"]) not in seen]
        dropped[s] = len(splits[s]) - len(keep)
        splits[s] = keep
        seen |= {dedup_key(r["text"]) for r in keep}
    return dropped


def dedupe(rows: list[dict]) -> tuple[list[dict], int, int]:
    """Keep one row per text. Texts that appear with different labels are dropped entirely.

    Returns (rows, exact_duplicates_removed, conflicting_texts_removed).
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[dedup_key(r["text"])].append(r)
    out, dups, conflicts = [], 0, 0
    for key, group in groups.items():
        if not key:
            dups += len(group)
            continue
        labels = {r["label"] for r in group}
        if len(labels) > 1:
            conflicts += 1
            dups += len(group) - 1
            continue
        out.append(group[0])
        dups += len(group) - 1
    return out, dups, conflicts


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_split(name: str, data_dir: Path | str = "data/processed") -> list[dict]:
    return read_jsonl(Path(data_dir) / f"{name}.jsonl")
