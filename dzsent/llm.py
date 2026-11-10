"""Zero-shot LLM baseline: prompt building and answer parsing (no network code here)."""

from __future__ import annotations

import json
import re

from . import LABELS

SYSTEM = (
    "You label the sentiment of short texts: YouTube comments in Algerian Arabic (Darija, in Arabic "
    "or Latin letters, often mixed with French), and tweets in Arabic, French or English. "
    "For each text answer negative, neutral or positive. Neutral means no clear positive or negative "
    "opinion (facts, questions, news, greetings). Answer with JSON only."
)


def build_prompt(texts: list[str]) -> str:
    lines = "\n".join(f"{i + 1}. {json.dumps(t, ensure_ascii=False)}" for i, t in enumerate(texts))
    return (
        f"Label each of these {len(texts)} texts.\n\n{lines}\n\n"
        'Reply exactly as {"labels": ["negative" | "neutral" | "positive", ...]} '
        f"with {len(texts)} labels, in the same order."
    )


_OBJ = re.compile(r"\{.*\}", re.S)


def parse_labels(reply: str, n: int) -> list[int] | None:
    """Return n label ids, or None if the reply is unusable."""
    m = _OBJ.search(reply or "")
    if not m:
        return None
    try:
        labels = json.loads(m.group(0)).get("labels")
    except (json.JSONDecodeError, AttributeError):
        return None
    if not isinstance(labels, list) or len(labels) != n:
        return None
    out = []
    for label in labels:
        key = str(label).strip().lower()
        if key not in LABELS:
            return None
        out.append(LABELS.index(key))
    return out


def sample_ids(rows: list[dict], per_lang: int, seed: int = 7) -> list[str]:
    """Same random sample of test ids per language for every run."""
    import random

    rng = random.Random(seed)
    ids = []
    for lang in sorted({r["lang"] for r in rows}):
        group = sorted((r["id"] for r in rows if r["lang"] == lang))
        ids += rng.sample(group, min(per_lang, len(group)))
    return sorted(ids)
