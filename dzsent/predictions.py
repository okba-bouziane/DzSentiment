"""Every system writes results/predictions/<name>.jsonl with {"id", "pred"} rows."""

from __future__ import annotations

import json
from pathlib import Path

PRED_DIR = Path("results/predictions")


def write_predictions(name: str, ids: list[str], preds: list[int], extra: dict | None = None) -> Path:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    path = PRED_DIR / f"{name}.jsonl"
    with path.open("w") as f:
        for i, p in zip(ids, preds):
            f.write(json.dumps({"id": i, "pred": int(p)}) + "\n")
    if extra is not None:
        (PRED_DIR / f"{name}.meta.json").write_text(json.dumps(extra, indent=2))
    return path


def read_predictions(name: str) -> dict[str, int]:
    path = PRED_DIR / f"{name}.jsonl"
    return {j["id"]: j["pred"] for j in map(json.loads, path.read_text().splitlines()) if j}
