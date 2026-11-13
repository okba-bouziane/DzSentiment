"""Average the class probabilities of fine-tuned models and write the result as a new system.

The weight of each model is chosen on the validation set only (grid search on the mean of
per-language macro-F1), then applied once to the test set. Needs the `<name>.logits.npy` and
`<name>.val_logits.npy` files that scripts/train.py writes.

  python scripts/ensemble.py --models xlmr-large dziribert --name ens-xlmr-large-dziribert
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dzsent.data import load_split  # noqa: E402
from dzsent.metrics import score  # noqa: E402
from dzsent.predictions import PRED_DIR, write_predictions  # noqa: E402


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--step", type=float, default=0.1)
    args = ap.parse_args()
    val, test = load_split("validation"), load_split("test")
    val_p = [softmax(np.load(PRED_DIR / f"{m}.val_logits.npy")) for m in args.models]
    test_p = [softmax(np.load(PRED_DIR / f"{m}.logits.npy")) for m in args.models]
    assert all(len(p) == len(val) for p in val_p) and all(len(p) == len(test) for p in test_p)

    grid = np.round(np.arange(0, 1 + 1e-9, args.step), 3)
    best = (-1.0, None)
    for w in itertools.product(grid, repeat=len(args.models)):
        if abs(sum(w) - 1) > 1e-6:
            continue
        probs = sum(wi * p for wi, p in zip(w, val_p))
        preds = probs.argmax(axis=1).tolist()
        s = score(val, dict(zip([r["id"] for r in val], preds)))["mean_lang_macro_f1"]
        if s > best[0]:
            best = (s, w)
    val_score, weights = best
    probs = sum(wi * p for wi, p in zip(weights, test_p))
    preds = probs.argmax(axis=1).tolist()
    meta = {"system": "probability average", "models": args.models, "weights": [float(w) for w in weights],
            "weights_chosen_on": "validation", "validation_mean_lang_macro_f1": round(val_score, 4)}
    write_predictions(args.name, [r["id"] for r in test], preds, meta)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
