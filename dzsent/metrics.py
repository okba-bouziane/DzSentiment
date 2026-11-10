"""Metrics computed the same way for every system, from a predictions file."""

from __future__ import annotations

from collections import defaultdict

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from . import LABELS, LANGS


def macro_f1(y_true: list[int], y_pred: list[int]) -> float:
    return float(f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0))


def score(rows: list[dict], preds: dict[str, int]) -> dict:
    """rows: test rows with id/label/lang/script. preds: id -> predicted label.

    Returns overall and per-language macro-F1, accuracy, per-class F1, confusion matrices,
    and the mean of the four per-language macro-F1 scores (each language weighs the same).
    """
    missing = [r["id"] for r in rows if r["id"] not in preds]
    if missing:
        raise ValueError(f"{len(missing)} test rows have no prediction, e.g. {missing[:3]}")

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["lang"]].append(r)
        if r["lang"] == "darija":
            groups[f"darija/{r['script']}"].append(r)

    def block(rs: list[dict]) -> dict:
        y = [r["label"] for r in rs]
        p = [preds[r["id"]] for r in rs]
        per_class = f1_score(y, p, labels=[0, 1, 2], average=None, zero_division=0)
        return {
            "n": len(rs),
            "macro_f1": macro_f1(y, p),
            "accuracy": float(accuracy_score(y, p)),
            "f1_per_class": {LABELS[i]: float(v) for i, v in enumerate(per_class)},
            "confusion": confusion_matrix(y, p, labels=[0, 1, 2]).tolist(),
        }

    out = {"overall": block(rows), "by_lang": {}, "darija_by_script": {}}
    for key, rs in groups.items():
        if key.startswith("darija/"):
            out["darija_by_script"][key.split("/", 1)[1]] = block(rs)
        else:
            out["by_lang"][key] = block(rs)
    present = [l for l in LANGS if l in out["by_lang"]]
    out["mean_lang_macro_f1"] = sum(out["by_lang"][l]["macro_f1"] for l in present) / len(present)
    return out
