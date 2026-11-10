"""Baseline: TF-IDF (word 1-2 grams + character 2-5 grams) and logistic regression.

C is chosen on the validation set (mean of per-language macro-F1), then the model is
refit on train only and scored on test. Saves the pipeline to models/tfidf-logreg.joblib.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dzsent.data import load_split  # noqa: E402
from dzsent.metrics import score  # noqa: E402
from dzsent.predictions import write_predictions  # noqa: E402


def build(c: float) -> Pipeline:
    feats = FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, lowercase=True)),
    ])
    return Pipeline([("feats", feats), ("clf", LogisticRegression(C=c, max_iter=2000, class_weight="balanced"))])


def main() -> None:
    train, val, test = load_split("train"), load_split("validation"), load_split("test")
    x = [r["text"] for r in train]
    y = [r["label"] for r in train]
    search = {}
    for c in (0.5, 1.0, 2.0, 4.0, 8.0):
        model = build(c).fit(x, y)
        preds = dict(zip([r["id"] for r in val], model.predict([r["text"] for r in val]).tolist()))
        search[c] = score(val, preds)["mean_lang_macro_f1"]
        print(f"C={c}: validation mean per-language macro-F1 {search[c]:.4f}", flush=True)
    best = max(search, key=search.get)
    t0 = time.time()
    model = build(best).fit(x, y)
    fit_s = time.time() - t0
    preds = model.predict([r["text"] for r in test]).tolist()
    write_predictions("tfidf-logreg", [r["id"] for r in test], preds,
                      {"C_search_validation": search, "best_C": best, "fit_seconds": round(fit_s, 1)})
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/tfidf-logreg.joblib")
    print(json.dumps({"best_C": best, "fit_seconds": round(fit_s, 1)}))


if __name__ == "__main__":
    main()
