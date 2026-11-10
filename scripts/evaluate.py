"""Score every predictions file in results/predictions against the test split.

Writes results/metrics.json and confusion-matrix figures in results/figures/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dzsent import LABELS, LANGS  # noqa: E402
from dzsent.data import load_split  # noqa: E402
from dzsent.llm import sample_ids  # noqa: E402
from dzsent.metrics import score  # noqa: E402
from dzsent.predictions import PRED_DIR, read_predictions  # noqa: E402


def plot_confusions(name: str, m: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    for ax, lang in zip(axes, LANGS):
        b = m["by_lang"].get(lang)
        if b is None:
            ax.axis("off")
            continue
        cm = b["confusion"]
        totals = [max(sum(row), 1) for row in cm]
        norm = [[v / t for v in row] for row, t in zip(cm, totals)]
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i][j]}\n{norm[i][j]:.0%}", ha="center", va="center", fontsize=9,
                        color="white" if norm[i][j] > 0.55 else "#1f2328")
        ax.set_xticks(range(3), [l[:3] for l in LABELS])
        ax.set_yticks(range(3), [l[:3] for l in LABELS])
        ax.set_xlabel("predicted")
        if lang == LANGS[0]:
            ax.set_ylabel("true")
        ax.set_title(f"{lang} (n={b['n']}, macro-F1 {b['macro_f1']:.3f})", fontsize=10)
    fig.suptitle(name, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    test = load_split("test")
    by_id = {r["id"]: r for r in test}
    sample = [by_id[i] for i in sample_ids(test, 200)]  # the rows the zero-shot LLMs labelled
    results = {}
    for path in sorted(PRED_DIR.glob("*.jsonl")):
        name = path.stem
        preds = read_predictions(name)
        rows = test if len(preds) == len(test) else [by_id[i] for i in preds]  # sampled systems
        m = score(rows, preds)
        m["sampled"] = len(preds) != len(test)
        if not m["sampled"]:
            sm = score(sample, preds)
            m["on_llm_sample"] = {"mean_lang_macro_f1": sm["mean_lang_macro_f1"],
                                  "by_lang": {l: b["macro_f1"] for l, b in sm["by_lang"].items()}}
        meta = PRED_DIR / f"{name}.meta.json"
        if meta.exists():
            m["meta"] = json.loads(meta.read_text())
        results[name] = m
        if not args.no_figures:
            Path("results/figures").mkdir(parents=True, exist_ok=True)
            plot_confusions(name, m, Path(f"results/figures/confusion-{name}.png"))
        langs = " ".join(f"{l} {m['by_lang'][l]['macro_f1']:.3f}" for l in LANGS if l in m["by_lang"])
        print(f"{name:28s} n={m['overall']['n']:5d} mean-lang {m['mean_lang_macro_f1']:.3f} | {langs}")
    Path("results/metrics.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
