"""Print the README tables from results/metrics.json and results/latency.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dzsent import LANGS  # noqa: E402

NAMES = {
    "tfidf-logreg": "TF-IDF + logistic regression",
    "zeroshot-gpt-oss-20b": "Zero-shot `gpt-oss-20b`",
    "zeroshot-gpt-oss-120b": "Zero-shot `gpt-oss-120b`",
    "xlmr-base": "XLM-RoBERTa base, fine-tuned",
    "xlmr-base-int8": "XLM-RoBERTa base, fine-tuned, 8-bit ONNX (in-browser demo)",
    "xlmr-base-os4": "XLM-RoBERTa base, fine-tuned, small languages x4",
    "dziribert": "DziriBERT, fine-tuned",
}


def f(x: float) -> str:
    return f"{x:.3f}"


def main() -> None:
    m = json.loads(Path("results/metrics.json").read_text())
    order = [k for k in NAMES if k in m] + [k for k in m if k not in NAMES]
    full = [k for k in order if not m[k]["sampled"]]

    print("Full test set (7,602 rows)\n")
    print("| System | Darija | Arabic | French | English | Mean of 4 | Overall macro-F1 | Accuracy |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for k in full:
        r = m[k]
        cells = [f(r["by_lang"][l]["macro_f1"]) for l in LANGS]
        print(f"| {NAMES.get(k, k)} | {' | '.join(cells)} | **{f(r['mean_lang_macro_f1'])}** | "
              f"{f(r['overall']['macro_f1'])} | {f(r['overall']['accuracy'])} |")

    print("\nSame 800-row sample the LLMs labelled (200 per language)\n")
    print("| System | Darija | Arabic | French | English | Mean of 4 |")
    print("| --- | --- | --- | --- | --- | --- |")
    for k in order:
        r = m[k]
        if r["sampled"]:
            langs, mean = {l: r["by_lang"][l]["macro_f1"] for l in LANGS}, r["mean_lang_macro_f1"]
        else:
            langs, mean = r["on_llm_sample"]["by_lang"], r["on_llm_sample"]["mean_lang_macro_f1"]
        print(f"| {NAMES.get(k, k)} | {' | '.join(f(langs[l]) for l in LANGS)} | **{f(mean)}** |")

    print("\nDarija by script (full test set)\n")
    scripts = ["arabic", "latin", "mixed"]
    n = {s: m[full[0]]["darija_by_script"][s]["n"] for s in scripts}
    print("| System | " + " | ".join(f"{s} letters (n={n[s]})" for s in scripts) + " |")
    print("| --- |" + " --- |" * len(scripts))
    for k in full:
        print(f"| {NAMES.get(k, k)} | " + " | ".join(f(m[k]["darija_by_script"][s]["macro_f1"]) for s in scripts) + " |")

    print("\nPer-class F1, full test set\n")
    print("| System | Language | negative | neutral | positive |")
    print("| --- | --- | --- | --- | --- |")
    for k in full:
        for l in LANGS:
            c = m[k]["by_lang"][l]["f1_per_class"]
            print(f"| {NAMES.get(k, k)} | {l} | {f(c['negative'])} | {f(c['neutral'])} | {f(c['positive'])} |")

    lat_path = Path("results/latency.json")
    if lat_path.exists():
        lat = json.loads(lat_path.read_text())
        print(f"\nCPU inference, {lat['threads']} threads, {lat['machine']}, {lat['comments']} test comments\n")
        print("| System | Size on disk | One comment p50 | One comment p95 | Batch of 32, per comment |")
        print("| --- | --- | --- | --- | --- |")
        for k, v in lat["models"].items():
            print(f"| {NAMES.get(k, k)} | {v['size_mb']} MB | {v['single_p50_ms']} ms | {v['single_p95_ms']} ms | "
                  f"{v['batch32_ms_per_comment']} ms |")

    for k in full:
        meta = m[k].get("meta", {})
        if "validation_per_epoch" in meta:
            print(f"\n{NAMES.get(k, k)}: {meta['train_minutes']} min on {meta['device']}, "
                  f"{meta['parameters_millions']}M parameters; validation mean-of-4 per epoch: "
                  + ", ".join(f(e["mean_lang_macro_f1"]) for e in meta["validation_per_epoch"]))


if __name__ == "__main__":
    main()
