"""Export a fine-tuned model to ONNX, quantise the weights to 8 bits, and score the 8-bit model.

  python scripts/export_onnx.py --model-dir models/xlmr-base --name xlmr-base

Writes build/onnx/<name>/onnx/model_quantized.onnx (the layout transformers.js expects, next to
config.json and the tokenizer files) and results/predictions/<name>-int8.jsonl, scored on the
full test set with ONNX Runtime on CPU.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dzsent.data import load_split  # noqa: E402
from dzsent.predictions import write_predictions  # noqa: E402


def export(model_dir: Path, out: Path) -> tuple[Path, Path]:
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    (out / "onnx").mkdir(parents=True, exist_ok=True)
    fp32 = out / "onnx" / "model.onnx"
    sample = tok(["salam", "merci bzaf khoya"], padding=True, return_tensors="pt")
    torch.onnx.export(
        model, (sample["input_ids"], sample["attention_mask"]), str(fp32),
        input_names=["input_ids", "attention_mask"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "sequence"}, "attention_mask": {0: "batch", 1: "sequence"},
                      "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    from onnxruntime.quantization import QuantType, quantize_dynamic

    q8 = out / "onnx" / "model_quantized.onnx"
    quantize_dynamic(str(fp32), str(q8), weight_type=QuantType.QInt8)
    for f in model_dir.iterdir():
        if f.name in {"config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                      "sentencepiece.bpe.model", "vocab.txt"}:
            shutil.copy(f, out / f.name)
    return fp32, q8


def score_onnx(path: Path, model_dir: Path, rows: list[dict], batch: int = 64) -> tuple[list[int], float]:
    import onnxruntime as ort

    tok = AutoTokenizer.from_pretrained(model_dir)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    preds = []
    t0 = time.time()
    for i in range(0, len(rows), batch):
        enc = tok([r["text"] for r in rows[i:i + batch]], truncation=True, max_length=128, padding=True,
                  return_tensors="np")
        logits = sess.run(["logits"], {"input_ids": enc["input_ids"].astype(np.int64),
                                       "attention_mask": enc["attention_mask"].astype(np.int64)})[0]
        preds += logits.argmax(-1).tolist()
    return preds, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/xlmr-base")
    ap.add_argument("--name", default="xlmr-base")
    args = ap.parse_args()
    model_dir, out = Path(args.model_dir), Path("build/onnx") / args.name
    fp32, q8 = export(model_dir, out)
    sizes = {p.name: round(p.stat().st_size / 1e6, 1) for p in (fp32, q8)}
    print("sizes MB:", sizes, flush=True)
    test = load_split("test")
    preds, secs = score_onnx(q8, model_dir, test)
    reference = dict(zip([r["id"] for r in test], preds))
    from dzsent.predictions import read_predictions

    fp = read_predictions(args.name)
    agree = sum(fp[i] == p for i, p in reference.items())
    write_predictions(f"{args.name}-int8", [r["id"] for r in test], preds, {
        "source_model": str(model_dir), "quantization": "onnxruntime quantize_dynamic, QInt8 weights",
        "onnx_sizes_mb": sizes, "agreement_with_full_precision": f"{agree}/{len(test)}",
        "cpu_seconds_full_test": round(secs, 1),
    })
    print(f"int8 agrees with full precision on {agree}/{len(test)} test rows")


if __name__ == "__main__":
    main()
