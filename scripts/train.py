"""Fine-tune a pre-trained encoder for 3-class sentiment on all four languages.

Example:
  python scripts/train.py --model FacebookAI/xlm-roberta-base --name xlmr-base
  python scripts/train.py --model alger-ia/dziribert --name dziribert

The checkpoint with the best validation score (mean of per-language macro-F1) is kept,
saved to models/<name>, and used to write test predictions.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    TrainerCallback,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    set_seed,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dzsent import LABELS  # noqa: E402
from dzsent.data import load_split  # noqa: E402
from dzsent.metrics import score  # noqa: E402
from dzsent.predictions import write_predictions  # noqa: E402
from dzsent.training import OrderedEvalTrainer  # noqa: E402

REVISIONS = {
    "FacebookAI/xlm-roberta-base": "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089",
    "alger-ia/dziribert": "4722886048b304627caaeed800733e6376f5d13d",
}


def oversample(rows: list[dict], factor: int, seed: int) -> list[dict]:
    """Repeat the non-Darija rows `factor` times (factor 1 = natural mix)."""
    if factor <= 1:
        return rows
    extra = [r for r in rows if r["lang"] != "darija"] * (factor - 1)
    out = rows + extra
    random.Random(seed).shuffle(out)
    return out


class FreeMpsCache(TrainerCallback):
    """Return cached GPU memory to macOS every N steps (variable shapes fragment the MPS cache)."""

    def __init__(self, every: int = 50):
        self.every = every

    def on_step_end(self, args, state, control, **kwargs):
        if torch.backends.mps.is_available() and state.global_step % self.every == 0:
            torch.mps.empty_cache()


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=16, help="per-step batch; effective batch = batch x accum")
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--freeze-embeddings", action="store_true",
                    help="keep the word-embedding matrix fixed (saves memory on large vocabularies)")
    ap.add_argument("--max-steps", type=int, default=-1, help="stop early (speed probe)")
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--oversample", type=int, default=1, help="repeat Arabic/French/English train rows N times")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="use only N train rows (smoke test)")
    args = ap.parse_args()
    set_seed(args.seed)

    train_rows = oversample(load_split("train"), args.oversample, args.seed)
    if args.limit:
        train_rows = train_rows[: args.limit]
    val_rows, test_rows = load_split("validation"), load_split("test")
    if args.limit:
        val_rows = val_rows[: max(200, args.limit // 4)]

    rev = REVISIONS.get(args.model)
    tok = AutoTokenizer.from_pretrained(args.model, revision=rev)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, revision=rev, num_labels=3,
        id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)},
    )

    if args.freeze_embeddings:
        model.get_input_embeddings().weight.requires_grad_(False)

    def to_ds(rows: list[dict]) -> Dataset:
        ds = Dataset.from_dict({"text": [r["text"] for r in rows], "label": [r["label"] for r in rows]})
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=args.max_len), batched=True,
                      remove_columns=["text"])

    train_ds, val_ds, test_ds = to_ds(train_rows), to_ds(val_rows), to_ds(test_rows)

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=-1).tolist()
        m = score(val_rows, dict(zip([r["id"] for r in val_rows], preds)))
        out = {"mean_lang_macro_f1": m["mean_lang_macro_f1"], "macro_f1": m["overall"]["macro_f1"]}
        out.update({f"f1_{l}": b["macro_f1"] for l, b in m["by_lang"].items()})
        return out

    out_dir = Path("models") / args.name
    targs = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        max_steps=args.max_steps,
        per_device_eval_batch_size=64,
        warmup_ratio=0.06,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="mean_lang_macro_f1",
        greater_is_better=True,
        group_by_length=True,
        logging_steps=100,
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=0,
        fp16=torch.cuda.is_available(),
    )
    trainer = OrderedEvalTrainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=DataCollatorWithPadding(tok, pad_to_multiple_of=32),
                      compute_metrics=compute_metrics, callbacks=[FreeMpsCache()])
    t0 = time.time()
    trainer.train()
    train_s = time.time() - t0

    epochs_log = [h for h in trainer.state.log_history if "eval_mean_lang_macro_f1" in h]
    best_val = trainer.evaluate()
    logits = trainer.predict(test_ds).predictions
    preds = np.argmax(logits, axis=-1).tolist()

    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    import shutil
    shutil.rmtree(out_dir / "checkpoints", ignore_errors=True)

    meta = {
        "base_model": args.model, "revision": rev, "device": device(),
        "args": vars(args), "train_rows": len(train_rows),
        "train_minutes": round(train_s / 60, 1),
        "best_validation": {k.removeprefix("eval_"): round(v, 4) for k, v in best_val.items()
                            if k.startswith("eval_") and "f1" in k},
        "validation_per_epoch": [
            {"epoch": h["epoch"], "mean_lang_macro_f1": round(h["eval_mean_lang_macro_f1"], 4),
             **{k.removeprefix("eval_"): round(v, 4) for k, v in h.items() if k.startswith("eval_f1_")}}
            for h in epochs_log
        ],
        "parameters_millions": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        "trainable_parameters_millions": round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 1),
    }
    if not args.limit:
        write_predictions(args.name, [r["id"] for r in test_rows], preds, meta)
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
