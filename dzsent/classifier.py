"""Inference wrapper used by the demo app and the latency script."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import LABELS


@dataclass
class Prediction:
    text: str
    label: str
    scores: dict[str, float]


class SentimentClassifier:
    """Loads a fine-tuned checkpoint and labels texts on CPU (or whichever device is given)."""

    def __init__(self, model_dir: str | Path, device: str = "cpu", max_length: int = 128, threads: int | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if threads:
            torch.set_num_threads(threads)
        self._torch = torch
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()

    def predict(self, texts: list[str], batch_size: int = 32) -> list[Prediction]:
        out: list[Prediction] = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            enc = self.tokenizer(chunk, truncation=True, max_length=self.max_length, padding=True,
                                 return_tensors="pt").to(self.device)
            with self._torch.inference_mode():
                probs = self.model(**enc).logits.softmax(-1).cpu().tolist()
            for text, p in zip(chunk, probs):
                best = max(range(len(LABELS)), key=p.__getitem__)
                out.append(Prediction(text, LABELS[best], {l: round(v, 4) for l, v in zip(LABELS, p)}))
        return out
