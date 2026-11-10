"""Demo: paste customer comments, get a sentiment per line.

  MODEL_DIR=models/xlmr-base uvicorn app.main:app --port 7860
"""

from __future__ import annotations

import os
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Protocol

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from dzsent import LABELS

MAX_TEXTS = 50
MAX_CHARS = 1000
STATIC = Path(__file__).parent / "static"


class Classifier(Protocol):
    def predict(self, texts: list[str]): ...


class ClassifyIn(BaseModel):
    texts: list[str]


class RateLimiter:
    """Sliding window per client: at most `limit` requests in `window` seconds (in memory, one process)."""

    def __init__(self, limit: int, window: float, clock=time.monotonic):
        self.limit, self.window, self.clock = limit, window, clock
        self.hits: dict[str, deque] = defaultdict(deque)

    def retry_after(self, key: str) -> int:
        """0 if the request may go ahead (and is counted), else seconds to wait."""
        now = self.clock()
        q = self.hits[key]
        while q and now - q[0] >= self.window:
            q.popleft()
        if len(q) >= self.limit:
            return max(1, int(self.window - (now - q[0])) + 1)
        q.append(now)
        return 0


def client_key(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")


def create_app(classifier: Classifier, model_name: str = "", limiter: RateLimiter | None = None) -> FastAPI:
    api = FastAPI(title="DzSentiment", docs_url=None, redoc_url=None)
    limiter = limiter or RateLimiter(int(os.environ.get("RATE_LIMIT_PER_MINUTE", "30")), 60)

    @api.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @api.get("/health")
    def health():
        return {"ok": True, "model": model_name}

    @api.post("/api/classify")
    def classify(body: ClassifyIn, request: Request):
        wait = limiter.retry_after(client_key(request))
        if wait:
            raise HTTPException(429, f"Too many requests. Try again in {wait} seconds.",
                                headers={"Retry-After": str(wait)})
        texts = [t.strip() for t in body.texts if t and t.strip()]
        if not texts:
            raise HTTPException(422, "Add at least one comment.")
        if len(texts) > MAX_TEXTS:
            raise HTTPException(422, f"At most {MAX_TEXTS} comments per request.")
        if any(len(t) > MAX_CHARS for t in texts):
            raise HTTPException(422, f"Each comment must be at most {MAX_CHARS} characters.")
        preds = classifier.predict(texts)
        counts = Counter(p.label for p in preds)
        return {
            "model": model_name,
            "results": [{"text": p.text, "label": p.label, "scores": p.scores} for p in preds],
            "counts": {l: counts.get(l, 0) for l in LABELS},
        }

    return api


def _default_app() -> FastAPI:
    from dzsent.classifier import SentimentClassifier

    model_dir = os.environ.get("MODEL_DIR", "models/xlmr-base")
    threads = int(os.environ.get("TORCH_THREADS", "2"))
    return create_app(SentimentClassifier(model_dir, threads=threads), Path(model_dir).name)


app = _default_app() if os.environ.get("DZSENT_NO_AUTOLOAD") != "1" else None
