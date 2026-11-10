"""Zero-shot baseline: an LLM labels a fixed random sample of the test set (200 per language).

Texts go 20 per request to keep token use low on a free tier. Needs an OpenAI-compatible
endpoint: LLM_BASE_URL (default Groq) and LLM_API_KEY.

  python scripts/baseline_llm.py --model openai/gpt-oss-20b
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from openai import OpenAI, RateLimitError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dzsent.data import load_split  # noqa: E402
from dzsent.llm import SYSTEM, build_prompt, parse_labels, sample_ids  # noqa: E402
from dzsent.predictions import write_predictions  # noqa: E402


def ask(client: OpenAI, model: str, texts: list[str], effort: str | None) -> tuple[list[int] | None, dict]:
    kwargs = {"reasoning_effort": effort} if effort else {}
    for attempt in range(6):
        try:
            r = client.chat.completions.create(
                model=model, temperature=0, response_format={"type": "json_object"},
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": build_prompt(texts)}],
                **kwargs,
            )
            usage = {"in": r.usage.prompt_tokens, "out": r.usage.completion_tokens}
            return parse_labels(r.choices[0].message.content, len(texts)), usage
        except RateLimitError:
            time.sleep(min(60, 5 * 2 ** attempt))
        except Exception as e:  # malformed JSON from json mode, timeouts
            print("request failed:", type(e).__name__, str(e)[:120], flush=True)
            time.sleep(3)
    return None, {"in": 0, "out": 0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--per-lang", type=int, default=200)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--effort", default="low", help="reasoning_effort, empty to omit")
    args = ap.parse_args()

    client = OpenAI(base_url=os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
                    api_key=os.environ["LLM_API_KEY"])
    test = {r["id"]: r for r in load_split("test")}
    ids = sample_ids(list(test.values()), args.per_lang)
    preds: dict[str, int] = {}
    tokens = {"in": 0, "out": 0}
    failed_batches = 0
    t0 = time.time()
    for start in range(0, len(ids), args.batch):
        chunk = ids[start:start + args.batch]
        labels, usage = ask(client, args.model, [test[i]["text"] for i in chunk], args.effort or None)
        tokens = {k: tokens[k] + usage[k] for k in tokens}
        if labels is None:  # retry one by one
            failed_batches += 1
            for i in chunk:
                one, usage = ask(client, args.model, [test[i]["text"]], args.effort or None)
                tokens = {k: tokens[k] + usage[k] for k in tokens}
                if one is not None:
                    preds[i] = one[0]
        else:
            preds.update(zip(chunk, labels))
        print(f"{len(preds)}/{len(ids)} labelled", flush=True)

    unanswered = [i for i in ids if i not in preds]
    for i in unanswered:  # counted as wrong: predict a label that is never the true one
        preds[i] = (test[i]["label"] + 1) % 3
    name = "zeroshot-" + args.model.split("/")[-1]
    write_predictions(name, list(preds), list(preds.values()), {
        "model": args.model, "endpoint": client.base_url.host, "reasoning_effort": args.effort,
        "texts_per_request": args.batch, "sample_size": len(ids), "unanswered_counted_wrong": len(unanswered),
        "batches_retried_one_by_one": failed_batches, "tokens": tokens,
        "minutes": round((time.time() - t0) / 60, 1),
    })
    print(name, "done; unanswered:", len(unanswered), "tokens:", tokens)


if __name__ == "__main__":
    main()
