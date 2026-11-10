# Full pipeline. Each step writes to results/; the README tables come from `make report`.
PY ?= .venv/bin/python
# On Apple GPUs, cap cached memory so long runs do not push macOS into swap (ignored elsewhere).
MPS_LIMITS = PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5

data:        ; $(PY) scripts/prepare_data.py
tfidf:       ; $(PY) scripts/baseline_tfidf.py
zeroshot:    ; $(PY) scripts/baseline_llm.py --model openai/gpt-oss-20b && $(PY) scripts/baseline_llm.py --model openai/gpt-oss-120b
xlmr:        ; $(MPS_LIMITS) $(PY) scripts/train.py --model FacebookAI/xlm-roberta-base --name xlmr-base
dziribert:   ; $(MPS_LIMITS) $(PY) scripts/train.py --model alger-ia/dziribert --name dziribert
evaluate:    ; $(PY) scripts/evaluate.py
latency:     ; $(PY) scripts/latency.py
export:      ; $(PY) scripts/export_onnx.py
report:      ; $(PY) scripts/report.py
test:        ; $(PY) -m pytest -q
demo:        ; MODEL_DIR=models/xlmr-base $(PY) -m uvicorn app.main:app --port 7860

.PHONY: data tfidf zeroshot xlmr dziribert evaluate latency export report test demo
