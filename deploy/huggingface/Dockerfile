# Hugging Face Docker Space: the demo app on CPU, port 7860.
# Build context = the staged Space folder (see deploy/huggingface/stage.sh): dzsent/, app/, this Dockerfile.
# The fine-tuned weights are downloaded from a Hub model repo at build time (MODEL_REPO).
FROM python:3.12-slim

ARG MODEL_REPO=okba-bouziane/dzsentiment-xlmr-base
ENV PIP_NO_CACHE_DIR=1 TORCH_THREADS=2 MODEL_DIR=/home/user/model HF_HOME=/home/user/.cache

RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 \
 && pip install "transformers>=4.45,<5" "fastapi>=0.115" "uvicorn>=0.30" "huggingface_hub>=0.25" "scikit-learn>=1.5"

RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL_REPO}', local_dir='/home/user/model')"
COPY --chown=user dzsent ./dzsent
COPY --chown=user app ./app

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
