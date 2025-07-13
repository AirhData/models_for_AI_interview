FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

RUN pip install uv

WORKDIR /app

COPY requirements.txt .

RUN uv pip install --system --no-cache -r requirements.txt

FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Pré-télécharger les modèles
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" && \
    python -c "from transformers import pipeline; pipeline('text-classification', model='astrosbd/french_emotion_camembert', return_all_scores=True)" && \
    python -c "from transformers import pipeline; pipeline('zero-shot-classification', model='joeddav/xlm-roberta-large-xnli')"

COPY . .

RUN mkdir uploads

# Variables d'environnement pour Cloud Run
ENV HOME=/tmp \
    HF_HOME=/tmp/.cache/huggingface \
    TRANSFORMERS_CACHE=/tmp/.cache/huggingface/transformers \
    SENTENCE_TRANSFORMERS_HOME=/tmp/.cache/torch/sentence_transformers

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
