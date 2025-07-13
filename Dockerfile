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

# Créer l'utilisateur app avec un répertoire home
RUN useradd -m -s /bin/bash app

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Pré-télécharger les modèles en tant que root
ENV HOME=/root
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" && \
    python -c "from transformers import pipeline; pipeline('text-classification', model='astrosbd/french_emotion_camembert', return_all_scores=True)" && \
    python -c "from transformers import pipeline; pipeline('zero-shot-classification', model='joeddav/xlm-roberta-large-xnli')"

# Copier les modèles vers le répertoire de l'utilisateur app
RUN cp -r /root/.cache /home/app/ 2>/dev/null || true && \
    chown -R app:app /home/app

COPY . .

RUN mkdir uploads && chown -R app:app uploads && chown -R app:app /app

USER app

# Variables d'environnement importantes
ENV HOME=/home/app \
    HF_HOME=/home/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/app/.cache/huggingface/transformers \
    SENTENCE_TRANSFORMERS_HOME=/home/app/.cache/torch/sentence_transformers

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
