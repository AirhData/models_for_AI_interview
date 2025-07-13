# Dockerfile optimisé pour Cloud Run
FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    PYTHONDONTWRITEBYTECODE=1

# Installation des dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

WORKDIR /app

COPY requirements.txt .

# Installation des dépendances Python
RUN uv pip install --system --no-cache -r requirements.txt

# Stage de production
FROM python:3.11-slim

# Installation des dépendances runtime
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Création d'un utilisateur non-root pour la sécurité
RUN addgroup --system app && adduser --system --group app

WORKDIR /app

# Copie des dépendances depuis le builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copie du code source
COPY . .

# Configuration des permissions
RUN chown -R app:app /app

# Configuration pour Cloud Run
ENV PORT=8080
ENV PYTHONPATH=/app
ENV TRANSFORMERS_CACHE=/tmp/transformers_cache
ENV HF_HOME=/tmp/huggingface

# Création des répertoires temporaires avec bonnes permissions
RUN mkdir -p /tmp/transformers_cache /tmp/huggingface && \
    chown -R app:app /tmp/transformers_cache /tmp/huggingface

USER app

# Exposition du port (Cloud Run utilise PORT env var)
EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:${PORT}/health')"

# Commande de démarrage optimisée pour Cloud Run
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1 --timeout-keep-alive 300
