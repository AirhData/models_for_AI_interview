# Dockerfile avec pré-chargement des modèles
FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    PYTHONDONTWRITEBYTECODE=1

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

WORKDIR /app

COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Stage de production avec pré-chargement
FROM python:3.11-slim

# Installation des dépendances runtime
RUN apt-get update && apt-get install -y \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Configuration des caches
ENV TRANSFORMERS_CACHE=/app/model_cache
ENV HF_HOME=/app/model_cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/model_cache

# Création des répertoires pour les modèles
RUN mkdir -p /app/model_cache && chmod 755 /app/model_cache

WORKDIR /app

# Copie des dépendances
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copie du code source
COPY . .

# Script de pré-chargement des modèles
COPY preload_models.py .

# PRÉ-CHARGEMENT DES MODÈLES AU BUILD TIME
RUN python preload_models.py

# Création de l'utilisateur non-root
RUN addgroup --system app && adduser --system --group app

# Attribution des permissions
RUN chown -R app:app /app

USER app

# Configuration pour Cloud Run
ENV PORT=8080
ENV PYTHONPATH=/app

EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:${PORT}/health')"

# Commande de démarrage
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1 --timeout-keep-alive 300
