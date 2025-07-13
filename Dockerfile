# Dockerfile corrigé pour Cloud Run
FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    PYTHONDONTWRITEBYTECODE=1

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

WORKDIR /app

COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Stage: Production
FROM python:3.11-slim

# Installation runtime dependencies
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copie des dépendances Python
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copie du code source
COPY . .

# Configuration pour Cloud Run - IMPORTANT
ENV PORT=8080
ENV PYTHONPATH=/app
ENV TRANSFORMERS_CACHE=/tmp/transformers
ENV HF_HOME=/tmp/hf
ENV CREW_STORAGE_DIR=/tmp/crew
ENV HOME=/tmp
ENV TMPDIR=/tmp

# Création des répertoires temporaires avec bonnes permissions
RUN mkdir -p /tmp/transformers /tmp/hf /tmp/crew && \
    chmod 777 /tmp/transformers /tmp/hf /tmp/crew

# Création utilisateur non-root
RUN addgroup --system app && adduser --system --group app
RUN chown -R app:app /app /tmp/transformers /tmp/hf

USER app

# EXPOSITION DU PORT DYNAMIQUE
EXPOSE $PORT

# Pas de HEALTHCHECK dans le Dockerfile pour Cloud Run
# Cloud Run gère ses propres health checks

# COMMANDE CORRIGÉE pour Cloud Run
CMD exec uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1
