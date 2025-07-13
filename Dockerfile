FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

RUN pip install uv

WORKDIR /app

COPY requirements.txt .

RUN uv pip install --system --no-cache -r requirements.txt


# --------------------------------------
# Runtime image
# --------------------------------------
FROM python:3.11-slim

RUN addgroup --system app && adduser --system --group app

WORKDIR /app

# Copie des dépendances Python installées
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Création du dossier pour les modèles
RUN mkdir -p /app/models

# Pré-téléchargement des modèles pour éviter le warm-up à chaque requête
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2', cache_folder='/app/models')" && \
    python -c "from transformers import pipeline; pipeline('text-classification', model='astrosbd/french_emotion_camembert', return_all_scores=True, cache_dir='/app/models')" && \
    python -c "from transformers import pipeline; pipeline('zero-shot-classification', model='joeddav/xlm-roberta-large-xnli', cache_dir='/app/models')"

# Copie du code de l'application
COPY . .

# Création du dossier d’uploads si utilisé
RUN mkdir -p /app/uploads && chown -R app:app /app/uploads /app/models

# Utilisateur non root
USER app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
