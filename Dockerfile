# Étape 1 : Builder - Installe les dépendances Python
FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

# Installe uv, un installateur de paquets rapide
RUN pip install uv

WORKDIR /app

# Copie uniquement requirements.txt pour profiter du cache Docker
COPY requirements.txt .

# Installe les dépendances dans le système
RUN uv pip install --system --no-cache -r requirements.txt


# ---


# Étape 2 : Image finale - Configure l'application
FROM python:3.11-slim

# Crée un groupe et un utilisateur non-root pour l'application
RUN addgroup --system app && adduser --system --ingroup app app

# Définit les variables d'environnement.
# HF_HOME est la variable standard pour le cache de Hugging Face.
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/home/app/.cache/huggingface \
    HOME=/home/app

# Copie les paquets installés depuis l'étape de build
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# === Pré-téléchargement des modèles ===
# Cette étape est exécutée en tant que root, mais les modèles sont
# sauvegardés directement au bon endroit grâce à la variable HF_HOME.
# On exécute tout dans une seule commande RUN pour optimiser les couches Docker.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" && \
    python -c "from transformers import pipeline; pipeline('text-classification', model='astrosbd/french_emotion_camembert')" && \
    python -c "from transformers import pipeline; pipeline('zero-shot-classification', model='joeddav/xlm-roberta-large-xnli')"

# Définit le répertoire de travail
WORKDIR ${HOME}/app

# Copie le code de l'application. Cette étape est placée après le
# téléchargement des modèles pour optimiser le cache.
COPY --chown=app:app . .

# Crée le dossier pour les uploads
RUN mkdir uploads

# Change le propriétaire de l'ensemble du répertoire HOME et des uploads
# pour l'utilisateur 'app'. Cela inclut le code et le cache des modèles.
RUN chown -R app:app ${HOME} /app/uploads

# Passe à l'utilisateur non-root
USER app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
