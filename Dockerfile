# === ÉTAPE 1: BUILDER ===
# Cette étape installe les dépendances Python dans un environnement propre.
FROM python:3.11-slim AS builder

# Variables d'environnement pour l'installation des paquets
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

# Installation de 'uv', un gestionnaire de paquets Python plus rapide
RUN pip install uv

WORKDIR /app
COPY requirements.txt .

# Installation des dépendances du projet
RUN uv pip install --system --no-cache -r requirements.txt


# === ÉTAPE 2: IMAGE FINALE ===
# Cette étape construit l'image finale légère avec le code et les dépendances.
FROM python:3.11-slim

# Création d'un utilisateur et d'un groupe non-root pour la sécurité
RUN addgroup --system app && adduser --system --ingroup app app

# --- Variables d'environnement critiques pour Cloud Run ---
# HOME=/tmp : Force les écritures temporaires dans le répertoire autorisé.
# HF_HOME/TRANSFORMERS_CACHE : Spécifie où les modèles Hugging Face doivent être mis en cache.
ENV PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    HF_HOME=/tmp/huggingface_cache \
    TRANSFORMERS_CACHE=/tmp/huggingface_cache

# Copie des dépendances installées depuis l'étape 'builder'
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Crée le répertoire de cache avant le téléchargement des modèles
RUN mkdir -p /tmp/huggingface_cache

# --- Pré-téléchargement des modèles d'IA ---
# Cette couche est exécutée en tant que root mais sauvegarde les modèles au bon endroit
# grâce aux variables d'environnement.
# Exécuter en une seule commande RUN optimise la mise en cache de Docker.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" && \
    python -c "from transformers import pipeline; pipeline('text-classification', model='astrosbd/french_emotion_camembert')" && \
    python -c "from transformers import pipeline; pipeline('zero-shot-classification', model='joeddav/xlm-roberta-large-xnli')"

# Définit le répertoire de travail pour l'application
WORKDIR /app

# Copie le code de l'application (après l'installation et le DL pour optimiser le cache)
COPY --chown=app:app . .

# Crée le dossier 'uploads' et s'assure que tous les fichiers appartiennent à l'utilisateur 'app'
# Cela inclut le code, le dossier uploads, et le cache des modèles dans /tmp.
RUN mkdir /app/uploads && \
    chown -R app:app /app /tmp/huggingface_cache

# Bascule vers l'utilisateur non-root
USER app

EXPOSE 8000

# Commande pour lancer l'application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
