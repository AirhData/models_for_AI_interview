FROM python:3.10-slim

WORKDIR /app

ENV PORT=8000

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" && \
    python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='astrosbd/french_emotion_camembert')" && \
    python -c "from transformers import pipeline; pipeline('zero-shot-classification', model='joeddav/xlm-roberta-large-xnli')" && \
    python -c "import spacy; spacy.cli.download('fr_core_news_sm')"

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "$PORT"]
