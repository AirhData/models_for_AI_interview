FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

WORKDIR /app

COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY . .

ENV PORT=8080
ENV PYTHONPATH=/app
ENV TRANSFORMERS_CACHE=/tmp/transformers
ENV HF_HOME=/tmp/hf
ENV CREW_STORAGE_DIR=/tmp/crew
ENV HOME=/tmp
ENV TMPDIR=/tmp

RUN mkdir -p /tmp/transformers /tmp/hf /tmp/crew && \
    chmod 777 /tmp/transformers /tmp/hf /tmp/crew

RUN addgroup --system app && adduser --system --group app
RUN chown -R app:app /app /tmp/transformers /tmp/hf

USER app

EXPOSE $PORT

CMD exec uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1
