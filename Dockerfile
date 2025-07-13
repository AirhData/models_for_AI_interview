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

RUN addgroup --system app && adduser --system --group app

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY . .

# Create uploads directory with proper permissions
RUN mkdir uploads && chown -R app:app uploads

# Ensure /tmp is writable (it usually is by default, but let's be explicit)
RUN chmod 1777 /tmp

USER app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
