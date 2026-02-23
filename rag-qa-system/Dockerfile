# Dockerfile  —  rag-qa-system
FROM python:3.11-slim

# Cleaner logs inside containers
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# ── Layer 1: dependencies (cached unless requirements.txt changes) ──
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Layer 2: application code ─────────────────────────────────────
COPY app/  ./app/
COPY data/ ./data/

# Run from inside app/ so imports resolve correctly
WORKDIR /app/app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
