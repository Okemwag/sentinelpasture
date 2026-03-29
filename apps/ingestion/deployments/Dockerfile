FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

COPY apps/ingestion apps/ingestion
COPY apps/ai apps/ai

RUN pip install --upgrade pip && \
    pip install -e apps/ai && \
    pip install -e apps/ingestion

CMD ["python", "-m", "uvicorn", "ingestion_service.main:app", "--host", "0.0.0.0", "--port", "8300"]
