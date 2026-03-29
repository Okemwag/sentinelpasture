FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

COPY apps/api apps/api
COPY apps/ai/data apps/ai/data

RUN pip install --upgrade pip && pip install -e apps/api

CMD ["python", "-m", "uvicorn", "api_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
