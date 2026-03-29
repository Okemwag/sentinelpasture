FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

COPY apps/scheduler apps/scheduler

RUN pip install --upgrade pip && pip install -e apps/scheduler

CMD ["python", "-m", "uvicorn", "scheduler_service.main:app", "--host", "0.0.0.0", "--port", "8200"]
