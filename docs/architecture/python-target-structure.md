# Python-First Target Structure

This document captures the intended long-term repository shape for a
Python-first platform, while allowing the current implementation to remain
incrementally migratable.

## Target Folder Tree

```text
platform/
  apps/
    api/
      src/
        api_service/
          main.py
          auth.py
          dashboard.py
          outcomes.py
          audit.py
      pyproject.toml
      requirements.lock
    ingestion/
      src/
        ingestion_service/
          main.py
          pipeline.py
          collectors/
          normalizers/
          storage/
      pyproject.toml
      requirements.lock
    feature_pipeline/
      src/
        feature_pipeline/
          main.py
          builders/
          validation/
          registry/
      pyproject.toml
      requirements.lock
    ai_training/
      src/
        ai_training/
          main.py
          datasets/
          training/
          evaluation/
          registry/
          governance/
      pyproject.toml
      requirements.lock
    ai_inference/
      src/
        ai_inference/
          main.py
          endpoints/
          runtime/
          explainability/
          interventions/
      pyproject.toml
      requirements.lock
    scheduler/
      src/
        scheduler_service/
          main.py
          jobs.py
          workflows/
      pyproject.toml
      requirements.lock
    web/
  shared/
    schemas/
    auth/
    observability/
    utils/
  contracts/
  docs/
  infra/
  tests/
```

## Mapping From Current Repository

The current repository already partially reflects this shape:

- `apps/api/` is the active FastAPI backend
- `apps/ingestion/` is the ingestion scaffold
- `apps/scheduler/` is the job orchestration scaffold
- `apps/ai/` currently combines training and inference concerns
- `apps/web/` is the active frontend

The next logical structural split is inside `apps/ai/`, where training and
serving can eventually be separated into distinct services without changing the
governance model or shared schemas.

## Design Rules

- Prefer shared Pydantic schemas over ad hoc payload dicts.
- Keep API contracts explicit and versioned before splitting services further.
- Isolate model training dependencies from online inference dependencies.
- Treat feature generation as a reproducible pipeline, not request-time logic.
- Keep audit, auth, and observability primitives reusable across services.
