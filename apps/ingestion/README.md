# apps/ingestion

Starter Python ingestion service.

Responsibilities:

- external collectors,
- schema validation,
- deduplication,
- normalization into canonical events/features,
- persistence of raw and processed data.

Current status:

- this service now exposes:
  - `GET /health`
  - `GET /pipeline` (declared stages)
  - `POST /pipeline/run` (runnable orchestration for feature/label build and training artifact generation),
- collectors and persistence backends are still lightweight stubs around the current file-based AI data flow.
