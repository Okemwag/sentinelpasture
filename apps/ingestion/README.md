# apps/ingestion

Starter Python ingestion service.

Responsibilities:

- external collectors,
- schema validation,
- deduplication,
- normalization into canonical events/features,
- persistence of raw and processed data.

Current status:

- this directory now contains a minimal Python ingestion service scaffold,
- it exposes a health endpoint and pipeline description,
- collector and persistence implementations still need to be filled in.
