# apps/scheduler

Starter Python scheduler service.

Responsibilities:

- periodic ingestion triggers,
- scoring orchestration,
- retraining triggers,
- data-quality jobs,
- operational timing and workflow control.

Current status:

- this service now exposes:
  - `GET /health`
  - `GET /jobs` (registered plan)
  - `POST /jobs/run/{job_name}` (executes job orchestration),
- scheduler clients now trigger ingestion pipeline runs and AI region scoring over HTTP.
