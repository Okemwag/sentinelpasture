# apps/scheduler

Starter Python scheduler service.

Responsibilities:

- periodic ingestion triggers,
- scoring orchestration,
- retraining triggers,
- data-quality jobs,
- operational timing and workflow control.

Current status:

- this directory now contains a minimal Python service scaffold,
- it exposes a health endpoint and registered job list,
- the next step is wiring the scheduler to ingestion, API, and AI service calls.
