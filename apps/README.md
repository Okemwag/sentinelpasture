# Applications

Target service layout:

- `api/`: Python FastAPI control-plane service.
- `ingestion/`: Python ingestion and normalization service.
- `scheduler/`: Python job scheduling and orchestration service.
- `ai/`: Python AI training and inference workspace.
- `web/`: frontend dashboard and presentation layer.

The frontend has already been migrated into `apps/web/`.
The backend runtime now lives in `apps/api/`.
The AI workspace lives in `apps/ai/` and still needs deeper implementation.
