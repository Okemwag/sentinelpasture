# Data Flow

1. Collectors and uploads enter through `apps/ingestion/`.
2. Raw source records are validated, deduplicated, and normalized into canonical
   events and features.
3. Processed features are stored and versioned.
4. `apps/scheduler/` triggers scoring and retraining jobs on defined cadences.
5. `apps/ai/` serves risk, explanation, and intervention outputs over strict
   service contracts.
6. `apps/api/` consumes those outputs, applies product policy and RBAC, and
   exposes dashboard-ready read models.
7. `apps/web/` renders posture maps, interventions, and outcome tracking.
8. Outcomes feed back into retraining, evaluation, and governance review.
