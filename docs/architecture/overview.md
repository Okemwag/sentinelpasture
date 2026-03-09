# Architecture Overview

The platform is a governance early-warning and coordination system.

Its purpose is to detect emerging instability before it becomes an acute crisis,
then help decision-makers choose proportional, auditable interventions.

For the implementation that is live on this branch today, see
[`current-state-end-to-end.md`](current-state-end-to-end.md).

## Core Product Loops

1. Ingest weak signals across domains.
2. Normalize them into a canonical governance ontology.
3. Score risk by region, corridor, or population segment.
4. Explain the score with drivers, confidence, and known data gaps.
5. Recommend bounded interventions.
6. Record outcomes and learn from what worked.

## Service Boundaries

`apps/api/`
Primary control-plane API. This is where RBAC, dashboard read models, reports,
audit trails, and explainability metadata are exposed. The preferred runtime is
FastAPI with shared Pydantic contracts.

`apps/ingestion/`
Data-plane ingestion service. This is where external feeds, manual uploads,
deduplication, normalization, and persistence happen.

`apps/feature_pipeline/`
Target feature-engineering boundary. This is where point-in-time safe feature
generation, validation, and reproducibility controls should converge as the data
layer matures.

`apps/scheduler/`
Job orchestration. This is where daily scoring, retraining triggers, and data
quality routines are managed.

`apps/ai/`
Modeling and inference core. This is where risk scoring, explanations, and
intervention ranking are trained and served. Early on, training and inference
can remain together here before being split into dedicated Python services.

`apps/web/`
User-facing dashboard and administrative UI.

## Operating Principle

The state should intervene early, proportionally, and with restraint. The system
exists to reduce lag between weak signal emergence and coordinated action.
