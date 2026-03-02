# Product Roadmap

This roadmap defines the phased path from the current prototype state to a
fully functional, Python-first governance platform.

## Target Architecture

Core platform stack:

- FastAPI for API and model-serving services
- SQLAlchemy plus Alembic for Postgres access and schema migrations
- Celery plus Redis for background jobs
- Prefect for scheduled data pipelines when orchestration outgrows app-local jobs
- Pydantic for service contracts and shared schemas
- LightGBM, XGBoost, and PyTorch for model development
- MLflow or BentoML for model registration and promotion
- Docker for all services, with Kubernetes introduced later only when needed

Target service boundaries:

- `apps/api/`: control-plane API for RBAC, dashboard reads, audit, and outcomes
- `apps/ingestion/`: source collection and normalization
- `apps/feature_pipeline/`: feature generation and data-quality validation
- `apps/ai_training/`: model training, evaluation, and registration
- `apps/ai_inference/`: online scoring, explanations, and intervention ranking
- `apps/scheduler/`: orchestration, cron triggers, and workflow coordination
- `apps/web/`: presentation layer and dashboard UI

The repository can remain a modular monolith early, but these boundaries define
the contracts the platform should converge toward.

## Phase 0: Governance Foundation (Weeks 1-2)

1. Define operating boundaries.
   Write the governance charter, prohibit surveillance and individual targeting,
   and define national versus county operating roles.

2. Lock the MVP geography and datasets.
   Choose the region unit and baseline data stack: ACLED, CHIRPS, MODIS NDVI,
   OSM, and optional market data.

Success criteria:

- stakeholders can explain what the platform is and is not,
- region-by-time tables can be produced consistently from the selected sources.

## Phase 1: Communication And Access Control (Weeks 2-5)

3. Launch the institutional landing page.
   Keep the message restrained, explicit, and easy to understand in under a
   minute.

4. Implement authentication, RBAC, and immutable audit logging.
   Every access and decision-changing action must be attributable.

Success criteria:

- the public-facing page explains scope and constraints clearly,
- access is enforceable and every sensitive action is logged.

## Phase 2: Backend Spine (Weeks 4-10)

5. Build the FastAPI backend.
   The minimum API surface includes:

- `/overview`
- `/regions/{id}/risk`
- `/regions/{id}/drivers`
- `/regions/{id}/interventions`
- `/outcomes`
- `/audit`
- `/exports`

6. Establish data storage and migrations.
   Define Postgres schemas for regions, features, model outputs, outcomes, and
   audit logs. Store raw files and parquet snapshots in object storage.

Success criteria:

- the frontend has stable, versioned contracts,
- migrations are reproducible and point-in-time data can be reconstructed.

## Phase 3: Ingestion And Feature Store (Weeks 8-14)

7. Build ingestion pipelines.
   Start with ACLED ingestion and region mapping, then CHIRPS, NDVI, and OSM
   corridor extraction.

8. Build the feature layer and data-quality checks.
   Generate region-by-time features with lags, rolling windows, and anomaly
   flags, and detect schema drift and missingness drift.

Success criteria:

- fresh data lands on schedule,
- feature generation is reproducible and silent failures are detectable.

## Phase 4: Core AI MVP (Weeks 12-20)

9. Deliver baseline predictive models.
   Lead with a LightGBM risk model using time-based splits and leakage controls.
   Add calibration and a simple anomaly detector.

10. Add explainability and model governance.
    Ship SHAP-based driver attribution, confidence bands, data-gap flags, and
    generated model cards.

11. Add intervention ranking.
    Start with a constrained rules-based ranking system, then add learned
    ranking only after the rules baseline is validated.

Success criteria:

- risk scores outperform incident-only baselines on lead time and precision at k,
- outputs are explainable enough for officials and auditors to defend.

## Phase 5: Dashboard (Weeks 16-24)

12. Build the governance UI.
    Deliver overview, regional risk, drivers, interventions, outcomes, and
    reporting views.

13. Wire frontend, API, and inference end to end.
    Daily scoring jobs should write outputs that the dashboard can read, with
    clear delayed-data states when sources lag.

Success criteria:

- users can identify top pressure zones quickly,
- the ingest-to-score-to-dashboard path runs reliably every day.

## Phase 6: Mobile Apps (Weeks 20-32)

14. Build the community signal app.
    Use structured forms, offline-first sync, and role-controlled reporter
    accounts without collecting unnecessary personal data.

15. Build the operations app.
    Support hotspot review, task execution, and outcome logging in the field.

Success criteria:

- field inputs sync reliably under weak connectivity,
- operational outcomes feed the learning loop.

## Phase 7: Pilot Deployment And Evaluation (Weeks 28-36)

16. Pilot in one or two regions.
    Run the platform alongside existing operations and establish weekly county
    and monthly national review rhythms.

17. Measure operational impact.
    Evaluate lead time gained, hotspot precision, false alarm cost, and
    intervention lift.

Success criteria:

- the platform changes planning decisions and response timing,
- pilot evidence is credible enough to justify scaling.

## Phase 8: Scale Upgrades (Months 9-12)

18. Introduce advanced models only after the baseline proves value.
    Candidates include STGNN for spillover effects, causal evaluation, and
    contextual bandits for intervention selection.

19. Prepare for national rollout.
    Add county isolation, HA, backup and DR coverage, and procurement-grade
    documentation.

Success criteria:

- model upgrades show measurable lift instead of complexity for its own sake,
- deployments are transferable across counties and other countries.

## Definition Of Fully Functional

The platform is fully functional when it can, on a reliable schedule:

- ingest and validate approved source data,
- generate point-in-time safe features,
- score regional risk with explainable outputs,
- recommend bounded interventions,
- record outcomes and audit events,
- present decision-ready views to authorized users,
- support pilot operations with measured impact.
