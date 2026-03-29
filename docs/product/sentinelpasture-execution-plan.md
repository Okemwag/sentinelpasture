# SentinelPasture Execution Plan

This document converts the SentinelPasture product definition into an
engineering program that can be executed against the current repository.

It is intentionally pragmatic:

- "100% complete" is not a meaningful engineering state;
- the useful target is operational completeness with measurable promotion gates;
- the current repository is a strong prototype, not a finished national
  intelligence platform.

## 1. Product Completion Standard

SentinelPasture should be considered operationally complete only when all of
the following are true:

1. Approved source data is ingested on schedule without manual file drops.
2. Region-by-time tables are reproducible and point-in-time safe.
3. Risk scoring is backed by validated baseline and advanced models with
   time-based evaluation.
4. Every model output carries explanations, confidence, and data-gap metadata.
5. Intervention recommendations are policy-constrained, reviewable, and linked
   to observed outcomes.
6. Dashboard workflows are usable by authorized national and county operators.
7. Audit, access control, and data-governance controls are production-grade.
8. Retraining, promotion, rollback, and incident procedures are defined and
   tested.

## 2. Current Reality

The current repository already provides:

- a Next.js web application and authenticated dashboard;
- a FastAPI API service with JWT auth, RBAC, and audit logging;
- a FastAPI AI inference service serving risk, explanation, intervention, and
  region-list endpoints;
- an ingestion service that validates raw CSV inputs and rebuilds processed
  features, labels, and the model artifact;
- a scheduler facade that triggers ingestion and scoring jobs over HTTP.

The current repository does not yet provide:

- production ingestion from remote source systems;
- durable Postgres or PostGIS-backed operational storage in the active path;
- a true feature store;
- production LightGBM, XGBoost, or forecasting runtime;
- SHAP-based explainability in the live path;
- model registry, promotion workflows, or calibrated evaluation reports;
- a mature intervention learning loop tied to logged field outcomes;
- a persistent workflow engine for scheduled jobs.

## 3. Gap By Product Module

### 3.1 Data Integration

Current state:

- file-based ingestion into `apps/ai/data/raw`;
- collectors mostly exist as stubs;
- canonical region mapping is incomplete in the active path.

Required completion work:

- implement collectors for ACLED, CHIRPS, MODIS NDVI, OSM corridor features,
  and approved market feeds;
- standardize region identity on county-level canonical keys first;
- persist raw snapshots, normalized tables, and processed feature tables into
  Postgres plus object storage;
- add schema-drift, freshness, and completeness checks.

### 3.2 Risk Engine

Current state:

- correlation-weighted baseline artifact marketed as an LGBM baseline;
- rainfall features plus event-derived labels drive the live scoring path.

Required completion work:

- implement true baseline tabular models: Logistic Regression, Random Forest,
  LightGBM, and XGBoost;
- add regional and national target definitions for conflict, protest,
  displacement, food insecurity, and flood impact where data supports them;
- support region-specific thresholds and calibrated probability outputs.

### 3.3 Forecasting Engine

Current state:

- short-horizon prediction is a shaped projection of current risk, not true
  forecasting.

Required completion work:

- build time-series forecasting for lead windows such as 7, 14, 30, and 90
  days;
- add backtesting with time-based splits and leakage controls;
- compare tree-based lag models against explicit time-series methods before
  adding more complex architectures.

### 3.4 Explainability Engine

Current state:

- contribution ranking from a weighted linear baseline;
- text summaries are template-based.

Required completion work:

- add SHAP explanations for promoted tabular models;
- show feature values versus baselines and recent trend movement;
- expose uncertainty, missingness, stale-data flags, and unsupported-regime
  warnings in every response.

### 3.5 Intervention Engine

Current state:

- rule-based mapping from top drivers to intervention options.

Required completion work:

- define a policy-constrained intervention library with owners, cost bands, and
  lead times;
- add ranking based on historical outcomes and context similarity;
- keep human review mandatory and disallow automated coercive recommendations.

### 3.6 Dashboard

Current state:

- overview, alerts, drivers, regional risk, interventions, outcomes, and
  reports pages exist;
- some views still depend on placeholder or synthetic backend outputs.

Required completion work:

- add a dedicated forecast page and richer region drill-down;
- connect outcomes to real intervention logging rather than synthetic deltas;
- expose data freshness, model versioning, and delayed-source states visibly.

### 3.7 Outcome Engine

Current state:

- outcome views are mostly placeholders.

Required completion work:

- define intervention logging schema and workflow;
- capture pre-intervention baseline, deployed action, owner, scope, and timing;
- measure lift, recurrence reduction, false-alarm cost, and follow-through
  rates;
- route outcome data back into training and evaluation.

## 4. AI Training Program

The AI program should progress in disciplined stages.

### Stage A: Data Readiness

Deliverables:

- canonical county-by-time training tables;
- source freshness checks;
- feature documentation and leakage review;
- reproducible dataset versioning.

Exit criteria:

- training sets can be rebuilt from source snapshots without manual edits;
- every feature has owner, definition, lag policy, and missingness behavior.

### Stage B: Baseline Models

Deliverables:

- anomaly detection: z-score baselines first, then Isolation Forest;
- supervised baselines: Logistic Regression, Random Forest, LightGBM, XGBoost;
- benchmark reports for each target and lead horizon.

Evaluation requirements:

- time-based train, validation, and test splits;
- precision at k, recall at k, lead-time gain, calibration, false-alarm cost,
  and geographic stability checks;
- comparisons against naive historical baselines.

Promotion rule:

- no model is promoted unless it beats the current baseline on lead time and
  precision while remaining interpretable enough for operational review.

### Stage C: Explainability And Governance

Deliverables:

- SHAP attribution service for promoted tree models;
- model cards, data cards, and threshold rationale;
- bias and drift checks across counties and major operating contexts.

Promotion rule:

- no model is production-eligible without explanation metadata, drift checks,
  and rollback procedures.

### Stage D: Forecasting

Deliverables:

- lead-horizon forecasts for selected targets;
- scenario comparison against "no action" baselines;
- confidence intervals and uncertainty reporting.

Candidate models:

- lagged gradient-boosted models first;
- explicit time-series models second;
- graph and sequence models only after baseline evidence justifies them.

### Stage E: Intervention Learning

Deliverables:

- historical intervention-outcome table;
- retrieval or ranking model for intervention selection;
- constrained ranking that respects policy, cost, geography, and response
  capacity.

Promotion rule:

- intervention ranking remains human-reviewed until validated against real
  operational outcomes.

## 5. Platform Architecture Target

The recommended production target is:

- `apps/api`: FastAPI control-plane API;
- `apps/ingestion`: source collection and normalization;
- `apps/feature_pipeline`: feature generation and data quality;
- `apps/ai_training`: training, evaluation, and registration;
- `apps/ai_inference`: online scoring, explanations, and interventions;
- `apps/scheduler`: orchestration;
- `apps/web`: dashboard and presentation layer;
- Postgres plus PostGIS for operational storage;
- Redis for queues and caching;
- Airflow or Prefect for scheduled workflows;
- object storage for raw snapshots, model artifacts, and exports.

The current repository does not need to split every service immediately, but it
should evolve toward these boundaries.

## 6. Phased Delivery

### Phase 0: Product And Contract Alignment

- align naming across repo, UI, and documentation to SentinelPasture;
- freeze initial geography, target types, and source approvals;
- define API contracts for risk, forecast, interventions, outcomes, and audit.

### Phase 1: Data Foundation

- implement remote collectors;
- stand up Postgres plus PostGIS schemas;
- build canonical region mapping and reproducible source snapshots.

### Phase 2: Feature And Label Layer

- implement point-in-time safe feature generation;
- add lagged, rolling, anomaly, and seasonal features;
- create target tables for the approved risk types.

### Phase 3: Baseline Training And Registry

- train benchmark models;
- build evaluation reports and model cards;
- introduce model registry and promotion gates.

### Phase 4: Production Inference And Explainability

- serve promoted models through a stable inference contract;
- attach SHAP, confidence, and data-gap metadata;
- add caching, latency budgets, and service-level health checks.

### Phase 5: Intervention And Outcomes Loop

- implement intervention logging and review workflows;
- connect outcome tracking to model evaluation;
- improve ranking with validated operational evidence.

### Phase 6: Operational Hardening

- add workflow persistence, retries, and job history;
- implement secret management, SSO options, and service-to-service auth;
- add observability, incident runbooks, and backup plus disaster recovery.

## 7. Near-Term Build Order For This Repository

The highest-leverage next engineering moves are:

1. Align product naming and API contracts.
2. Replace file-only active storage with Postgres-backed normalized tables while
   keeping CSV rebuilds for reproducibility.
3. Build the first true baseline LightGBM training and evaluation pipeline.
4. Add explicit forecast endpoints and a forecast page in the dashboard.
5. Replace placeholder outcomes with a real intervention log and outcome schema.

## 8. Definition Of Done For The AI

The AI is not "done" when a model exists.

The AI is done only when:

- data arrives reliably;
- models are validated on time-based tests;
- predictions are explainable and governed;
- interventions are constrained and reviewable;
- outcomes are measured and fed back into learning;
- operations teams trust the system enough to use it repeatedly.
