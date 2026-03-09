# Current State End-to-End

This document describes what is implemented in this repository today, how the
system works end to end, what algorithms are actually in use, and how to run
or trigger each stage.

It separates:

- current implemented functionality,
- current code footprint,
- planned-but-not-yet-live architecture from the roadmap.

## 1. Functional Distribution

Two different percentage views are useful here.

### A. Core runtime footprint by service

Method: percentage of executable source lines in the active Python services
only (`api`, `ai`, `ingestion`, `scheduler`).

| Service | Source lines | Share of core runtime code |
| --- | ---: | ---: |
| `apps/api` | 1,598 | 46.1% |
| `apps/ai` | 1,172 | 33.8% |
| `apps/ingestion` | 383 | 11.0% |
| `apps/scheduler` | 316 | 9.1% |

Interpretation:

- most of the current backend functionality sits in `api` and `ai`;
- ingestion and scheduling exist, but are thinner orchestration layers around
  the current file-based AI pipeline.

### B. Whole product footprint including frontend

Method: percentage of executable source lines in the active application
directories.

| Surface | Source lines | Share of app code |
| --- | ---: | ---: |
| `apps/web` | 6,031 | 63.5% |
| `apps/api` | 1,598 | 16.8% |
| `apps/ai` | 1,172 | 12.3% |
| `apps/ingestion` | 383 | 4.0% |
| `apps/scheduler` | 316 | 3.3% |

Interpretation:

- the largest code footprint is the web/dashboard layer;
- the operational intelligence flow is still driven mainly by `api` plus `ai`;
- ingestion and scheduler are present but not yet the dominant implementation
  centers.

### C. Estimated implementation maturity by domain

This is a current-state estimate based on the code, not a roadmap promise.

| Domain | Estimated maturity | Why |
| --- | ---: | --- |
| Web | 75% | Many pages, dashboard views, auth gate, and API integration are live, but some views still depend on placeholder backend data. |
| API / backend | 70% | Auth, RBAC, audit logging, dashboard endpoints, and AI gateway logic are implemented; migrations, deeper admin/reporting workflows, and durable data models are still thin. |
| AI | 60% | Feature building, label building, training artifact generation, inference, explanations, and intervention ranking are live; true LightGBM runtime, evaluation, registry, and advanced models are not. |
| Ingestion | 40% | There is a runnable orchestration endpoint, validation, and pipeline execution into AI builders/trainers; collectors, normalizers, dedup, and persistence backends are mostly stubs. |
| Scheduler | 35% | Job definitions and job-trigger endpoints exist; there is no full persistent job runner, queue, or workflow state management. |

## 2. What Actually Runs Today

The active platform path is:

1. `apps/web` renders the UI and calls the backend.
2. `apps/api` authenticates users, applies RBAC, logs audit events, and shapes
   dashboard responses.
3. `apps/api` calls `apps/ai` over HTTP for risk, explanation, intervention,
   and region-list responses.
4. `apps/ai` reads processed CSV snapshots and a JSON model artifact from
   `apps/ai/data/`.
5. `apps/ingestion` can rebuild those processed CSVs and the model artifact by
   invoking the AI dataset builders and trainer.
6. `apps/scheduler` can trigger ingestion or scoring jobs over HTTP.

The important current constraint is this:

- the live data path is still file-based;
- raw CSVs are expected in `apps/ai/data/raw`;
- processed feature tables are written to `apps/ai/data/processed`;
- the serving model artifact is loaded from
  `apps/ai/data/artifacts/baseline_risk_model.json`.

## 3. End-to-End Flow

### Step 1: Raw data lands

Current raw inputs are CSV files placed in `apps/ai/data/raw`.

The live pipeline expects at least:

- `ken-rainfall-subnat-full.csv`
- `kenya_demonstration_events_by_month-year_as-of-25feb2026.csv`
- `kenya_civilian_targeting_events_and_fatalities_by_month-year_as-of-25feb2026.csv`
- `kenya_political_violence_events_and_fatalities_by_month-year_as-of-25feb2026.csv`

Today, ingestion does not fetch those sources from remote providers in the
runtime path. The collector modules exist, but they are stubs.

### Step 2: Ingestion validates and orchestrates

`POST /pipeline/run` on `apps/ingestion` orchestrates the current pipeline.

Modes:

- `full`: validate raw files, build features, build labels, train artifact
- `features`: validate raw files, build features only
- `train`: train from already-built processed files

The ingestion runtime currently performs:

- raw file discovery,
- required-file validation,
- row counting,
- delegation to AI feature builders and training modules.

It does not yet perform live external collection, canonical normalization,
durable raw persistence, or processed database writes in the active path.

### Step 3: Feature engineering runs

`ai.datasets.builders.build_features` reads
`ken-rainfall-subnat-full.csv` and produces:

- `rainfall_features_monthly_national.csv`
- `rainfall_features_monthly_subnational.csv`

The feature builder aggregates by month and computes:

- `rfh_mean`
- `rfh_avg_mean`
- `rfh_anomaly = rfh_mean - rfh_avg_mean`
- `r1h_mean`
- `r3h_mean`
- `rfq_mean`
- observation counts

This is a deterministic monthly aggregation pipeline, not a learned feature
store yet.

### Step 4: Label building runs

`ai.datasets.builders.build_labels` reads the three ACLED-derived monthly CSVs
and produces `event_labels_monthly_national.csv`.

It calculates:

- demonstrations events,
- civilian targeting events and fatalities,
- political violence events and fatalities,
- `total_events`,
- `total_fatalities`.

### Step 5: Training artifact is produced

`ai.training.train_lgbm` is the active trainer, but it is not running a real
LightGBM model yet.

What it actually does:

1. Join monthly features and monthly labels on `period`.
2. Build a scalar target:
   `target_score = total_events + min(total_fatalities, 100) / 10`
3. Compute low and high risk cutoffs using tertiles.
4. Compute a signed correlation between each feature and the normalized target.
5. Convert absolute correlations into normalized feature weights.
6. Persist a JSON artifact to
   `apps/ai/data/artifacts/baseline_risk_model.json`.

As of March 5, 2026, the checked-in artifact reports:

- `model_version = baseline-risk-model-v1`
- `training_rows = 350`
- `source_period = 1997-01` through `2026-02`

So the active algorithm is best described as:

- a correlation-weighted statistical baseline,
- not a trained LightGBM tree ensemble,
- not TFT,
- not STGNN,
- not an anomaly autoencoder.

Those advanced model files exist mostly as placeholders for later work.

### Step 6: Online inference runs

`apps/ai` serves:

- `POST /infer/risk`
- `POST /infer/explain`
- `POST /infer/interventions`
- `GET /infer/regions`
- `GET /health`

For inference, the AI service:

1. Loads `baseline_risk_model.json`.
2. Loads the latest feature snapshot for the requested region from processed
   CSV files.
3. For each feature, applies min-max normalization using training-time
   `feature_min` and `feature_max`.
4. Multiplies normalized values by the learned statistical weights.
5. Sums those contributions into a bounded raw signal.
6. Converts that raw signal to:
   `risk_score = 0.05 + 0.9 * clamp(raw_signal, 0, 1)`
7. Derives top drivers from the largest absolute feature contributions.
8. Derives confidence from observation-count heuristics.
9. Adds known data-gap messages for thin observation windows or unstable region
   mapping.

If `/infer/risk` receives ad hoc `signals`, it applies a small additive score
adjustment based on signal count, capped at `0.08`.

### Step 7: Explanation and interventions are built

Explainability is currently contribution-based:

- top drivers are the features with the largest weighted contributions;
- explanation text is generated from those drivers and metadata;
- uncertainty notes are rule-based text, not SHAP output.

Intervention ranking is currently rule-based:

- the top drivers are mapped to categories such as climate response,
  contingency planning, or human analyst review;
- output includes `expected_impact`, `time_to_effect`, and
  `constraints_applied`;
- policy constraints are declarative rules, not learned optimization.

### Step 8: Backend API shapes the product response

`apps/api` is the active control-plane backend.

It currently provides:

- auth login and current-user endpoints,
- JWT validation,
- role-based access checks,
- audit logging into SQLite,
- dashboard-ready endpoints,
- report downloads,
- AI orchestration through `AIGateway`.

The backend uses `apps/ai` as the real inference source when
`AI_INFERENCE_URL` is configured. If the remote AI service is unavailable and
fallback is enabled, it can return mock responses instead.

This means the backend is functional, but some product behavior still depends
on:

- mock fallback logic,
- generated placeholder report exports,
- synthetic outcome summaries.

### Step 9: Web dashboard consumes backend APIs

`apps/web` talks to `apps/api`, not directly to model internals.

The main dashboard reads:

- stability index,
- stats,
- drivers,
- regional data,
- regional map,
- alerts,
- interventions,
- outcomes,
- reports.

The UI path is therefore:

`browser -> apps/web -> apps/api -> apps/ai -> processed CSV + model artifact`

### Step 10: Scheduler triggers orchestration

`apps/scheduler` exposes:

- `GET /health`
- `GET /jobs`
- `POST /jobs/run/{job_name}`

Registered jobs:

- `ingest_daily`
- `compute_features_daily`
- `run_scoring_daily`
- `retrain_monthly`
- `data_quality_reports`

Current behavior:

- scheduler triggers ingestion or scoring over HTTP;
- it does not yet own a persistent cron engine, queue, worker fleet, or job
  history store;
- it is an orchestration facade, not a full workflow system yet.

## 4. Algorithms In Use Now

This is the current live algorithm set.

### Ingestion / preprocessing

- required-field validation helper
- exact dedup key helper
- semantic-key helper
- monthly CSV aggregation

Important caveat:

- validation and dedup helpers exist, but they are not the main active runtime
  path for the current ingestion orchestration;
- external collectors, canonical mapping, geo resolution, and storage backends
  are mostly stubbed.

### Feature engineering

- deterministic monthly averaging
- anomaly calculation by subtraction from baseline rainfall average
- national and subnational aggregation

### Labeling

- monthly event-count and fatality-count aggregation

### Training

- scalar target construction:
  `total_events + min(total_fatalities, 100) / 10`
- tertile thresholding for risk bands
- signed Pearson-style correlation against normalized target
- normalized absolute feature weighting

### Online scoring

- min-max feature normalization
- weighted linear aggregation
- bounded score transform into `0.05` to `0.95`

### Confidence

- observation-count heuristic

### Explainability

- top absolute contribution ranking
- template-driven narrative explanation

### Intervention ranking

- rule-based driver-to-intervention mapping
- fixed policy constraints

## 5. What Is Planned But Not Live Yet

The roadmap points to a richer future stack, but the following items are not
the active runtime on this branch today:

- real LightGBM training and serving
- SHAP-based explainability
- calibrated model evaluation outputs
- ML registry / promotion flow
- TFT, STGNN, and anomaly autoencoder production training
- durable feature store
- external-source ingestion collectors in the live path
- processed database writes as the primary serving source
- persistent scheduler / worker execution model

## 6. How To Run It End To End

### Start services

From the repo root:

```bash
npm run dev
./scripts/start-backend.sh
./scripts/start-ai.sh
./scripts/start-ingestion.sh
./scripts/start-scheduler.sh
```

Ports:

- web: `3000`
- api: `8000`
- ai: `8100`
- scheduler: `8200`
- ingestion: `8300`

### Rebuild processed data and model artifact

Run the ingestion pipeline directly:

```bash
curl -X POST http://localhost:8300/pipeline/run \
  -H 'Content-Type: application/json' \
  -d '{"mode":"full","train":true}'
```

That will:

- validate raw CSV presence,
- rebuild processed feature tables,
- rebuild processed labels,
- retrain and rewrite `baseline_risk_model.json`.

### Rebuild features only

```bash
curl -X POST http://localhost:8300/pipeline/run \
  -H 'Content-Type: application/json' \
  -d '{"mode":"features","train":false}'
```

### Retrain only

```bash
curl -X POST http://localhost:8300/pipeline/run \
  -H 'Content-Type: application/json' \
  -d '{"mode":"train","train":true}'
```

### Trigger through the scheduler facade

Examples:

```bash
curl http://localhost:8200/jobs
curl -X POST http://localhost:8200/jobs/run/ingest_daily
curl -X POST http://localhost:8200/jobs/run/run_scoring_daily
curl -X POST http://localhost:8200/jobs/run/retrain_monthly
```

### Call AI inference directly

Risk:

```bash
curl -X POST http://localhost:8100/infer/risk \
  -H 'Content-Type: application/json' \
  -d '{"region_id":"national","at_time":"2026-03-09T00:00:00Z","signals":[]}'
```

Explanation:

```bash
curl -X POST http://localhost:8100/infer/explain \
  -H 'Content-Type: application/json' \
  -d '{"region_id":"national","risk_score":0.6}'
```

Interventions:

```bash
curl -X POST http://localhost:8100/infer/interventions \
  -H 'Content-Type: application/json' \
  -d '{"region_id":"national","risk_score":0.6}'
```

Regions list:

```bash
curl 'http://localhost:8100/infer/regions?limit=20'
```

### Call the backend product API

Login first:

```bash
curl -X POST http://localhost:8000/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"admin","password":"admin123!"}'
```

Then use the returned bearer token with endpoints such as:

```bash
curl http://localhost:8000/api/dashboard/stability \
  -H 'Authorization: Bearer <token>'
```

Other useful endpoints:

- `GET /api/dashboard/stats`
- `GET /api/drivers/list`
- `GET /api/regional/data`
- `GET /api/regional/map`
- `GET /api/alerts/list`
- `GET /api/interventions/list`
- `GET /api/outcomes/list`
- `GET /api/reports/list`
- `GET /api/audit` for admin users

## 7. Short Answer

If the question is "where does most of the actual working functionality live
today?", the answer is:

- product UI: `apps/web`
- control-plane behavior: `apps/api`
- scoring logic and model behavior: `apps/ai`
- pipeline automation around that scoring logic: light but real in
  `apps/ingestion` and `apps/scheduler`

If the question is "what algorithm is live today?", the answer is:

- a correlation-weighted statistical baseline over engineered rainfall
  features, with rule-based explainability and rule-based intervention ranking,
  not a production LightGBM or deep-learning model yet.
