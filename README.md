# Governance Intel Platform

This repository is being reshaped into a monorepo for a governance early-warning,
coordination, and learning system.

The product is designed to help public-sector teams act before instability
becomes expensive, violent, or politically destabilizing. It is not just an
incident dashboard. It is a decision-support platform that:

- fuses weak signals across markets, climate, mobility, services, and incidents,
- produces an explainable risk posture for regions and corridors,
- recommends proportional interventions with constraints,
- preserves auditability so decisions can be reviewed later,
- learns from outcomes to improve future interventions.

## What You Are Building

The platform is useful in scenarios such as:

- banditry escalation in frontier counties,
- election-period tension and localized unrest risk,
- drought to displacement to resource conflict,
- food security instability and market panic,
- urban crime spikes linked to economic stress,
- cross-border insecurity spillover,
- critical infrastructure disruption risk,
- public-service stress as a precursor to instability,
- post-incident stabilization and recurrence prevention,
- multi-hazard governance for floods, landslides, and epidemics.

## Target Monorepo

The target shape of this repository is:

- `docs/` for architecture, governance, runbooks, and product decisions.
- `contracts/` for shared API specs, schemas, and service contracts.
- `infra/` for deployment, Terraform, Kubernetes, Helm, and operational scripts.
- `apps/api/` for the FastAPI backend service.
- `apps/ingestion/` for the Python ingestion and normalization service.
- `apps/scheduler/` for the Python job orchestration service.
- `apps/ai/` for the Python modeling and inference workspace.
- `apps/web/` for the frontend dashboard and presentation layer.

## Current State

The currently runnable implementation is still in transition:

- the active Next.js frontend now lives in `apps/web/`;
- the active backend runtime now lives in `apps/api/`;
- the active AI inference runtime now lives in `apps/ai/`.

This is intentional during migration so the system stays runnable while the new
monorepo structure is established.

## Delivery Roadmap

The implementation plan is now tracked in
[`docs/product/roadmap.md`](docs/product/roadmap.md). It defines:

- phased delivery from governance foundations through pilot deployment,
- the MVP backend, data, AI, dashboard, and mobile milestones,
- the definition of "fully functional" for this platform.

For the implementation that is live on this branch today, see
[`docs/architecture/current-state-end-to-end.md`](docs/architecture/current-state-end-to-end.md).

## Run Locally

Frontend:

```bash
npm run dev
```

Backend:

```bash
./scripts/start-backend.sh
```

AI inference service:

```bash
./scripts/start-ai.sh
```

Ingestion service:

```bash
./scripts/start-ingestion.sh
```

Scheduler service:

```bash
./scripts/start-scheduler.sh
```

## Architecture Flow (Mermaid)

```mermaid
flowchart TD
    U[Analyst/User] --> FE[Frontend\nNext.js apps/web :3000\n- Sign-in\n- Dashboard pages\n- Reports download]

    FE -->|POST /auth/login\nGET /auth/me\nGET /api/*| API[Backend API\nFastAPI apps/api :8000\n- Auth + RBAC + Audit\n- Dashboard contract endpoints\n- AI gateway orchestration]

    API --> AUTH[Auth service\n- JWT bearer validation\n- Role checks: viewer/analyst/operator/admin]
    API --> AUDIT[(Audit DB table\nsqlite: governance_intel.db\nrecords action/resource/outcome)]
    API --> GW[AIGateway\n- Calls remote AI /infer/*\n- Optional mock fallback if AI unreachable]

    GW -->|HTTP| AI[AI Inference Service\nFastAPI apps/ai :8100\n/infer/risk\n/infer/explain\n/infer/interventions\n/infer/regions]

    AI --> ML[Model Loader\nreads data/artifacts/baseline_risk_model.json]
    AI --> FF[Feature Fetcher\nreads latest processed CSV snapshots\n- national/subnational]
    AI --> SCORE[Scoring + Explanation + Intervention Builder\nresponse_builder.py]

    RAW[(Raw Data Files\napps/ai/data/raw\n- ken-rainfall-subnat-full.csv\n- ACLED monthly event CSVs)] --> BF[Feature Builder\nbuild_features.py\naggregates monthly rainfall features]
    RAW --> BL[Label Builder\nbuild_labels.py\naggregates monthly event labels]
    BF --> PROC[(Processed Data\napps/ai/data/processed\nrainfall_features_monthly_*.csv)]
    BL --> PROC2[(Processed Labels\nevent_labels_monthly_national.csv)]

    PROC --> TRAIN[Trainer\ntrain_lgbm.py baseline pipeline]
    PROC2 --> TRAIN
    TRAIN --> ART[(Model Artifact\napps/ai/data/artifacts/\nbaseline_risk_model.json)]
    ART --> ML
    PROC --> FF

    ING[Ingestion service apps/ingestion\nCurrent state: runnable orchestration\n- /health\n- /pipeline\n- /pipeline/run\nvalidates raw files\nbuilds features/labels\ntrains baseline artifact] -. current file-based path .-> RAW
    SCH[Scheduler service apps/scheduler\nCurrent state: trigger facade\n- /health\n- /jobs\n- /jobs/run/{job_name}\ninvokes ingestion and scoring over HTTP] -. current automation hooks .-> BF
    SCH -. planned automation .-> BL
    SCH -. planned automation .-> TRAIN
```

```mermaid
flowchart LR
    A[AI algorithm now in code\n(statistical baseline, not full LightGBM runtime)] --> B[Training target construction\nfor each month:\ntarget_score = total_events + min(total_fatalities,100)/10]
    B --> C[Feature weighting\nsigned correlation(feature, normalized target)\nabs-normalized to weights]
    C --> D[Inference normalization\nper feature:\nnormalized = clamp((x-min)/(max-min),0,1)]
    D --> E[Risk score\nraw = sum(normalized_i * weight_i)\nrisk_score = 0.05 + 0.9*clamp(raw,0,1)]
    E --> F[Confidence\nfrom observation count metadata\nband low/medium/high by thresholds]
    F --> G[Top drivers\nlargest absolute contributions]
    G --> H[Explanations\ntext summary + uncertainty notes]
    G --> I[Interventions\nrule-based mapping from top drivers\n+ policy constraints]
```

### Ingestion Engine Status

- `apps/ingestion` exposes `GET /health`, `GET /pipeline`, and `POST /pipeline/run`.
- The active pipeline validates raw CSV inputs, then calls the AI feature builders, label builders, and baseline trainer.
- Declared stages remain `collect -> validate -> deduplicate -> normalize -> persist_raw -> persist_processed`, but collector, dedup, normalizer, and storage modules are still lightweight stubs.
- Effective data path today is: raw CSVs in `apps/ai/data/raw` -> builders/trainer in `apps/ai` -> inference/API/frontend.
