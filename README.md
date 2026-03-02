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
- `apps/api/` for the Go API service.
- `apps/ingestion/` for the Go ingestion and normalization service.
- `apps/scheduler/` for the Go job orchestration service.
- `apps/ai/` for the Python modeling and inference workspace.
- `apps/web/` for the frontend dashboard and presentation layer.

## Current State

The currently runnable implementation is still in transition:

- the active Next.js frontend now lives in `apps/web/`;
- the active Python backend is still in `backend/`;
- the existing `ai-engine/` directory is still experimental and not yet the
  production inference path.

This is intentional during migration so the system stays runnable while the new
monorepo structure is established.

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
