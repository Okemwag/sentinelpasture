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
