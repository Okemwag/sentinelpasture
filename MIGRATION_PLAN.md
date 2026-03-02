# Migration Plan

This repository is transitioning from a prototype layout into a Python-first,
service-oriented monorepo.

## Phase 1: Define The Shape

- create `docs/`, `contracts/`, `infra/`, and `apps/`,
- document product scope, architecture, governance, and runbooks,
- keep the current Next.js and Python code runnable in place.

## Phase 2: Move The Web App

- move `app/`, `components/`, `lib/`, `public/`, and frontend config into
  `apps/web/`,
- update package scripts and workspace settings,
- preserve identical runtime behavior.

Status: completed.

## Phase 3: Replace The API Service

- implement `apps/api/` in FastAPI,
- move dashboard read models, auth, audit, and reporting there,
- standardize shared schemas and service contracts in Python.

Status: in progress.

## Phase 4: Split Ingestion, Features, And Scheduling

- move ETL and collectors into `apps/ingestion/`,
- extract reusable feature-building logic toward a dedicated pipeline boundary,
- move jobs into `apps/scheduler/`,
- connect services through versioned Python contracts.

## Phase 5: Stabilize The AI Workspace

- keep the AI workspace active in `apps/ai/` while the platform matures,
- enforce strict inference contracts,
- add model registry, evaluation, and governance artifacts,
- split training and inference services only when operationally necessary.

Status: in progress.
