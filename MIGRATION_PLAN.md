# Migration Plan

This repository is transitioning from a prototype layout into a service-oriented
monorepo.

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

- implement `apps/api/` in Go,
- move dashboard read models, auth, audit, and reporting there,
- keep Python limited to model inference and training.

## Phase 4: Split Ingestion And Scheduling

- move ETL and collectors into `apps/ingestion/`,
- move jobs into `apps/scheduler/`,
- connect services through versioned contracts.

## Phase 5: Stabilize The AI Workspace

- migrate `ai-engine/` into `apps/ai/`,
- enforce strict inference contracts,
- add model registry, evaluation, and governance artifacts.
