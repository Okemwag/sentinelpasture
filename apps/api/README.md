# apps/api

Starter Go API service.

Responsibilities:

- auth and RBAC,
- dashboard read models,
- explainability metadata endpoints,
- intervention and outcome workflows,
- audit trails and admin APIs.

Current status:

- this directory now contains a minimal runnable HTTP service scaffold,
- the production-like backend is still the Python service in `backend/`,
- this Go service is the place to migrate auth, RBAC, read models, audit, and
  reporting next.
