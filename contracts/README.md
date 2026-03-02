# Contracts

This directory is the future source of truth for cross-service contracts.

- `openapi/` for external and admin HTTP APIs.
- `proto/` for internal gRPC contracts when needed.
- `schemas/` for shared canonical feature, event, and model-output schemas.

The current running code does not fully consume these contracts yet. They are
being established first so service boundaries can harden before deeper rewrites.
