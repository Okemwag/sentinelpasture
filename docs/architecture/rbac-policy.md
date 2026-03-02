# RBAC Policy

Baseline roles:

- `viewer`: can read dashboards and reports.
- `analyst`: can inspect drivers, explanations, and submit notes.
- `operator`: can execute approved intervention workflows.
- `admin`: can manage users, policy packs, and system configuration.

High-risk actions such as policy changes, export of sensitive reports, and model
promotion should require tighter control than read-only access.
