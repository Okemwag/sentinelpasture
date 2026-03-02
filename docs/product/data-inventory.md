# Data Inventory And Mapping Rules

This document defines the minimum viable dataset stack for the pilot and the
baseline geography assumptions.

## Region Unit

The recommended MVP unit is `county`.

Reasons:

- county boundaries match existing administrative workflows,
- county-level coordination is operationally realistic for a pilot,
- the data sources in scope can be aggregated consistently at county level.

Ward and grid units can be introduced later only when governance, data quality,
and privacy controls are strong enough to support them.

## MVP Dataset Stack

- `ACLED`: incident labels and historical instability outcomes
- `CHIRPS`: rainfall and climate stress indicators
- `MODIS NDVI`: vegetation and drought proxy indicators
- `OSM`: roads, corridors, and adjacency context
- `Market prices` (optional): staple and commodity stress indicators

## Mapping Rules

- All source records must map to a canonical county identifier before entering
  feature generation.
- Region-time tables must use the same county key across ingestion, features,
  scoring, and dashboard views.
- Raw source geometry should be preserved only in source storage; dashboard
  views should consume aggregated county-level summaries.
- When a source arrives below county precision, it must be joined to the county
  table using a documented fallback mapping rule.
- When a source arrives above county precision, it must be aggregated upward to
  county level before it becomes operationally visible.

## Region By Time Contract

The minimum analytical unit for the MVP is:

- one row per `county x time_window`
- one consistent feature timestamp convention
- one label timestamp convention
- one reproducible mapping from raw source record to county aggregate

## Success Standard

This phase is complete when the same county-by-time table can be reproduced
consistently from ACLED, CHIRPS, MODIS NDVI, OSM, and any approved optional
market feeds.
