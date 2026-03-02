"""Fetch current feature snapshots for region-by-time inference."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
PROCESSED_DIR = ROOT / "data" / "processed"
SUBNATIONAL_FEATURES = PROCESSED_DIR / "rainfall_features_monthly_subnational.csv"
NATIONAL_FEATURES = PROCESSED_DIR / "rainfall_features_monthly_national.csv"


@dataclass(frozen=True)
class FeatureSnapshot:
    region_id: str
    feature_values: dict[str, float]
    feature_period: str
    observed_at: datetime
    metadata: dict[str, str]


def load_feature_snapshot(region_id: str) -> FeatureSnapshot:
    normalized = _normalize(region_id)
    if normalized in {"", "national"}:
        row = _latest_row(NATIONAL_FEATURES)
        if not row:
            raise FileNotFoundError(f"Missing feature data: {NATIONAL_FEATURES}")
        return _national_snapshot(row)

    matches = _matching_subnational_rows(normalized)
    if not matches:
        row = _latest_row(NATIONAL_FEATURES)
        if not row:
            raise FileNotFoundError(f"No feature snapshot available for region '{region_id}'")
        return _national_snapshot(row)

    return _subnational_snapshot(matches[-1])


def list_available_regions(limit: int = 100) -> list[FeatureSnapshot]:
    rows = _read_rows(SUBNATIONAL_FEATURES)
    if not rows:
        latest_national = _latest_row(NATIONAL_FEATURES)
        return [_national_snapshot(latest_national)] if latest_national else []

    latest_by_region: dict[str, dict[str, str]] = {}
    for row in rows:
        key = row.get("adm_id") or row.get("pcode") or row.get("adm_level") or "unknown"
        current = latest_by_region.get(key)
        if current is None or current.get("period", "") <= row.get("period", ""):
            latest_by_region[key] = row

    snapshots = [_subnational_snapshot(row) for row in latest_by_region.values()]
    snapshots.sort(key=lambda item: item.region_id)
    return snapshots[:limit]


def _matching_subnational_rows(normalized_region: str) -> list[dict[str, str]]:
    rows = _read_rows(SUBNATIONAL_FEATURES)
    matches: list[dict[str, str]] = []
    for row in rows:
        candidates = {
            _normalize(row.get("adm_id", "")),
            _normalize(row.get("pcode", "")),
            _normalize(row.get("adm_level", "")),
        }
        if normalized_region in candidates:
            matches.append(row)
    matches.sort(key=lambda item: item.get("period", ""))
    return matches


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _latest_row(path: Path) -> dict[str, str] | None:
    rows = _read_rows(path)
    if not rows:
        return None
    rows.sort(key=lambda item: item.get("period", ""))
    return rows[-1]


def _national_snapshot(row: dict[str, str]) -> FeatureSnapshot:
    return FeatureSnapshot(
        region_id="national",
        feature_values={
            "rfh_mean": _as_float(row, "rfh_mean"),
            "rfh_avg_mean": _as_float(row, "rfh_avg_mean"),
            "rfh_anomaly": _as_float(row, "rfh_anomaly"),
            "r1h_mean": _as_float(row, "r1h_mean"),
            "r3h_mean": _as_float(row, "r3h_mean"),
            "rfq_mean": _as_float(row, "rfq_mean"),
        },
        feature_period=row["period"],
        observed_at=_period_to_datetime(row["period"]),
        metadata={"level": "national", "observations": row.get("observations", "0")},
    )


def _subnational_snapshot(row: dict[str, str]) -> FeatureSnapshot:
    region_id = row.get("adm_id") or row.get("pcode") or "unknown"
    return FeatureSnapshot(
        region_id=region_id,
        feature_values={
            "rfh_mean": _as_float(row, "rfh_mean"),
            "rfh_avg_mean": _as_float(row, "rfh_avg_mean"),
            "rfh_anomaly": _as_float(row, "rfh_anomaly"),
            "r1h_mean": _as_float(row, "r1h_mean"),
            "r3h_mean": _as_float(row, "r3h_mean"),
            "rfq_mean": _as_float(row, "rfq_mean"),
        },
        feature_period=row["period"],
        observed_at=_period_to_datetime(row["period"]),
        metadata={
            "level": row.get("adm_level", "subnational"),
            "adm_id": row.get("adm_id", ""),
            "pcode": row.get("pcode", ""),
            "observations": row.get("observations", "0"),
        },
    )


def _as_float(row: dict[str, str], key: str) -> float:
    raw = (row.get(key) or "").strip()
    return float(raw) if raw else 0.0


def _period_to_datetime(period: str) -> datetime:
    return datetime.strptime(f"{period}-01", "%Y-%m-%d")


def _normalize(value: str) -> str:
    return "".join(ch for ch in value.lower().strip() if ch.isalnum())
