"""Build a baseline training table and persist a real scoring artifact."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

from ai.datasets.builders.build_features import build_monthly_features
from ai.datasets.builders.build_labels import build_monthly_labels


ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = ROOT / "data" / "processed"
ARTIFACTS_DIR = ROOT / "data" / "artifacts"


def build_baseline_training_dataset() -> None:
    _ensure_source_tables()

    features = _read_indexed(PROCESSED_DIR / "rainfall_features_monthly_national.csv")
    labels = _read_indexed(PROCESSED_DIR / "event_labels_monthly_national.csv")

    merged_rows = []
    target_scores = []
    for period in sorted(set(features) & set(labels)):
        feature = features[period]
        label = labels[period]
        target_score = round(
            float(label["total_events"]) + (min(float(label["total_fatalities"]), 100.0) / 10.0),
            4,
        )
        target_scores.append(target_score)
        merged_rows.append(
            {
                "period": period,
                "year": label["year"],
                "month": label["month"],
                "rfh_mean": feature["rfh_mean"],
                "rfh_avg_mean": feature["rfh_avg_mean"],
                "rfh_anomaly": feature["rfh_anomaly"],
                "r1h_mean": feature["r1h_mean"],
                "r3h_mean": feature["r3h_mean"],
                "rfq_mean": feature["rfq_mean"],
                "demonstrations_events": label["demonstrations_events"],
                "civilian_targeting_events": label["civilian_targeting_events"],
                "civilian_targeting_fatalities": label["civilian_targeting_fatalities"],
                "political_violence_events": label["political_violence_events"],
                "political_violence_fatalities": label["political_violence_fatalities"],
                "total_events": label["total_events"],
                "total_fatalities": label["total_fatalities"],
                "target_score": target_score,
            }
        )

    low_cut, high_cut = _tertile_thresholds(target_scores)

    output_csv = PROCESSED_DIR / "training_baseline_monthly_national.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(merged_rows[0].keys()) + ["risk_band"] if merged_rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
        for row in merged_rows:
            row = dict(row)
            row["risk_band"] = _risk_band(float(row["target_score"]), low_cut, high_cut)
            writer.writerow(row)

    summary = {
        "dataset": "training_baseline_monthly_national",
        "rows": len(merged_rows),
        "date_range": {
            "start": merged_rows[0]["period"] if merged_rows else None,
            "end": merged_rows[-1]["period"] if merged_rows else None,
        },
        "feature_columns": [
            "rfh_mean",
            "rfh_avg_mean",
            "rfh_anomaly",
            "r1h_mean",
            "r3h_mean",
            "rfq_mean",
        ],
        "label_columns": [
            "demonstrations_events",
            "civilian_targeting_events",
            "civilian_targeting_fatalities",
            "political_violence_events",
            "political_violence_fatalities",
            "total_events",
            "total_fatalities",
        ],
        "target_definition": "total_events + min(total_fatalities, 100) / 10",
        "risk_band_thresholds": {
            "low_upper": low_cut,
            "moderate_upper": high_cut,
        },
    }

    output_json = PROCESSED_DIR / "lgbm_baseline_summary.json"
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    _write_model_artifact(merged_rows, low_cut, high_cut)


def _ensure_source_tables() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if not (PROCESSED_DIR / "rainfall_features_monthly_national.csv").exists():
        try:
            build_monthly_features()
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "Missing rainfall source data. Populate apps/ai/data/raw/ with the CHIRPS-derived "
                "rainfall export before training."
            ) from exc
    if not (PROCESSED_DIR / "event_labels_monthly_national.csv").exists():
        try:
            build_monthly_labels()
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "Missing ACLED label source data. Populate apps/ai/data/raw/ with the event exports "
                "before training."
            ) from exc


def _read_indexed(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {row["period"]: row for row in reader}


def _tertile_thresholds(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    ordered = sorted(values)
    low_index = max(0, (len(ordered) // 3) - 1)
    high_index = max(0, ((2 * len(ordered)) // 3) - 1)
    return ordered[low_index], ordered[high_index]


def _risk_band(score: float, low_cut: float, high_cut: float) -> str:
    if score <= low_cut:
        return "low"
    if score <= high_cut:
        return "moderate"
    return "high"


def _write_model_artifact(rows: list[dict[str, object]], low_cut: float, high_cut: float) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No training rows were produced; cannot train baseline artifact.")

    feature_names = [
        "rfh_mean",
        "rfh_avg_mean",
        "rfh_anomaly",
        "r1h_mean",
        "r3h_mean",
        "rfq_mean",
    ]

    feature_values: dict[str, list[float]] = {
        name: [float(row[name]) for row in rows] for name in feature_names
    }
    targets = [float(row["target_score"]) for row in rows]
    target_min = min(targets)
    target_max = max(targets)
    target_span = max(target_max - target_min, 1e-9)
    normalized_targets = [(value - target_min) / target_span for value in targets]

    raw_weights = {
        name: _signed_correlation(feature_values[name], normalized_targets) for name in feature_names
    }
    weight_sum = sum(abs(value) for value in raw_weights.values()) or 1.0
    feature_weights = {
        name: round(abs(raw_weights[name]) / weight_sum, 6) for name in feature_names
    }

    payload = {
        "model_version": "baseline-risk-model-v1",
        "trained_at": datetime.now(UTC).isoformat(),
        "feature_names": feature_names,
        "feature_min": {name: min(values) for name, values in feature_values.items()},
        "feature_max": {name: max(values) for name, values in feature_values.items()},
        "feature_mean": {
            name: round(sum(values) / max(len(values), 1), 6) for name, values in feature_values.items()
        },
        "feature_weights": feature_weights,
        "target_bounds": {"min": target_min, "max": target_max},
        "band_thresholds": {"moderate": low_cut, "high": high_cut},
        "training_rows": len(rows),
        "source_period": {"start": str(rows[0]["period"]), "end": str(rows[-1]["period"])},
    }

    with (ARTIFACTS_DIR / "baseline_risk_model.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _signed_correlation(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = 0.0
    left_sq = 0.0
    right_sq = 0.0
    for left_value, right_value in zip(left, right, strict=True):
        left_delta = left_value - left_mean
        right_delta = right_value - right_mean
        numerator += left_delta * right_delta
        left_sq += left_delta * left_delta
        right_sq += right_delta * right_delta
    denominator = (left_sq * right_sq) ** 0.5
    if denominator <= 0:
        return 0.0
    return numerator / denominator


if __name__ == "__main__":
    build_baseline_training_dataset()
