"""Build a first baseline training table and summary artifact.

This is a no-dependency placeholder for the first baseline workflow. It
constructs a merged monthly dataset and computes risk bands that a later
LightGBM implementation can consume directly.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = ROOT / "data" / "processed"


def build_baseline_training_dataset() -> None:
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


if __name__ == "__main__":
    build_baseline_training_dataset()
