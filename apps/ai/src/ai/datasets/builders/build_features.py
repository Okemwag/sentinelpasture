"""Build canonical rainfall features from raw source files."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple


ROOT = Path(__file__).resolve().parents[4]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_FILE = RAW_DIR / "ken-rainfall-subnat-full.csv"


def build_monthly_features() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    national: Dict[Tuple[int, int], dict] = defaultdict(_empty_bucket)
    subnational: Dict[Tuple[int, int, str, str, str], dict] = defaultdict(_empty_bucket)

    with RAW_FILE.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            year, month = _parse_year_month(row["date"])

            national_bucket = national[(year, month)]
            _accumulate(national_bucket, row)

            sub_key = (
                year,
                month,
                row["adm_level"],
                row["adm_id"],
                row["PCODE"],
            )
            sub_bucket = subnational[sub_key]
            _accumulate(sub_bucket, row)

    _write_national(national)
    _write_subnational(subnational)


def _empty_bucket() -> dict:
    return {
        "count": 0,
        "rfh": 0.0,
        "rfh_avg": 0.0,
        "r1h": 0.0,
        "r1h_avg": 0.0,
        "r3h": 0.0,
        "r3h_avg": 0.0,
        "rfq": 0.0,
        "r1q": 0.0,
        "r3q": 0.0,
    }


def _accumulate(bucket: dict, row: dict) -> None:
    bucket["count"] += 1
    for key in ("rfh", "rfh_avg", "r1h", "r1h_avg", "r3h", "r3h_avg", "rfq", "r1q", "r3q"):
        raw_value = (row.get(key) or "").strip()
        if not raw_value:
            continue
        bucket[key] += float(raw_value)


def _parse_year_month(raw_date: str) -> Tuple[int, int]:
    year_text, month_text, _ = raw_date.split("-", 2)
    return int(year_text), int(month_text)


def _average(bucket: dict, key: str) -> float:
    return round(bucket[key] / max(bucket["count"], 1), 4)


def _write_national(rows: Dict[Tuple[int, int], dict]) -> None:
    output = PROCESSED_DIR / "rainfall_features_monthly_national.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "period",
                "year",
                "month",
                "rfh_mean",
                "rfh_avg_mean",
                "rfh_anomaly",
                "r1h_mean",
                "r1h_avg_mean",
                "r3h_mean",
                "r3h_avg_mean",
                "rfq_mean",
                "r1q_mean",
                "r3q_mean",
                "observations",
            ]
        )

        for (year, month) in sorted(rows):
            bucket = rows[(year, month)]
            rfh_mean = _average(bucket, "rfh")
            rfh_avg_mean = _average(bucket, "rfh_avg")
            writer.writerow(
                [
                    f"{year:04d}-{month:02d}",
                    year,
                    month,
                    rfh_mean,
                    rfh_avg_mean,
                    round(rfh_mean - rfh_avg_mean, 4),
                    _average(bucket, "r1h"),
                    _average(bucket, "r1h_avg"),
                    _average(bucket, "r3h"),
                    _average(bucket, "r3h_avg"),
                    _average(bucket, "rfq"),
                    _average(bucket, "r1q"),
                    _average(bucket, "r3q"),
                    bucket["count"],
                ]
            )


def _write_subnational(rows: Dict[Tuple[int, int, str, str, str], dict]) -> None:
    output = PROCESSED_DIR / "rainfall_features_monthly_subnational.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "period",
                "year",
                "month",
                "adm_level",
                "adm_id",
                "pcode",
                "rfh_mean",
                "rfh_avg_mean",
                "rfh_anomaly",
                "r1h_mean",
                "r3h_mean",
                "rfq_mean",
                "observations",
            ]
        )

        for key in sorted(rows):
            year, month, adm_level, adm_id, pcode = key
            bucket = rows[key]
            rfh_mean = _average(bucket, "rfh")
            rfh_avg_mean = _average(bucket, "rfh_avg")
            writer.writerow(
                [
                    f"{year:04d}-{month:02d}",
                    year,
                    month,
                    adm_level,
                    adm_id,
                    pcode,
                    rfh_mean,
                    rfh_avg_mean,
                    round(rfh_mean - rfh_avg_mean, 4),
                    _average(bucket, "r1h"),
                    _average(bucket, "r3h"),
                    _average(bucket, "rfq"),
                    bucket["count"],
                ]
            )


if __name__ == "__main__":
    build_monthly_features()
