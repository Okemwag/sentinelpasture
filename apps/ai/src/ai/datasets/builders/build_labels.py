"""Build canonical monthly label series from raw event exports."""

from __future__ import annotations

import calendar
import csv
import logging
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
logger = logging.getLogger("ai.datasets.builders.labels")


LABEL_FILES = {
    "demonstrations": RAW_DIR / "kenya_demonstration_events_by_month-year_as-of-25feb2026.csv",
    "civilian_targeting": RAW_DIR / "kenya_civilian_targeting_events_and_fatalities_by_month-year_as-of-25feb2026.csv",
    "political_violence": RAW_DIR / "kenya_political_violence_events_and_fatalities_by_month-year_as-of-25feb2026.csv",
}


def build_monthly_labels() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("building monthly event labels from %d source files", len(LABEL_FILES))

    rows: dict[tuple[int, int], dict] = {}
    for label_name, path in LABEL_FILES.items():
        logger.info("loading label source %s from %s", label_name, path)
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                year = int(row["Year"])
                month = list(calendar.month_name).index(row["Month"])
                bucket = rows.setdefault(
                    (year, month),
                    {
                        "country": row["Country"],
                        "year": year,
                        "month": month,
                        "demonstrations_events": 0,
                        "civilian_targeting_events": 0,
                        "civilian_targeting_fatalities": 0,
                        "political_violence_events": 0,
                        "political_violence_fatalities": 0,
                    },
                )

                if label_name == "demonstrations":
                    bucket["demonstrations_events"] = int(row["Events"])
                elif label_name == "civilian_targeting":
                    bucket["civilian_targeting_events"] = int(row["Events"])
                    bucket["civilian_targeting_fatalities"] = int(row["Fatalities"])
                elif label_name == "political_violence":
                    bucket["political_violence_events"] = int(row["Events"])
                    bucket["political_violence_fatalities"] = int(row["Fatalities"])

    output = PROCESSED_DIR / "event_labels_monthly_national.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "period",
                "country",
                "year",
                "month",
                "demonstrations_events",
                "civilian_targeting_events",
                "civilian_targeting_fatalities",
                "political_violence_events",
                "political_violence_fatalities",
                "total_events",
                "total_fatalities",
            ]
        )

        for key in sorted(rows):
            row = rows[key]
            total_events = (
                row["demonstrations_events"]
                + row["civilian_targeting_events"]
                + row["political_violence_events"]
            )
            total_fatalities = (
                row["civilian_targeting_fatalities"]
                + row["political_violence_fatalities"]
            )
            writer.writerow(
                [
                    f"{row['year']:04d}-{row['month']:02d}",
                    row["country"],
                    row["year"],
                    row["month"],
                    row["demonstrations_events"],
                    row["civilian_targeting_events"],
                    row["civilian_targeting_fatalities"],
                    row["political_violence_events"],
                    row["political_violence_fatalities"],
                    total_events,
                    total_fatalities,
                ]
            )
    logger.info("monthly event labels ready periods=%d output=%s", len(rows), output)


if __name__ == "__main__":
    from ai.common.logging import configure_logging

    configure_logging("ai.datasets.builders.labels")
    build_monthly_labels()
