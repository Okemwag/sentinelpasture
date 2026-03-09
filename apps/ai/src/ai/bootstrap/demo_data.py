"""Generate synthetic demo data when raw training inputs are unavailable."""

from __future__ import annotations

import argparse
import calendar
import csv
import json
import logging
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from ai.common.logging import configure_logging


ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
ARTIFACTS_DIR = ROOT / "data" / "artifacts"

RAINFALL_FILE = "ken-rainfall-subnat-full.csv"
DEMONSTRATIONS_FILE = "kenya_demonstration_events_by_month-year_as-of-25feb2026.csv"
CIVILIAN_TARGETING_FILE = "kenya_civilian_targeting_events_and_fatalities_by_month-year_as-of-25feb2026.csv"
POLITICAL_VIOLENCE_FILE = "kenya_political_violence_events_and_fatalities_by_month-year_as-of-25feb2026.csv"

REGIONS = [
    ("51325", "KE019"),
    ("51326", "KE020"),
    ("51327", "KE021"),
    ("51328", "KE022"),
    ("51329", "KE023"),
    ("51330", "KE024"),
    ("51331", "KE025"),
    ("51332", "KE026"),
    ("51333", "KE027"),
    ("51334", "KE028"),
    ("51335", "KE029"),
    ("51336", "KE030"),
]
OBSERVATION_DAYS = (1, 11, 21)
REQUIRED_RAW_FILES = (
    RAINFALL_FILE,
    DEMONSTRATIONS_FILE,
    CIVILIAN_TARGETING_FILE,
    POLITICAL_VIOLENCE_FILE,
)
REQUIRED_PROCESSED_FILES = (
    "rainfall_features_monthly_national.csv",
    "rainfall_features_monthly_subnational.csv",
    "event_labels_monthly_national.csv",
)
REQUIRED_ARTIFACT_FILES = ("baseline_risk_model.json",)


logger = logging.getLogger("ai.bootstrap.demo_data")


@dataclass(frozen=True)
class DemoBootstrapSummary:
    synthetic_raw_generated: bool
    raw_files: list[str]
    processed_rebuilt: bool
    artifact_rebuilt: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "synthetic_raw_generated": self.synthetic_raw_generated,
            "raw_files": self.raw_files,
            "processed_rebuilt": self.processed_rebuilt,
            "artifact_rebuilt": self.artifact_rebuilt,
        }


def missing_raw_files(*, raw_dir: Path | None = None) -> list[str]:
    target_dir = raw_dir or RAW_DIR
    return [name for name in REQUIRED_RAW_FILES if not (target_dir / name).exists()]


def generate_synthetic_raw_data(*, force: bool = False, raw_dir: Path | None = None) -> dict[str, object]:
    target_dir = raw_dir or RAW_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    missing = missing_raw_files(raw_dir=target_dir)
    if not force and not missing:
        logger.info("synthetic raw-data bootstrap skipped; required files already exist in %s", target_dir)
        return {
            "generated": False,
            "raw_dir": str(target_dir),
            "missing_before": [],
            "files": sorted(path.name for path in target_dir.glob("*.csv")),
        }

    logger.info(
        "generating synthetic demo raw data in %s force=%s missing_before=%s",
        target_dir,
        force,
        missing,
    )

    end_year, end_month = _last_complete_month()
    rainfall_rows = _write_rainfall_csv(target_dir / RAINFALL_FILE, end_year=end_year, end_month=end_month)
    demonstrations_rows = _write_event_csv(
        target_dir / DEMONSTRATIONS_FILE,
        event_type="demonstrations",
        end_year=end_year,
        end_month=end_month,
    )
    civilian_rows = _write_event_csv(
        target_dir / CIVILIAN_TARGETING_FILE,
        event_type="civilian_targeting",
        end_year=end_year,
        end_month=end_month,
    )
    political_rows = _write_event_csv(
        target_dir / POLITICAL_VIOLENCE_FILE,
        event_type="political_violence",
        end_year=end_year,
        end_month=end_month,
    )

    summary = {
        "generated": True,
        "raw_dir": str(target_dir),
        "missing_before": missing,
        "files": sorted(path.name for path in target_dir.glob("*.csv")),
        "row_counts": {
            RAINFALL_FILE: rainfall_rows,
            DEMONSTRATIONS_FILE: demonstrations_rows,
            CIVILIAN_TARGETING_FILE: civilian_rows,
            POLITICAL_VIOLENCE_FILE: political_rows,
        },
    }
    logger.info("synthetic demo raw data ready: %s", json.dumps(summary["row_counts"], sort_keys=True))
    return summary


def ensure_demo_assets(*, force_synthetic_raw: bool = False, force_rebuild: bool = False) -> DemoBootstrapSummary:
    configure_logging("ai.bootstrap.demo_data")

    generated_raw = False
    if force_synthetic_raw or missing_raw_files():
        generated_raw = bool(generate_synthetic_raw_data(force=force_synthetic_raw)["generated"])
    else:
        logger.info("raw demo inputs already available; synthetic generation not required")

    processed_missing = [name for name in REQUIRED_PROCESSED_FILES if not (PROCESSED_DIR / name).exists()]
    processed_rebuilt = False
    if force_rebuild or processed_missing:
        logger.info("building processed feature and label tables missing=%s", processed_missing)
        from ai.datasets.builders.build_features import build_monthly_features
        from ai.datasets.builders.build_labels import build_monthly_labels

        build_monthly_features()
        build_monthly_labels()
        processed_rebuilt = True
    else:
        logger.info("processed tables already available; rebuild not required")

    artifact_missing = [name for name in REQUIRED_ARTIFACT_FILES if not (ARTIFACTS_DIR / name).exists()]
    artifact_rebuilt = False
    if force_rebuild or artifact_missing:
        logger.info("training baseline artifact missing=%s", artifact_missing)
        from ai.training.train_lgbm import build_baseline_training_dataset

        build_baseline_training_dataset()
        artifact_rebuilt = True
    else:
        logger.info("baseline artifact already available; rebuild not required")

    summary = DemoBootstrapSummary(
        synthetic_raw_generated=generated_raw,
        raw_files=sorted(path.name for path in RAW_DIR.glob("*.csv")),
        processed_rebuilt=processed_rebuilt,
        artifact_rebuilt=artifact_rebuilt,
    )
    logger.info("demo asset bootstrap summary: %s", json.dumps(summary.as_dict(), sort_keys=True))
    return summary


def _write_rainfall_csv(path: Path, *, end_year: int, end_month: int) -> int:
    row_count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "date",
                "adm_level",
                "adm_id",
                "PCODE",
                "n_pixels",
                "rfh",
                "rfh_avg",
                "r1h",
                "r1h_avg",
                "r3h",
                "r3h_avg",
                "rfq",
                "r1q",
                "r3q",
                "version",
            ]
        )

        for year, month in _month_iter(1997, 1, end_year, end_month):
            for region_index, (adm_id, pcode) in enumerate(REGIONS):
                for day_index, day in enumerate(OBSERVATION_DAYS):
                    values = _rainfall_values(
                        year=year,
                        month=month,
                        day_index=day_index,
                        region_index=region_index,
                    )
                    writer.writerow(
                        [
                            f"{year:04d}-{month:02d}-{day:02d}",
                            1,
                            adm_id,
                            pcode,
                            220 + (region_index * 17),
                            values["rfh"],
                            values["rfh_avg"],
                            values["r1h"],
                            values["r1h_avg"],
                            values["r3h"],
                            values["r3h_avg"],
                            values["rfq"],
                            values["r1q"],
                            values["r3q"],
                            "synthetic-demo",
                        ]
                    )
                    row_count += 1
    return row_count


def _write_event_csv(path: Path, *, event_type: str, end_year: int, end_month: int) -> int:
    row_count = 0
    include_fatalities = event_type != "demonstrations"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if include_fatalities:
            writer.writerow(["Country", "Month", "Year", "Events", "Fatalities"])
        else:
            writer.writerow(["Country", "Month", "Year", "Events"])

        for year, month in _month_iter(1997, 1, end_year, end_month):
            events, fatalities = _event_values(year=year, month=month, event_type=event_type)
            row = ["Kenya", calendar.month_name[month], year, events]
            if include_fatalities:
                row.append(fatalities)
            writer.writerow(row)
            row_count += 1
    return row_count


def _rainfall_values(*, year: int, month: int, day_index: int, region_index: int) -> dict[str, float]:
    year_index = year - 1997
    month_angle = ((month - 1) / 12.0) * (2.0 * math.pi)
    region_bias = (region_index - ((len(REGIONS) - 1) / 2.0)) * 1.25
    submonthly_wave = (-3.5, 0.0, 3.5)[day_index]

    baseline = 28.0 + (18.0 * math.sin(month_angle - 0.7)) + (6.0 * math.sin((2.0 * month_angle) + 0.35))
    climate_cycle = 8.5 * math.sin((year_index / 3.4) + (region_index * 0.45))
    anomaly = climate_cycle + (4.5 * math.cos((3.0 * month_angle) + (day_index * 0.8)))

    rfh_avg = max(4.0, baseline + (region_bias * 0.6) + 16.0)
    rfh = max(0.2, rfh_avg + anomaly + submonthly_wave)
    r1h_avg = max(6.0, rfh_avg * 2.9)
    r1h = max(0.5, r1h_avg + (anomaly * 2.2) + (submonthly_wave * 2.0))
    r3h_avg = max(20.0, rfh_avg * 8.8)
    r3h = max(1.0, r3h_avg + (anomaly * 6.0) + (region_bias * 4.0))
    rfq = max(8.0, (rfh * 3.35) + (12.0 * math.cos(month_angle + region_index)))
    r1q = max(8.0, r1h * 1.22)
    r3q = max(8.0, r3h * 0.52)

    return {
        "rfh": round(rfh, 6),
        "rfh_avg": round(rfh_avg, 6),
        "r1h": round(r1h, 6),
        "r1h_avg": round(r1h_avg, 6),
        "r3h": round(r3h, 6),
        "r3h_avg": round(r3h_avg, 6),
        "rfq": round(rfq, 6),
        "r1q": round(r1q, 6),
        "r3q": round(r3q, 6),
    }


def _event_values(*, year: int, month: int, event_type: str) -> tuple[int, int]:
    year_index = year - 1997
    month_angle = ((month - 1) / 12.0) * (2.0 * math.pi)

    structural_pressure = 2.2 + (1.1 * math.sin(year_index / 2.7)) + (0.7 * math.cos(month_angle - 0.25))
    seasonal_pressure = max(0.0, 1.3 * math.sin(month_angle + 0.9))
    drought_pressure = max(0.0, 1.9 * math.sin((year_index / 3.6) + (month * 0.5)))
    pressure = max(0.0, structural_pressure + seasonal_pressure + drought_pressure)

    if event_type == "demonstrations":
        events = max(0, int(round(2 + (pressure * 2.1) + (0.25 * (year_index / 2.0)))))
        return events, 0

    if event_type == "civilian_targeting":
        events = max(0, int(round(1 + (pressure * 1.35) + max(0.0, drought_pressure * 0.8))))
        fatalities = max(0, int(round((events * 3.2) + (pressure * 5.5))))
        return events, fatalities

    events = max(0, int(round(2 + (pressure * 1.7) + max(0.0, seasonal_pressure * 1.2))))
    fatalities = max(0, int(round((events * 4.1) + (pressure * 6.8))))
    return events, fatalities


def _month_iter(start_year: int, start_month: int, end_year: int, end_month: int):
    year = start_year
    month = start_month
    while (year, month) <= (end_year, end_month):
        yield year, month
        month += 1
        if month > 12:
            month = 1
            year += 1


def _last_complete_month(today: date | None = None) -> tuple[int, int]:
    anchor = today or date.today()
    if anchor.month == 1:
        return anchor.year - 1, 12
    return anchor.year, anchor.month - 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic governance-intel demo data.")
    parser.add_argument("--raw-only", action="store_true", help="Generate only the raw CSV inputs.")
    parser.add_argument("--force", action="store_true", help="Overwrite the synthetic demo inputs even if files exist.")
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild processed data and artifacts.")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional raw-data output directory for raw-only generation.",
    )
    args = parser.parse_args()

    configure_logging("ai.bootstrap.demo_data")

    if args.raw_only:
        output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
        summary = generate_synthetic_raw_data(force=args.force, raw_dir=output_dir)
        print(json.dumps(summary, sort_keys=True))
        return

    summary = ensure_demo_assets(force_synthetic_raw=args.force, force_rebuild=args.force_rebuild)
    print(json.dumps(summary.as_dict(), sort_keys=True))


if __name__ == "__main__":
    main()
