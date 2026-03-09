"""Ingestion pipeline orchestration."""

from __future__ import annotations

import csv
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from .config import Settings


logger = logging.getLogger("ingestion.pipeline")


class Orchestrator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def describe(self) -> dict:
        return {
            "environment": self.settings.environment,
            "collectors": [
                "acled",
                "chirps",
                "ndvi_modis",
                "market_prices",
                "corridors",
                "manual_upload",
            ],
            "stages": [
                "collect",
                "validate",
                "deduplicate",
                "normalize",
                "persist_raw",
                "persist_processed",
            ],
        }

    def run(self, *, mode: str = "full", train: bool = True) -> dict:
        started_at = time.time()
        normalized_mode = mode.strip().lower()
        if normalized_mode not in {"full", "features", "train"}:
            raise ValueError("mode must be one of: full, features, train")

        stages: list[dict[str, object]] = []
        logger.info("pipeline_run started mode=%s train=%s", normalized_mode, train)

        if normalized_mode in {"full", "features"}:
            stages.append(self._bootstrap_stage())
            stages.append(self._collect_stage())
            stages.append(self._validate_stage())
            stages.append(self._build_features_stage())
            if normalized_mode == "full":
                stages.append(self._build_labels_stage())

        if normalized_mode in {"full", "train"} and train:
            stages.append(self._train_stage())

        duration = round(time.time() - started_at, 3)
        generated = {
            "processed": sorted(p.name for p in self.settings.ai_data_processed_path.glob("*.csv")),
            "artifacts": sorted(p.name for p in self.settings.ai_data_artifacts_path.glob("*.json")),
        }
        logger.info("pipeline_run completed mode=%s duration_seconds=%.3f", normalized_mode, duration)
        return {
            "status": "completed",
            "mode": normalized_mode,
            "train": train,
            "duration_seconds": duration,
            "stages": stages,
            "generated": generated,
        }

    def _bootstrap_stage(self) -> dict[str, object]:
        required = [
            "ken-rainfall-subnat-full.csv",
            "kenya_demonstration_events_by_month-year_as-of-25feb2026.csv",
            "kenya_civilian_targeting_events_and_fatalities_by_month-year_as-of-25feb2026.csv",
            "kenya_political_violence_events_and_fatalities_by_month-year_as-of-25feb2026.csv",
        ]
        missing_before = [name for name in required if not (self.settings.ai_data_raw_path / name).exists()]
        if not missing_before and not self.settings.synthetic_bootstrap_force:
            logger.info("bootstrap stage skipped; required raw files already available")
            return {
                "stage": "bootstrap_demo_data",
                "status": "skipped",
                "generated": False,
                "missing_before": [],
            }

        if not self.settings.synthetic_bootstrap_enabled:
            raise FileNotFoundError(
                "Required raw files are missing and synthetic bootstrap is disabled: "
                + ", ".join(missing_before)
            )

        logger.info(
            "bootstrap stage generating synthetic demo raw data force=%s missing_before=%s",
            self.settings.synthetic_bootstrap_force,
            missing_before,
        )
        args = ["--raw-only"]
        if self.settings.synthetic_bootstrap_force:
            args.append("--force")
        self._run_python_module("ai.bootstrap.demo_data", args=args)
        raw_files = sorted(p.name for p in self.settings.ai_data_raw_path.glob("*.csv"))
        return {
            "stage": "bootstrap_demo_data",
            "status": "ok",
            "generated": True,
            "missing_before": missing_before,
            "raw_file_count": len(raw_files),
            "raw_files": raw_files,
        }

    def _collect_stage(self) -> dict[str, object]:
        raw_files = sorted(p.name for p in self.settings.ai_data_raw_path.glob("*.csv"))
        if not raw_files:
            raise FileNotFoundError(f"No raw data files found in {self.settings.ai_data_raw_path}")
        logger.info("collect stage discovered %d raw files", len(raw_files))
        return {"stage": "collect", "status": "ok", "raw_file_count": len(raw_files), "raw_files": raw_files}

    def _validate_stage(self) -> dict[str, object]:
        required = [
            "ken-rainfall-subnat-full.csv",
            "kenya_demonstration_events_by_month-year_as-of-25feb2026.csv",
            "kenya_civilian_targeting_events_and_fatalities_by_month-year_as-of-25feb2026.csv",
            "kenya_political_violence_events_and_fatalities_by_month-year_as-of-25feb2026.csv",
        ]
        missing = [name for name in required if not (self.settings.ai_data_raw_path / name).exists()]
        if missing:
            raise FileNotFoundError(f"Missing required raw files: {', '.join(missing)}")

        row_counts: dict[str, int] = {}
        for filename in required:
            path = self.settings.ai_data_raw_path / filename
            with path.open(newline="", encoding="utf-8") as handle:
                row_counts[filename] = max(sum(1 for _ in csv.reader(handle)) - 1, 0)
        logger.info("validate stage ok for %d files", len(required))
        return {"stage": "validate", "status": "ok", "row_counts": row_counts}

    def _build_features_stage(self) -> dict[str, object]:
        self._run_python_module("ai.datasets.builders.build_features")
        outputs = [
            "rainfall_features_monthly_national.csv",
            "rainfall_features_monthly_subnational.csv",
        ]
        logger.info("normalize stage built features")
        return {"stage": "normalize_features", "status": "ok", "outputs": outputs}

    def _build_labels_stage(self) -> dict[str, object]:
        self._run_python_module("ai.datasets.builders.build_labels")
        outputs = ["event_labels_monthly_national.csv"]
        logger.info("normalize stage built labels")
        return {"stage": "normalize_labels", "status": "ok", "outputs": outputs}

    def _train_stage(self) -> dict[str, object]:
        self._run_python_module("ai.training.train_lgbm")
        outputs = [
            "training_baseline_monthly_national.csv",
            "lgbm_baseline_summary.json",
            "baseline_risk_model.json",
        ]
        logger.info("train stage built baseline artifact")
        return {"stage": "train", "status": "ok", "outputs": outputs}

    def _run_python_module(self, module: str, *, args: list[str] | None = None) -> None:
        env = os.environ.copy()
        ai_src = str(self.settings.ai_src_path)
        env["PYTHONPATH"] = ai_src if not env.get("PYTHONPATH") else f"{ai_src}:{env['PYTHONPATH']}"
        env["PYTHONUNBUFFERED"] = "1"
        cmd = [sys.executable, "-m", module, *(args or [])]
        process = subprocess.Popen(
            cmd,
            cwd=self.settings.repo_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        output_lines: list[str] = []
        assert process.stdout is not None
        for line in process.stdout:
            message = line.rstrip()
            if not message:
                continue
            output_lines.append(message)
            logger.info("[%s] %s", module, message)
        return_code = process.wait()
        if return_code != 0:
            details = output_lines[-1] if output_lines else "no output"
            logger.error("module %s failed: %s", module, details)
            raise RuntimeError(f"Failed to run {module}: {details}")
