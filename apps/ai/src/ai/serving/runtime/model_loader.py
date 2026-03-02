"""Load and select model versions for serving."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
ARTIFACTS_DIR = ROOT / "data" / "artifacts"
MODEL_ARTIFACT = ARTIFACTS_DIR / "baseline_risk_model.json"


@dataclass(frozen=True)
class BaselineModel:
    model_version: str
    trained_at: str
    feature_names: list[str]
    feature_min: dict[str, float]
    feature_max: dict[str, float]
    feature_mean: dict[str, float]
    feature_weights: dict[str, float]
    target_bounds: dict[str, float]
    band_thresholds: dict[str, float]
    training_rows: int
    source_period: dict[str, str | None]


def load_baseline_model() -> BaselineModel:
    if not MODEL_ARTIFACT.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {MODEL_ARTIFACT}. Run the baseline trainer first."
        )
    payload = json.loads(MODEL_ARTIFACT.read_text(encoding="utf-8"))
    return BaselineModel(
        model_version=payload["model_version"],
        trained_at=payload["trained_at"],
        feature_names=payload["feature_names"],
        feature_min=payload["feature_min"],
        feature_max=payload["feature_max"],
        feature_mean=payload["feature_mean"],
        feature_weights=payload["feature_weights"],
        target_bounds=payload["target_bounds"],
        band_thresholds=payload["band_thresholds"],
        training_rows=payload["training_rows"],
        source_period=payload["source_period"],
    )
