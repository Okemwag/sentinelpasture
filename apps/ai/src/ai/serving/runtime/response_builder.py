"""Normalize inference outputs for upstream API consumers."""

from __future__ import annotations

from dataclasses import dataclass

from ..api.schemas import Driver, InterventionOption
from .feature_fetcher import FeatureSnapshot
from .model_loader import BaselineModel


@dataclass(frozen=True)
class ScoredSnapshot:
    snapshot: FeatureSnapshot
    risk_score: float
    confidence: float
    confidence_band: str
    drivers: list[Driver]
    known_data_gaps: list[str]


INTERVENTION_RULES = {
    "rfh_anomaly": (
        "Climate response coordination",
        "Short",
        ["proportional response", "county coordination"],
    ),
    "rfq_mean": (
        "Quarterly resilience review",
        "Medium",
        ["proportional response", "budget review"],
    ),
    "r1h_mean": (
        "Near-term field validation",
        "Short",
        ["civilian protection", "human review"],
    ),
    "r3h_mean": (
        "Medium-horizon contingency planning",
        "Medium",
        ["minimum force posture", "county coordination"],
    ),
    "rfh_mean": (
        "Resource allocation adjustment",
        "Short",
        ["proportional response", "human review"],
    ),
    "rfh_avg_mean": (
        "Baseline conditions review",
        "Medium",
        ["human review", "policy oversight"],
    ),
}


def score_snapshot(model: BaselineModel, snapshot: FeatureSnapshot) -> ScoredSnapshot:
    contributions: list[tuple[str, float]] = []
    for feature_name in model.feature_names:
        raw_value = snapshot.feature_values.get(feature_name, model.feature_mean.get(feature_name, 0.0))
        low = model.feature_min.get(feature_name, 0.0)
        high = model.feature_max.get(feature_name, low)
        if high <= low:
            normalized = 0.5
        else:
            normalized = max(0.0, min(1.0, (raw_value - low) / (high - low)))
        weight = model.feature_weights.get(feature_name, 0.0)
        contributions.append((feature_name, normalized * weight))

    raw_signal = sum(value for _, value in contributions)
    bounded_signal = max(0.0, min(1.0, raw_signal))
    risk_score = round(0.05 + (bounded_signal * 0.9), 4)

    ordered = sorted(contributions, key=lambda item: abs(item[1]), reverse=True)
    drivers = [
        Driver(
            name=_driver_label(name),
            contribution=round(abs(value), 4),
            direction="up" if value >= 0 else "down",
        )
        for name, value in ordered[:3]
    ]

    observations = int(snapshot.metadata.get("observations", "0") or "0")
    known_data_gaps: list[str] = []
    if observations < 3:
        known_data_gaps.append("Low observation count for the latest feature window.")
    if snapshot.metadata.get("level") != "national" and not snapshot.metadata.get("pcode"):
        known_data_gaps.append("Subnational region is missing a stable pcode mapping.")

    confidence = round(min(0.95, 0.55 + (min(observations, 20) * 0.015)), 2)
    if risk_score >= 0.75:
        confidence_band = "high"
    elif risk_score >= 0.45:
        confidence_band = "medium"
    else:
        confidence_band = "low"

    return ScoredSnapshot(
        snapshot=snapshot,
        risk_score=risk_score,
        confidence=confidence,
        confidence_band=confidence_band,
        drivers=drivers,
        known_data_gaps=known_data_gaps,
    )


def build_explanation(scored: ScoredSnapshot) -> tuple[str, list[str]]:
    if not scored.drivers:
        summary = "No sufficient feature signal is available to produce a reliable explanation."
    else:
        lead = scored.drivers[0]
        summary = (
            f"Risk for {scored.snapshot.region_id} is {scored.confidence_band} and is most influenced "
            f"by {lead.name.lower()} in the latest {scored.snapshot.feature_period} feature window."
        )

    notes = list(scored.known_data_gaps)
    notes.append(
        f"Model trained on {scored.snapshot.metadata.get('level', 'regional')} feature conventions may underfit new regimes."
    )
    return summary, notes


def build_interventions(scored: ScoredSnapshot) -> list[InterventionOption]:
    options: list[InterventionOption] = []
    impact = "High" if scored.risk_score >= 0.7 else "Moderate" if scored.risk_score >= 0.45 else "Low"
    confidence_label = "High" if scored.confidence >= 0.8 else "Medium"

    for driver in scored.drivers[:2]:
        rule = INTERVENTION_RULES.get(_feature_name_from_label(driver.name))
        if not rule:
            continue
        category, time_to_effect, constraints = rule
        options.append(
            InterventionOption(
                category=category,
                expected_impact=impact,
                time_to_effect=time_to_effect,
                confidence=confidence_label,
                constraints_applied=list(constraints),
            )
        )

    if not options:
        options.append(
            InterventionOption(
                category="Human analyst review",
                expected_impact=impact,
                time_to_effect="Short",
                confidence=confidence_label,
                constraints_applied=["human review", "policy oversight"],
            )
        )
    return options


def _driver_label(feature_name: str) -> str:
    labels = {
        "rfh_mean": "Rainfall level",
        "rfh_avg_mean": "Rainfall baseline",
        "rfh_anomaly": "Rainfall anomaly",
        "r1h_mean": "One-month pressure",
        "r3h_mean": "Three-month pressure",
        "rfq_mean": "Quarterly rainfall shift",
    }
    return labels.get(feature_name, feature_name.replace("_", " ").title())


def _feature_name_from_label(label: str) -> str:
    reverse = {
        "Rainfall level": "rfh_mean",
        "Rainfall baseline": "rfh_avg_mean",
        "Rainfall anomaly": "rfh_anomaly",
        "One-month pressure": "r1h_mean",
        "Three-month pressure": "r3h_mean",
        "Quarterly rainfall shift": "rfq_mean",
    }
    return reverse.get(label, "")
