"""
Lightweight AI runtime for local development.

This keeps the backend's AI endpoints operational even when the separate
experimental ai-engine package is not installable in the current environment.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List


class LocalAIRuntime:
    """Small deterministic inference layer for local backend development."""

    engine_name = "lightweight_local_ai"
    model_version = "local-dev-1.0"

    async def process_signals(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        indicators = [self._build_indicator(signal) for signal in signals]
        threat_level = self._calculate_threat_level(indicators)
        confidence = round(min(0.96, 0.58 + (len(indicators) * 0.07)), 2)
        risk_factors = [indicator["type"].replace("_", " ").title() for indicator in indicators[:3]]

        return {
            "assessment": {
                "threat_level": threat_level,
                "confidence": confidence,
                "risk_factors": risk_factors or ["Baseline monitoring"],
            },
            "indicators": indicators,
            "recommendations": self._build_recommendations(indicators, threat_level),
            "metadata": {
                "model_version": self.model_version,
                "processing_time": round(0.04 + (len(signals) * 0.015), 3),
                "engine": self.engine_name,
                "timestamp": datetime.utcnow().isoformat(),
            },
        }

    async def predict_risk(self, region: str, timeframe: str) -> Dict[str, Any]:
        days = self._parse_days(timeframe)
        start = datetime.utcnow().date()
        base_risk = 48 + min(days, 14)

        predictions = []
        for offset in range(min(days, 7)):
            predictions.append(
                {
                    "date": (start + timedelta(days=offset)).isoformat(),
                    "risk_level": min(95, base_risk + offset),
                    "confidence": round(max(0.62, 0.85 - (offset * 0.02)), 2),
                }
            )

        return {
            "region": region,
            "timeframe": timeframe,
            "predictions": predictions,
            "confidence": 0.81,
            "factors": [
                "Signal velocity",
                "Recent anomaly density",
                "Historical trend carry-over",
            ],
            "metadata": {
                "model_version": self.model_version,
                "engine": self.engine_name,
            },
        }

    async def analyze_drivers(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        numeric_items = []
        for key, value in payload.items():
            if isinstance(value, (int, float)):
                numeric_items.append((key, float(value)))

        numeric_items.sort(key=lambda item: item[1], reverse=True)
        top_items = numeric_items[:3] or [("baseline_stress", 0.5)]
        total = sum(value for _, value in top_items) or 1.0

        drivers = [
            {
                "name": key.replace("_", " ").title(),
                "contribution": round(value / total, 2),
                "trend": "increasing" if value >= (top_items[0][1] * 0.7) else "stable",
                "confidence": 0.84,
            }
            for key, value in top_items
        ]

        causal_relationships = []
        if len(drivers) >= 2:
            causal_relationships.append(
                {
                    "from": drivers[0]["name"],
                    "to": drivers[1]["name"],
                    "strength": 0.63,
                }
            )

        return {
            "drivers": drivers,
            "causal_relationships": causal_relationships,
            "metadata": {
                "model_version": self.model_version,
                "engine": self.engine_name,
            },
        }

    async def recommend_interventions(
        self, region: str, risk_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        threat_level = int(risk_profile.get("threat_level", 55))
        interventions = [
            {
                "category": "Targeted field assessment",
                "expectedImpact": "High" if threat_level >= 70 else "Moderate",
                "timeToEffect": "Short",
                "costBand": "Low",
                "confidence": "High",
                "effectiveness_score": 0.83,
            },
            {
                "category": "Multi-agency coordination cell",
                "expectedImpact": "High",
                "timeToEffect": "Medium",
                "costBand": "Medium",
                "confidence": "High",
                "effectiveness_score": 0.79,
            },
        ]

        return {
            "region": region,
            "interventions": interventions,
            "metadata": {
                "model_version": self.model_version,
                "engine": self.engine_name,
            },
        }

    def _build_indicator(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        signal_type = str(signal.get("type", "general"))
        signal_data = signal.get("data", {}) or {}
        severity = self._score_signal(signal_type, signal_data)

        return {
            "type": signal_type,
            "severity": severity,
            "description": self._describe_signal(signal_type, signal_data, severity),
        }

    def _score_signal(self, signal_type: str, signal_data: Dict[str, Any]) -> float:
        score = 0.35
        numeric_values = [float(value) for value in signal_data.values() if isinstance(value, (int, float))]
        if numeric_values:
            score += min(sum(numeric_values) / max(len(numeric_values), 1) / 200.0, 0.35)

        text_values = " ".join(str(value).lower() for value in signal_data.values() if isinstance(value, str))
        for keyword, weight in (
            ("protest", 0.12),
            ("shortage", 0.10),
            ("violence", 0.18),
            ("drought", 0.09),
            ("price", 0.07),
            ("unemployment", 0.08),
        ):
            if keyword in text_values:
                score += weight

        type_bias = {
            "economic": 0.08,
            "environmental": 0.06,
            "social": 0.07,
            "security": 0.12,
        }
        score += type_bias.get(signal_type, 0.03)
        return round(min(score, 0.98), 2)

    def _describe_signal(
        self, signal_type: str, signal_data: Dict[str, Any], severity: float
    ) -> str:
        key_summary = ", ".join(list(signal_data.keys())[:2]) or "limited telemetry"
        return (
            f"{signal_type.replace('_', ' ').title()} signal flagged from {key_summary}; "
            f"severity scored at {severity:.2f}"
        )

    def _calculate_threat_level(self, indicators: List[Dict[str, Any]]) -> int:
        if not indicators:
            return 42

        avg_severity = sum(indicator["severity"] for indicator in indicators) / len(indicators)
        return max(1, min(100, int(round(avg_severity * 100))))

    def _build_recommendations(
        self, indicators: List[Dict[str, Any]], threat_level: int
    ) -> List[Dict[str, Any]]:
        recommendations = [
            {
                "action": "Increase analyst review cadence",
                "priority": 1,
                "estimated_impact": "High" if threat_level >= 65 else "Moderate",
            }
        ]

        if any(indicator["type"] == "economic" for indicator in indicators):
            recommendations.append(
                {
                    "action": "Escalate economic stress monitoring",
                    "priority": 2,
                    "estimated_impact": "High",
                }
            )

        if any(indicator["type"] == "environmental" for indicator in indicators):
            recommendations.append(
                {
                    "action": "Prepare resilience response coordination",
                    "priority": 3,
                    "estimated_impact": "Moderate",
                }
            )

        return recommendations

    def _parse_days(self, timeframe: str) -> int:
        cleaned = (timeframe or "7d").strip().lower()
        if cleaned.endswith("d") and cleaned[:-1].isdigit():
            return max(1, int(cleaned[:-1]))
        return 7

