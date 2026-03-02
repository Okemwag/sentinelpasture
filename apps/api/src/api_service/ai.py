"""AI gateway for the API service."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from urllib import error, request


class AIGatewayError(RuntimeError):
    pass


@dataclass
class AIGateway:
    base_url: str = ""
    timeout_seconds: float = 3.0

    def __post_init__(self) -> None:
        configured_url = self.base_url or os.getenv("AI_INFERENCE_URL", "").strip()
        self.base_url = configured_url.rstrip("/") if configured_url else ""

    @property
    def remote_enabled(self) -> bool:
        return bool(self.base_url)

    @property
    def engine_name(self) -> str:
        return "contract_ai_service" if self.remote_enabled else "local_python_fallback_ai"

    async def process_signals(self, signals: list[dict[str, Any]]) -> dict[str, Any]:
        if not self.remote_enabled:
            return self._fallback_process(signals)
        region_id = self._region_from_signals(signals)
        try:
            risk = await self._post_json("/infer/risk", {"region_id": region_id, "at_time": datetime.utcnow().isoformat(), "signals": signals})
            explain = await self._post_json("/infer/explain", {"region_id": region_id, "risk_score": risk["risk_score"]})
            interventions = await self._post_json("/infer/interventions", {"region_id": region_id, "risk_score": risk["risk_score"]})
            return {
                "assessment": {
                    "threat_level": int(round(float(risk["risk_score"]) * 100)),
                    "confidence": float(risk["confidence"]),
                    "risk_factors": [driver["name"] for driver in risk["top_drivers"]],
                },
                "indicators": [
                    {
                        "type": str(driver["name"]).lower().replace(" ", "_"),
                        "severity": round(float(driver["contribution"]), 2),
                        "description": f"{driver['name']} driving risk in {region_id}; direction is {driver['direction']}",
                    }
                    for driver in risk["top_drivers"]
                ],
                "recommendations": [
                    {"action": item["category"], "priority": index + 1, "estimated_impact": item["expected_impact"]}
                    for index, item in enumerate(interventions["interventions"])
                ],
                "metadata": {
                    "model_version": risk["model_version"],
                    "processing_time": 0.12,
                    "engine": "contract_ai_service",
                    "feature_snapshot_timestamp": risk["feature_snapshot_timestamp"],
                    "known_data_gaps": risk["known_data_gaps"],
                    "explanation_summary": explain["summary"],
                },
            }
        except AIGatewayError:
            return self._fallback_process(signals)

    async def predict_risk(self, region: str, timeframe: str) -> dict[str, Any]:
        if not self.remote_enabled:
            return self._fallback_predict(region, timeframe)
        try:
            risk = await self._post_json("/infer/risk", {"region_id": region, "at_time": datetime.utcnow().isoformat(), "signals": []})
            days = self._parse_days(timeframe)
            predictions = [
                {
                    "date": (datetime.utcnow().date() + timedelta(days=offset)).isoformat(),
                    "risk_level": min(99, int(round(float(risk["risk_score"]) * 100)) + offset),
                    "confidence": max(0.5, round(float(risk["confidence"]) - (offset * 0.02), 2)),
                }
                for offset in range(min(days, 7))
            ]
            return {
                "region": region,
                "timeframe": timeframe,
                "predictions": predictions,
                "confidence": risk["confidence"],
                "factors": [driver["name"] for driver in risk["top_drivers"]],
                "metadata": {"model_version": risk["model_version"], "engine": "contract_ai_service"},
            }
        except AIGatewayError:
            return self._fallback_predict(region, timeframe)

    async def analyze_drivers(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.remote_enabled:
            return self._fallback_analyze_drivers(payload)
        try:
            explain = await self._post_json("/infer/explain", {"region_id": payload.get("region_id", "national"), "risk_score": payload.get("risk_score", 0.6)})
            drivers = [
                {
                    "name": driver["name"],
                    "contribution": round(float(driver["contribution"]), 2),
                    "trend": "increasing" if driver["direction"] == "up" else "stable",
                    "confidence": 0.84,
                }
                for driver in explain["top_drivers"]
            ]
            relationships = []
            if len(drivers) >= 2:
                relationships.append({"from": drivers[0]["name"], "to": drivers[1]["name"], "strength": 0.61})
            return {
                "drivers": drivers,
                "causal_relationships": relationships,
                "metadata": {"model_version": explain["model_version"], "engine": "contract_ai_service", "uncertainty_notes": explain["uncertainty_notes"]},
            }
        except AIGatewayError:
            return self._fallback_analyze_drivers(payload)

    async def recommend_interventions(self, region: str, risk_profile: dict[str, Any]) -> dict[str, Any]:
        if not self.remote_enabled:
            return self._fallback_recommend(region, risk_profile)
        try:
            risk_score = float(risk_profile.get("risk_score", risk_profile.get("threat_level", 55) / 100))
            response = await self._post_json("/infer/interventions", {"region_id": region, "risk_score": risk_score})
            return {
                "region": region,
                "interventions": [
                    {
                        "category": item["category"],
                        "expectedImpact": item["expected_impact"],
                        "timeToEffect": item["time_to_effect"],
                        "costBand": "Policy-defined",
                        "confidence": item["confidence"],
                        "effectiveness_score": 0.78,
                        "constraintsApplied": item["constraints_applied"],
                    }
                    for item in response["interventions"]
                ],
                "metadata": {"model_version": response["model_version"], "engine": "contract_ai_service"},
            }
        except AIGatewayError:
            return self._fallback_recommend(region, risk_profile)

    async def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.base_url:
            raise AIGatewayError("AI inference URL is not configured")

        def send() -> dict[str, Any]:
            body = json.dumps(payload).encode("utf-8")
            req = request.Request(f"{self.base_url}{path}", data=body, headers={"Content-Type": "application/json"}, method="POST")
            try:
                with request.urlopen(req, timeout=self.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (error.URLError, error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
                raise AIGatewayError(str(exc)) from exc

        return await asyncio.to_thread(send)

    def _fallback_process(self, signals: list[dict[str, Any]]) -> dict[str, Any]:
        indicators = [self._build_indicator(signal) for signal in signals]
        threat_level = 42 if not indicators else int(round(sum(ind["severity"] for ind in indicators) / len(indicators) * 100))
        recommendations = [{"action": "Increase analyst review cadence", "priority": 1, "estimated_impact": "High" if threat_level >= 65 else "Moderate"}]
        return {
            "assessment": {
                "threat_level": threat_level,
                "confidence": round(min(0.96, 0.58 + (len(indicators) * 0.07)), 2),
                "risk_factors": [ind["type"].replace("_", " ").title() for ind in indicators[:3]] or ["Baseline monitoring"],
            },
            "indicators": indicators,
            "recommendations": recommendations,
            "metadata": {
                "model_version": "local-dev-1.0",
                "processing_time": round(0.04 + (len(signals) * 0.015), 3),
                "engine": "local_python_fallback_ai",
                "timestamp": datetime.utcnow().isoformat(),
            },
        }

    def _fallback_predict(self, region: str, timeframe: str) -> dict[str, Any]:
        days = self._parse_days(timeframe)
        base_risk = 48 + min(days, 14)
        return {
            "region": region,
            "timeframe": timeframe,
            "predictions": [
                {
                    "date": (datetime.utcnow().date() + timedelta(days=offset)).isoformat(),
                    "risk_level": min(95, base_risk + offset),
                    "confidence": round(max(0.62, 0.85 - (offset * 0.02)), 2),
                }
                for offset in range(min(days, 7))
            ],
            "confidence": 0.81,
            "factors": ["Signal velocity", "Recent anomaly density", "Historical trend carry-over"],
            "metadata": {"model_version": "local-dev-1.0", "engine": "local_python_fallback_ai"},
        }

    def _fallback_analyze_drivers(self, payload: dict[str, Any]) -> dict[str, Any]:
        numeric = [(key, float(value)) for key, value in payload.items() if isinstance(value, (int, float))]
        numeric.sort(key=lambda item: item[1], reverse=True)
        top = numeric[:3] or [("baseline_stress", 0.5)]
        total = sum(value for _, value in top) or 1.0
        drivers = [{"name": key.replace("_", " ").title(), "contribution": round(value / total, 2), "trend": "increasing" if value >= top[0][1] * 0.7 else "stable", "confidence": 0.84} for key, value in top]
        relationships = [{"from": drivers[0]["name"], "to": drivers[1]["name"], "strength": 0.63}] if len(drivers) >= 2 else []
        return {"drivers": drivers, "causal_relationships": relationships, "metadata": {"model_version": "local-dev-1.0", "engine": "local_python_fallback_ai"}}

    def _fallback_recommend(self, region: str, risk_profile: dict[str, Any]) -> dict[str, Any]:
        threat_level = int(risk_profile.get("threat_level", 55))
        impact = "High" if threat_level >= 70 else "Moderate"
        return {
            "region": region,
            "interventions": [
                {"category": "Targeted field assessment", "expectedImpact": impact, "timeToEffect": "Short", "costBand": "Low", "confidence": "High", "effectiveness_score": 0.83},
                {"category": "Multi-agency coordination cell", "expectedImpact": "High", "timeToEffect": "Medium", "costBand": "Medium", "confidence": "High", "effectiveness_score": 0.79},
            ],
            "metadata": {"model_version": "local-dev-1.0", "engine": "local_python_fallback_ai"},
        }

    def _build_indicator(self, signal: dict[str, Any]) -> dict[str, Any]:
        signal_type = str(signal.get("type", "general"))
        signal_data = signal.get("data") or {}
        severity = self._score_signal(signal_type, signal_data)
        keys = ", ".join(list(signal_data.keys())[:2]) or "limited telemetry"
        return {"type": signal_type, "severity": severity, "description": f"{signal_type.replace('_', ' ').title()} signal flagged from {keys}; severity scored at {severity:.2f}"}

    def _score_signal(self, signal_type: str, signal_data: dict[str, Any]) -> float:
        score = 0.35
        numeric = [float(v) for v in signal_data.values() if isinstance(v, (int, float))]
        if numeric:
            score += min(sum(numeric) / max(len(numeric), 1) / 200.0, 0.35)
        text_blob = " ".join(str(v).lower() for v in signal_data.values() if isinstance(v, str))
        for keyword, weight in (("protest", 0.12), ("shortage", 0.10), ("violence", 0.18), ("drought", 0.09), ("price", 0.07), ("unemployment", 0.08)):
            if keyword in text_blob:
                score += weight
        score += {"economic": 0.08, "environmental": 0.06, "social": 0.07, "security": 0.12}.get(signal_type, 0.03)
        return round(min(score, 0.98), 2)

    def _region_from_signals(self, signals: list[dict[str, Any]]) -> str:
        for signal in signals:
            location = signal.get("location") or {}
            region = location.get("region") or location.get("name")
            if region:
                return str(region)
        return "national"

    def _parse_days(self, timeframe: str) -> int:
        cleaned = (timeframe or "7d").strip().lower()
        if cleaned.endswith("d") and cleaned[:-1].isdigit():
            return max(1, int(cleaned[:-1]))
        return 7

