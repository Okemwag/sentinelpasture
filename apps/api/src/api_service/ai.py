"""AI gateway for the API service."""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from urllib import error, parse, request


class AIGatewayError(RuntimeError):
    pass


@dataclass
class AIGateway:
    base_url: str = ""
    timeout_seconds: float = 5.0

    def __post_init__(self) -> None:
        configured_url = self.base_url or os.getenv("AI_INFERENCE_URL", "").strip()
        self.base_url = configured_url.rstrip("/") if configured_url else ""

    @property
    def remote_enabled(self) -> bool:
        return bool(self.base_url)

    @property
    def engine_name(self) -> str:
        return "contract_ai_service" if self.remote_enabled else "unconfigured_ai_service"

    async def process_signals(self, signals: list[dict[str, Any]]) -> dict[str, Any]:
        self._require_remote()
        started_at = time.perf_counter()
        region_id = self._region_from_signals(signals)
        risk = await self._post_json(
            "/infer/risk",
            {"region_id": region_id, "at_time": datetime.utcnow().isoformat(), "signals": signals},
        )
        explain = await self._post_json(
            "/infer/explain",
            {"region_id": region_id, "risk_score": risk["risk_score"]},
        )
        interventions = await self._post_json(
            "/infer/interventions",
            {"region_id": region_id, "risk_score": risk["risk_score"]},
        )
        return {
            "assessment": {
                "threat_level": int(round(float(risk["risk_score"]) * 100)),
                "confidence": float(risk["confidence"]),
                "risk_factors": [driver["name"] for driver in risk["top_drivers"]],
            },
            "indicators": [
                {
                    "type": str(driver["name"]).lower().replace(" ", "_"),
                    "severity": round(float(driver["contribution"]), 4),
                    "description": f"{driver['name']} is driving current risk in {risk['region_id']}.",
                }
                for driver in risk["top_drivers"]
            ],
            "recommendations": [
                {
                    "action": item["category"],
                    "priority": index + 1,
                    "estimated_impact": item["expected_impact"],
                }
                for index, item in enumerate(interventions["interventions"])
            ],
            "metadata": {
                "model_version": risk["model_version"],
                "processing_time": round(time.perf_counter() - started_at, 4),
                "engine": self.engine_name,
                "feature_snapshot_timestamp": risk["feature_snapshot_timestamp"],
                "known_data_gaps": risk["known_data_gaps"],
                "explanation_summary": explain["summary"],
            },
        }

    async def predict_risk(self, region: str, timeframe: str) -> dict[str, Any]:
        self._require_remote()
        risk = await self._post_json(
            "/infer/risk",
            {"region_id": region, "at_time": datetime.utcnow().isoformat(), "signals": []},
        )
        days = self._parse_days(timeframe)
        predictions = [
            {
                "date": (datetime.utcnow().date() + timedelta(days=offset)).isoformat(),
                "risk_level": int(round(float(risk["risk_score"]) * 100)),
                "confidence": round(float(risk["confidence"]), 2),
            }
            for offset in range(min(days, 7))
        ]
        return {
            "region": risk["region_id"],
            "timeframe": timeframe,
            "predictions": predictions,
            "confidence": risk["confidence"],
            "factors": [driver["name"] for driver in risk["top_drivers"]],
            "known_data_gaps": risk["known_data_gaps"],
            "metadata": {
                "model_version": risk["model_version"],
                "engine": self.engine_name,
                "feature_snapshot_timestamp": risk["feature_snapshot_timestamp"],
            },
        }

    async def analyze_drivers(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._require_remote()
        explain = await self._post_json(
            "/infer/explain",
            {"region_id": payload.get("region_id", "national"), "risk_score": payload.get("risk_score")},
        )
        drivers = [
            {
                "name": driver["name"],
                "contribution": round(float(driver["contribution"]), 4),
                "trend": "increasing" if driver["direction"] == "up" else "decreasing",
                "confidence": 0.8,
            }
            for driver in explain["top_drivers"]
        ]
        relationships = []
        if len(drivers) >= 2:
            relationships.append(
                {
                    "from": drivers[0]["name"],
                    "to": drivers[1]["name"],
                    "strength": round(max(drivers[0]["contribution"], drivers[1]["contribution"]), 4),
                }
            )
        return {
            "drivers": drivers,
            "causal_relationships": relationships,
            "metadata": {
                "model_version": explain["model_version"],
                "engine": self.engine_name,
                "uncertainty_notes": explain["uncertainty_notes"],
                "feature_snapshot_timestamp": explain["feature_snapshot_timestamp"],
            },
        }

    async def recommend_interventions(self, region: str, risk_profile: dict[str, Any]) -> dict[str, Any]:
        self._require_remote()
        risk_score = float(risk_profile.get("risk_score", risk_profile.get("threat_level", 0) / 100))
        response = await self._post_json(
            "/infer/interventions",
            {"region_id": region, "risk_score": risk_score},
        )
        return {
            "region": response["region_id"],
            "interventions": [
                {
                    "category": item["category"],
                    "expectedImpact": item["expected_impact"],
                    "timeToEffect": item["time_to_effect"],
                    "costBand": "Policy-defined",
                    "confidence": item["confidence"],
                    "constraintsApplied": item["constraints_applied"],
                }
                for item in response["interventions"]
            ],
            "metadata": {
                "model_version": response["model_version"],
                "engine": self.engine_name,
                "feature_snapshot_timestamp": response["feature_snapshot_timestamp"],
            },
        }

    async def list_regions(self, limit: int = 50) -> dict[str, Any]:
        self._require_remote()
        return await self._get_json("/infer/regions", {"limit": limit})

    async def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self._require_remote()

        def send() -> dict[str, Any]:
            body = json.dumps(payload).encode("utf-8")
            req = request.Request(
                f"{self.base_url}{path}",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with request.urlopen(req, timeout=self.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (error.URLError, error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
                raise AIGatewayError(str(exc)) from exc

        return await asyncio.to_thread(send)

    async def _get_json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._require_remote()

        def send() -> dict[str, Any]:
            query = f"?{parse.urlencode(params)}" if params else ""
            req = request.Request(f"{self.base_url}{path}{query}", method="GET")
            try:
                with request.urlopen(req, timeout=self.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (error.URLError, error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
                raise AIGatewayError(str(exc)) from exc

        return await asyncio.to_thread(send)

    def _require_remote(self) -> None:
        if not self.remote_enabled:
            raise AIGatewayError("AI inference URL is not configured")

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
