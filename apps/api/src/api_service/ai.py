"""AI gateway for the API service."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from urllib import error, parse, request


class AIGatewayError(RuntimeError):
    pass


logger = logging.getLogger("api_service.ai_gateway")


@dataclass
class AIGateway:
    base_url: str = ""
    timeout_seconds: float = 5.0
    fallback_to_mock: bool = True

    def __post_init__(self) -> None:
        configured_url = self.base_url or os.getenv("AI_INFERENCE_URL", "http://localhost:8100").strip()
        self.base_url = configured_url.rstrip("/") if configured_url else ""
        self.fallback_to_mock = os.getenv("AI_FALLBACK_TO_MOCK", "true").strip().lower() not in {"0", "false", "no"}

    @property
    def remote_enabled(self) -> bool:
        return bool(self.base_url)

    @property
    def engine_name(self) -> str:
        if self.remote_enabled:
            return "contract_ai_service"
        return "mock_ai_service" if self.fallback_to_mock else "unconfigured_ai_service"

    async def process_signals(self, signals: list[dict[str, Any]]) -> dict[str, Any]:
        logger.info("process_signals called", extra={"signal_count": len(signals), "remote_enabled": self.remote_enabled})
        if not self.remote_enabled:
            if not self.fallback_to_mock:
                self._require_remote()
            logger.info("process_signals using mock fallback (remote disabled)")
            return self._mock_process_signals(signals)
        try:
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
        except AIGatewayError:
            if not self.fallback_to_mock:
                raise
            logger.warning("process_signals remote call failed; using mock fallback")
            return self._mock_process_signals(signals)

    async def predict_risk(self, region: str, timeframe: str) -> dict[str, Any]:
        logger.info("predict_risk called", extra={"region": region, "timeframe": timeframe, "remote_enabled": self.remote_enabled})
        if not self.remote_enabled:
            if not self.fallback_to_mock:
                self._require_remote()
            logger.info("predict_risk using mock fallback (remote disabled)", extra={"region": region})
            return self._mock_predict_risk(region, timeframe)
        try:
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
        except AIGatewayError:
            if not self.fallback_to_mock:
                raise
            logger.warning("predict_risk remote call failed; using mock fallback", extra={"region": region})
            return self._mock_predict_risk(region, timeframe)

    async def analyze_drivers(self, payload: dict[str, Any]) -> dict[str, Any]:
        logger.info("analyze_drivers called", extra={"region": payload.get("region_id", "national"), "remote_enabled": self.remote_enabled})
        if not self.remote_enabled:
            if not self.fallback_to_mock:
                self._require_remote()
            logger.info("analyze_drivers using mock fallback (remote disabled)")
            return self._mock_driver_analysis(payload)
        try:
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
        except AIGatewayError:
            if not self.fallback_to_mock:
                raise
            logger.warning("analyze_drivers remote call failed; using mock fallback")
            return self._mock_driver_analysis(payload)

    async def recommend_interventions(self, region: str, risk_profile: dict[str, Any]) -> dict[str, Any]:
        logger.info("recommend_interventions called", extra={"region": region, "remote_enabled": self.remote_enabled})
        if not self.remote_enabled:
            if not self.fallback_to_mock:
                self._require_remote()
            logger.info("recommend_interventions using mock fallback (remote disabled)", extra={"region": region})
            return self._mock_interventions(region, risk_profile)
        try:
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
        except AIGatewayError:
            if not self.fallback_to_mock:
                raise
            logger.warning("recommend_interventions remote call failed; using mock fallback", extra={"region": region})
            return self._mock_interventions(region, risk_profile)

    async def list_regions(self, limit: int = 50) -> dict[str, Any]:
        logger.info("list_regions called", extra={"limit": limit, "remote_enabled": self.remote_enabled})
        if not self.remote_enabled:
            if not self.fallback_to_mock:
                self._require_remote()
            logger.info("list_regions using mock fallback (remote disabled)")
            return self._mock_regions(limit)
        try:
            return await self._get_json("/infer/regions", {"limit": limit})
        except AIGatewayError:
            if not self.fallback_to_mock:
                raise
            logger.warning("list_regions remote call failed; using mock fallback")
            return self._mock_regions(limit)

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

    def _mock_regions(self, limit: int) -> dict[str, Any]:
        timestamp = datetime.utcnow().isoformat()
        rows = [
            {
                "region_id": "north-frontier",
                "risk_score": 0.72,
                "confidence": 0.82,
                "confidence_band": "high",
                "primary_driver": "Resource stress",
                "feature_snapshot_timestamp": timestamp,
            },
            {
                "region_id": "coastal-belt",
                "risk_score": 0.58,
                "confidence": 0.79,
                "confidence_band": "medium",
                "primary_driver": "Market volatility",
                "feature_snapshot_timestamp": timestamp,
            },
            {
                "region_id": "capital-corridor",
                "risk_score": 0.44,
                "confidence": 0.77,
                "confidence_band": "medium",
                "primary_driver": "Service disruption",
                "feature_snapshot_timestamp": timestamp,
            },
            {
                "region_id": "lake-region",
                "risk_score": 0.39,
                "confidence": 0.75,
                "confidence_band": "medium",
                "primary_driver": "Mobility pressure",
                "feature_snapshot_timestamp": timestamp,
            },
        ]
        return {
            "regions": rows[: max(limit, 1)],
            "model_version": "mock-model-v1",
            "feature_count": 6,
            "training_rows": 60,
        }

    def _mock_predict_risk(self, region: str, timeframe: str) -> dict[str, Any]:
        target_region = region or "national"
        today = datetime.utcnow().date()
        days = min(self._parse_days(timeframe), 7)
        base = 68 if target_region in {"national", "north-frontier"} else 56
        predictions = [
            {
                "date": (today + timedelta(days=offset)).isoformat(),
                "risk_level": min(95, base + offset),
                "confidence": 0.8,
            }
            for offset in range(days)
        ]
        return {
            "region": target_region,
            "timeframe": timeframe,
            "predictions": predictions,
            "confidence": 0.8,
            "factors": ["Resource stress", "Market volatility", "Service disruption"],
            "known_data_gaps": [],
            "metadata": {
                "model_version": "mock-model-v1",
                "engine": self.engine_name,
                "feature_snapshot_timestamp": datetime.utcnow().isoformat(),
            },
        }

    def _mock_driver_analysis(self, payload: dict[str, Any]) -> dict[str, Any]:
        region = str(payload.get("region_id", "national"))
        return {
            "drivers": [
                {"name": "Resource stress", "contribution": 0.34, "trend": "increasing", "confidence": 0.84},
                {"name": "Market volatility", "contribution": 0.27, "trend": "increasing", "confidence": 0.81},
                {"name": "Service disruption", "contribution": 0.19, "trend": "decreasing", "confidence": 0.78},
            ],
            "causal_relationships": [
                {"from": "Resource stress", "to": "Market volatility", "strength": 0.34}
            ],
            "metadata": {
                "model_version": "mock-model-v1",
                "engine": self.engine_name,
                "uncertainty_notes": [f"Mock driver profile for {region}."],
                "feature_snapshot_timestamp": datetime.utcnow().isoformat(),
            },
        }

    def _mock_interventions(self, region: str, risk_profile: dict[str, Any]) -> dict[str, Any]:
        risk_score = float(risk_profile.get("risk_score", risk_profile.get("threat_level", 0) / 100))
        confidence = "High" if risk_score >= 0.65 else "Medium"
        return {
            "region": region or "national",
            "interventions": [
                {
                    "category": "Economic stabilization support",
                    "expectedImpact": "High",
                    "timeToEffect": "2-6 weeks",
                    "costBand": "Policy-defined",
                    "confidence": confidence,
                    "constraintsApplied": ["budget_ceiling", "equity_screen"],
                },
                {
                    "category": "Targeted community mediation",
                    "expectedImpact": "Medium",
                    "timeToEffect": "1-3 weeks",
                    "costBand": "Policy-defined",
                    "confidence": "Medium",
                    "constraintsApplied": ["civilian_protection_rule"],
                },
            ],
            "metadata": {
                "model_version": "mock-model-v1",
                "engine": self.engine_name,
                "feature_snapshot_timestamp": datetime.utcnow().isoformat(),
            },
        }

    def _mock_process_signals(self, signals: list[dict[str, Any]]) -> dict[str, Any]:
        region_id = self._region_from_signals(signals)
        prediction = self._mock_predict_risk(region_id, "7d")
        first = prediction["predictions"][0]
        interventions = self._mock_interventions(region_id, {"risk_score": first["risk_level"] / 100})
        drivers = self._mock_driver_analysis({"region_id": region_id})["drivers"]
        return {
            "assessment": {
                "threat_level": first["risk_level"],
                "confidence": prediction["confidence"],
                "risk_factors": [row["name"] for row in drivers],
            },
            "indicators": [
                {
                    "type": row["name"].lower().replace(" ", "_"),
                    "severity": row["contribution"],
                    "description": f"{row['name']} is elevating near-term risk in {region_id}.",
                }
                for row in drivers
            ],
            "recommendations": [
                {
                    "action": row["category"],
                    "priority": idx + 1,
                    "estimated_impact": row["expectedImpact"],
                }
                for idx, row in enumerate(interventions["interventions"])
            ],
            "metadata": {
                "model_version": "mock-model-v1",
                "processing_time": 0.01,
                "engine": self.engine_name,
                "feature_snapshot_timestamp": datetime.utcnow().isoformat(),
                "known_data_gaps": [],
                "explanation_summary": "Mock response used because remote AI service is unavailable.",
            },
        }
