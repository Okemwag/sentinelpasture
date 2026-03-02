"""Dashboard read-model payloads."""

from __future__ import annotations

from datetime import datetime, timedelta


class DashboardService:
    model_version = "policy-pack-v0.1"

    def metadata(self) -> dict:
        return {
            "modelVersion": self.model_version,
            "lastUpdated": datetime.utcnow().isoformat(),
            "confidence": "High",
        }

    def stability(self) -> dict:
        return {
            "data": {"value": 67, "trend": "down", "confidence": "High", "change": "-3.2 points"},
            "success": True,
            "metadata": self.metadata(),
        }

    def stats(self) -> dict:
        return {
            "data": {"activeAlerts": 3, "regionsMonitored": 47, "dataSources": 24, "lastUpdate": "2 minutes ago"},
            "success": True,
            "metadata": self.metadata(),
        }

    def drivers(self) -> dict:
        return {
            "data": [
                {"label": "Economic stress", "percentage": 34, "trend": "up", "confidence": "High"},
                {"label": "Climate anomaly", "percentage": 28, "trend": "up", "confidence": "Medium"},
                {"label": "Mobility disruption", "percentage": 18, "trend": "stable", "confidence": "High"},
                {"label": "Education attendance decline", "percentage": 12, "trend": "down", "confidence": "Medium"},
                {"label": "Incident contagion", "percentage": 8, "trend": "up", "confidence": "High"},
            ],
            "success": True,
            "metadata": self.metadata(),
        }

    def alerts(self) -> dict:
        now = datetime.utcnow()
        return {
            "data": [
                {
                    "id": 1,
                    "severity": "elevated",
                    "title": "Economic stress indicators rising in Turkana County",
                    "description": "Unemployment rate increased by 2.3% over the past 30 days. Food prices up 8.4%.",
                    "timestamp": (now - timedelta(hours=2)).isoformat(),
                    "region": "Turkana County",
                    "status": "active",
                },
                {
                    "id": 2,
                    "severity": "moderate",
                    "title": "Drought conditions persisting in Northern regions",
                    "description": "Rainfall 22% below seasonal average. Water scarcity affecting 3 counties.",
                    "timestamp": (now - timedelta(hours=24)).isoformat(),
                    "region": "Northern Region",
                    "status": "active",
                },
                {
                    "id": 3,
                    "severity": "elevated",
                    "title": "Youth unemployment spike in Nairobi County",
                    "description": "Youth unemployment reached 14.2%, up from 12.1% last quarter.",
                    "timestamp": (now - timedelta(hours=48)).isoformat(),
                    "region": "Nairobi County",
                    "status": "monitoring",
                },
                {
                    "id": 4,
                    "severity": "moderate",
                    "title": "Healthcare capacity strain in Mombasa",
                    "description": "Hospital bed occupancy at 89%. Staffing levels below recommended threshold.",
                    "timestamp": (now - timedelta(hours=72)).isoformat(),
                    "region": "Mombasa County",
                    "status": "resolved",
                },
            ],
            "success": True,
            "metadata": self.metadata(),
        }

    def regional_data(self) -> dict:
        return {
            "data": [
                {"region": "Nairobi County", "population": "4.4M", "stabilityIndex": 72, "trend": "Stable", "primaryDriver": "Economic stress", "confidence": "High"},
                {"region": "Mombasa County", "population": "1.2M", "stabilityIndex": 68, "trend": "Moderating", "primaryDriver": "Infrastructure strain", "confidence": "High"},
                {"region": "Kisumu County", "population": "1.1M", "stabilityIndex": 81, "trend": "Stable", "primaryDriver": "Mobility disruption", "confidence": "Medium"},
                {"region": "Nakuru County", "population": "2.2M", "stabilityIndex": 75, "trend": "Stable", "primaryDriver": "Climate anomaly", "confidence": "High"},
                {"region": "Turkana County", "population": "926K", "stabilityIndex": 64, "trend": "Elevated", "primaryDriver": "Climate anomaly", "confidence": "Medium"},
                {"region": "Kiambu County", "population": "2.4M", "stabilityIndex": 78, "trend": "Stable", "primaryDriver": "Economic stress", "confidence": "High"},
            ],
            "success": True,
            "metadata": self.metadata(),
        }

