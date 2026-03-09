"""HTTP client stub for feature refresh orchestration."""

from __future__ import annotations

import json
from urllib import request
from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureClient:
    base_url: str = "http://localhost:8300"

    def refresh_features(self) -> dict:
        payload = json.dumps({"mode": "features", "train": False}).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/pipeline/run",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
