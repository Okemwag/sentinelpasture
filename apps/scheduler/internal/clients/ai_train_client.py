"""HTTP client stub for AI retraining calls."""

from __future__ import annotations

import json
from urllib import request
from dataclasses import dataclass


@dataclass(frozen=True)
class AITrainClient:
    base_url: str = "http://localhost:8300"

    def retrain(self) -> dict:
        payload = json.dumps({"mode": "train", "train": True}).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/pipeline/run",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
