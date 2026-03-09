"""HTTP client stub for AI inference calls."""

from __future__ import annotations

import json
from urllib import parse, request
from dataclasses import dataclass


@dataclass(frozen=True)
class AIInferClient:
    base_url: str = "http://localhost:8100"

    def list_regions(self, *, limit: int = 50) -> dict:
        query = parse.urlencode({"limit": limit})
        with request.urlopen(f"{self.base_url}/infer/regions?{query}", timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
