"""CLI entrypoint for the ingestion service."""

from __future__ import annotations

import sys
from pathlib import Path

import uvicorn


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    uvicorn.run("ingestion_service.main:app", host="0.0.0.0", port=8300, reload=True)


if __name__ == "__main__":
    main()
