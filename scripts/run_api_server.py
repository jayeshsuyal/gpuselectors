#!/usr/bin/env python3
"""Run the optional FastAPI backend for frontend integration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from inference_atlas.api_server import create_app  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run InferenceAtlas API server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "uvicorn not installed. Install with: pip install 'uvicorn>=0.30,<1.0'"
        ) from exc

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
