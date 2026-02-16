"""Sync pipeline for catalog v2."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from inference_atlas.catalog_v2.connectors import fetch_openai_rows

CATALOG_V2_PATH = Path(__file__).resolve().parents[3] / "data" / "catalog_v2" / "pricing_catalog.json"


def sync_catalog_v2(providers: list[str] | None = None) -> dict[str, object]:
    """Sync selected provider connectors into catalog_v2/pricing_catalog.json."""
    selected = set(providers or ["openai"])

    rows = []
    connector_counts: dict[str, int] = {}
    if "openai" in selected:
        openai_rows = [row.to_dict() for row in fetch_openai_rows()]
        rows.extend(openai_rows)
        connector_counts["openai"] = len(openai_rows)

    payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "providers_synced": sorted(selected),
        "row_count": len(rows),
        "connector_counts": connector_counts,
        "rows": rows,
    }

    CATALOG_V2_PATH.parent.mkdir(parents=True, exist_ok=True)
    CATALOG_V2_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
