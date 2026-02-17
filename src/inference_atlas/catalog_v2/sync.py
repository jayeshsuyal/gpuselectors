"""Sync pipeline for catalog v2."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from inference_atlas.catalog_v2.connectors import (
    fetch_rows_for_provider,
    list_available_providers,
)

CATALOG_V2_PATH = Path(__file__).resolve().parents[3] / "data" / "catalog_v2" / "pricing_catalog.json"


def sync_catalog_v2(providers: list[str] | None = None) -> dict[str, object]:
    """Sync selected provider connectors into catalog_v2/pricing_catalog.json."""
    available = set(list_available_providers())
    raw_requested = providers or ["all"]
    requested = {p.strip() for p in raw_requested if p.strip()}
    if "all" in requested:
        selected = available
    else:
        selected = requested.intersection(available)

    rows = []
    connector_counts: dict[str, int] = {}
    for provider_id in sorted(selected):
        provider_rows = [row.to_dict() for row in fetch_rows_for_provider(provider_id)]
        rows.extend(provider_rows)
        connector_counts[provider_id] = len(provider_rows)

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
