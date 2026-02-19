"""Sync pipeline for catalog v2."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from inference_atlas.catalog_v2.connectors import (
    fetch_rows_for_provider,
    list_available_providers,
)
from inference_atlas.contracts import ConfidenceLevel

CATALOG_V2_PATH = Path(__file__).resolve().parents[3] / "data" / "catalog_v2" / "pricing_catalog.json"

_SOURCE_KIND_SCORE = {
    "provider_api": 3,
    "provider_csv": 2,
    "normalized_catalog": 1,
}


def _dedupe_key(row: dict[str, object]) -> tuple[str, str, str, str]:
    return (
        str(row.get("provider") or "").strip(),
        str(row.get("sku_key") or "").strip(),
        str(row.get("unit_name") or "").strip(),
        str(row.get("region") or "").strip(),
    )


def _confidence_score(value: object) -> int:
    try:
        return ConfidenceLevel(str(value or "estimated")).score
    except ValueError:
        return 0


def _source_date_score(value: object) -> float:
    raw = str(value or "").strip()
    if not raw:
        return 0.0
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _row_priority(row: dict[str, object]) -> tuple[int, int, float, int]:
    return (
        _confidence_score(row.get("confidence")),
        _SOURCE_KIND_SCORE.get(str(row.get("source_kind") or "").strip(), 0),
        _source_date_score(row.get("source_date")),
        1 if row.get("throughput_value") not in (None, "") else 0,
    )


def _dedupe_rows(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], int]:
    best_by_key: dict[tuple[str, str, str, str], dict[str, object]] = {}
    for row in rows:
        key = _dedupe_key(row)
        current = best_by_key.get(key)
        if current is None or _row_priority(row) > _row_priority(current):
            best_by_key[key] = row
    deduped = sorted(
        best_by_key.values(),
        key=lambda row: (
            str(row.get("provider") or ""),
            str(row.get("workload_type") or ""),
            str(row.get("sku_key") or ""),
            str(row.get("unit_name") or ""),
            str(row.get("region") or ""),
        ),
    )
    return deduped, len(rows) - len(deduped)


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
    rows, _ = _dedupe_rows(rows)

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
