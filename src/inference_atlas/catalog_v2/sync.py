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
CATALOG_V2_DELTAS_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "catalog_v2" / "pricing_deltas.json"
)
CATALOG_V2_HISTORY_DIR = Path(__file__).resolve().parents[3] / "data" / "catalog_v2" / "history"
MAX_HISTORY_SNAPSHOTS = 30

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


def _load_previous_rows() -> tuple[str | None, dict[tuple[str, str, str, str], float]]:
    if not CATALOG_V2_PATH.exists():
        return None, {}
    try:
        payload = json.loads(CATALOG_V2_PATH.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None, {}
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        return None, {}
    price_by_key: dict[tuple[str, str, str, str], float] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            price_by_key[_dedupe_key(row)] = float(row["unit_price_usd"])
        except (KeyError, TypeError, ValueError):
            continue
    return str(payload.get("generated_at_utc") or ""), price_by_key


def _compute_price_deltas(
    rows: list[dict[str, object]],
    previous_prices: dict[tuple[str, str, str, str], float],
) -> tuple[list[dict[str, object]], int, int]:
    delta_rows: list[dict[str, object]] = []
    matched_rows = 0
    changed_rows = 0
    for row in rows:
        key = _dedupe_key(row)
        prev_price = previous_prices.get(key)
        if prev_price is None:
            continue
        matched_rows += 1
        current_price = float(row["unit_price_usd"])
        abs_change = current_price - prev_price
        pct_change: float | None = None
        if prev_price > 0:
            pct_change = (abs_change / prev_price) * 100.0
        if abs(abs_change) > 1e-12:
            changed_rows += 1

        row["previous_unit_price_usd"] = prev_price
        row["price_change_abs_usd"] = abs_change
        if pct_change is not None:
            row["price_change_pct"] = pct_change

        delta_rows.append(
            {
                "provider": key[0],
                "sku_key": key[1],
                "unit_name": key[2],
                "region": key[3],
                "previous_unit_price_usd": prev_price,
                "unit_price_usd": current_price,
                "price_change_abs_usd": abs_change,
                "price_change_pct": pct_change,
            }
        )
    delta_rows.sort(
        key=lambda row: (
            str(row.get("provider") or ""),
            str(row.get("sku_key") or ""),
            str(row.get("unit_name") or ""),
            str(row.get("region") or ""),
        )
    )
    return delta_rows, matched_rows, changed_rows


def _snapshot_history(payload: dict[str, object]) -> None:
    CATALOG_V2_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot_path = CATALOG_V2_HISTORY_DIR / f"pricing_catalog_{timestamp}.json"
    snapshot_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    snapshots = sorted(CATALOG_V2_HISTORY_DIR.glob("pricing_catalog_*.json"))
    if len(snapshots) <= MAX_HISTORY_SNAPSHOTS:
        return
    for stale_path in snapshots[: len(snapshots) - MAX_HISTORY_SNAPSHOTS]:
        stale_path.unlink(missing_ok=True)


def sync_catalog_v2(providers: list[str] | None = None) -> dict[str, object]:
    """Sync selected provider connectors into catalog_v2/pricing_catalog.json."""
    available = set(list_available_providers())
    raw_requested = providers or ["all"]
    requested = {p.strip() for p in raw_requested if p.strip()}
    if "all" in requested:
        selected = available
    else:
        selected = requested.intersection(available)

    previous_generated_at, previous_prices = _load_previous_rows()

    rows = []
    connector_counts: dict[str, int] = {}
    for provider_id in sorted(selected):
        provider_rows = [row.to_dict() for row in fetch_rows_for_provider(provider_id)]
        rows.extend(provider_rows)
        connector_counts[provider_id] = len(provider_rows)
    rows, _ = _dedupe_rows(rows)

    delta_rows, matched_rows, changed_rows = _compute_price_deltas(rows, previous_prices)

    payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "providers_synced": sorted(selected),
        "row_count": len(rows),
        "connector_counts": connector_counts,
        "rows": rows,
    }
    deltas_payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "generated_at_utc": payload["generated_at_utc"],
        "baseline_generated_at_utc": previous_generated_at,
        "row_count": len(rows),
        "matched_rows": matched_rows,
        "changed_rows": changed_rows,
        "changes": delta_rows,
    }

    CATALOG_V2_PATH.parent.mkdir(parents=True, exist_ok=True)
    CATALOG_V2_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    CATALOG_V2_DELTAS_PATH.write_text(json.dumps(deltas_payload, indent=2) + "\n", encoding="utf-8")
    _snapshot_history(payload)
    return payload
