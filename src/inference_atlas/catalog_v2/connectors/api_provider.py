"""API-first connector utilities for catalog_v2 providers.

The connector expects provider APIs (or internal endpoints) to return either:
1) a JSON object with a `rows` list, or
2) a top-level list of row objects.

Rows are normalized into CanonicalPricingRow. On any fetch/parse failure, callers
should gracefully fall back to normalized_catalog rows.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from inference_atlas.catalog_v2.schema import CanonicalPricingRow


def _coerce_row(provider_id: str, row: dict[str, Any], source_kind: str) -> CanonicalPricingRow:
    """Normalize a provider API row into CanonicalPricingRow."""
    billing_mode = str(row.get("billing_mode") or row.get("billing_type") or "")
    if not billing_mode:
        raise ValueError("missing billing_mode/billing_type")

    source_url = str(row.get("source_url") or row.get("source") or "")
    if not source_url:
        source_url = f"api:{provider_id}"

    throughput_value_raw = row.get("throughput_value")
    throughput_value: float | None = None
    if throughput_value_raw not in (None, "", "null"):
        throughput_value = float(throughput_value_raw)
        if throughput_value <= 0:
            throughput_value = None

    return CanonicalPricingRow(
        provider=str(row.get("provider") or provider_id),
        workload_type=str(row.get("workload_type") or ""),
        sku_key=str(row.get("sku_key") or ""),
        sku_name=str(row.get("sku_name") or ""),
        model_key=str(row.get("model_key") or ""),
        billing_mode=billing_mode,
        unit_price_usd=float(row.get("unit_price_usd")),
        unit_name=str(row.get("unit_name") or ""),
        region=str(row.get("region") or "global"),
        source_url=source_url,
        source_date=str(row.get("source_date") or ""),
        confidence=str(row.get("confidence") or "official"),
        source_kind=source_kind,
        throughput_value=throughput_value,
        throughput_unit=(str(row.get("throughput_unit") or "").strip() or None),
    )


def _extract_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        rows = payload.get("rows")
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    return []


def fetch_api_rows(
    *,
    provider_id: str,
    endpoint_env: str,
    token_env: str | None = None,
    source_kind: str = "provider_api",
    timeout_seconds: float = 8.0,
) -> list[CanonicalPricingRow]:
    """Fetch CanonicalPricingRows from a provider API endpoint.

    The endpoint URL is read from `endpoint_env`. If absent, this function
    returns an empty list (caller should fallback to normalized catalog).
    """
    endpoint = os.getenv(endpoint_env, "").strip()
    if not endpoint:
        return []

    headers = {"Accept": "application/json"}
    if token_env:
        token = os.getenv(token_env, "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"

    request = Request(endpoint, headers=headers, method="GET")
    try:
        with urlopen(request, timeout=timeout_seconds) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        return []

    rows = _extract_rows(payload)
    out: list[CanonicalPricingRow] = []
    for row in rows:
        try:
            out.append(_coerce_row(provider_id, row, source_kind))
        except (TypeError, ValueError):
            continue
    return out


def rows_to_dicts(rows: list[CanonicalPricingRow]) -> list[dict[str, object]]:
    """Serialize canonical rows for debugging/tests."""
    return [asdict(row) for row in rows]
