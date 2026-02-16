"""OpenAI connector for catalog v2.

This connector currently maps existing normalized catalog rows into v2 schema.
It is intentionally conservative and safe while we establish v2 plumbing.
"""

from __future__ import annotations

from inference_atlas.catalog_v2.schema import CanonicalPricingRow
from inference_atlas.data_loader import get_pricing_records
from inference_atlas.workload_types import WorkloadType


def fetch_openai_rows() -> list[CanonicalPricingRow]:
    """Fetch OpenAI pricing rows from normalized dataset and map to canonical v2 rows."""
    rows = get_pricing_records(WorkloadType.LLM)
    out: list[CanonicalPricingRow] = []
    for row in rows:
        if row.provider != "openai":
            continue
        out.append(
            CanonicalPricingRow(
                provider=row.provider,
                workload_type=row.workload_type.value,
                sku_key=row.sku_key,
                sku_name=row.sku_name,
                model_key=row.model_key,
                billing_mode=row.billing_type,
                unit_price_usd=row.unit_price_usd,
                unit_name=row.unit_name,
                region=row.region,
                source_url=row.source_url,
                source_date=row.source_date,
                confidence=row.confidence or "estimated",
                source_kind="normalized_catalog",
            )
        )
    return out
