"""Generic connector that maps normalized pricing rows into catalog v2 schema."""

from __future__ import annotations

from inference_atlas.catalog_v2.schema import CanonicalPricingRow
from inference_atlas.data_loader import get_pricing_records


def list_available_providers() -> list[str]:
    """Return sorted provider IDs available in normalized pricing records."""
    rows = get_pricing_records()
    providers = sorted({row.provider for row in rows})
    return providers


def fetch_rows_for_provider(provider_id: str) -> list[CanonicalPricingRow]:
    """Map all normalized rows for a provider into canonical catalog_v2 rows."""
    rows = get_pricing_records()
    out: list[CanonicalPricingRow] = []
    for row in rows:
        if row.provider != provider_id:
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
