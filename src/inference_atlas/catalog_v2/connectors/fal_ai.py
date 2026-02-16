"""fal.ai API connector for catalog_v2."""

from __future__ import annotations

from inference_atlas.catalog_v2.connectors.api_provider import fetch_api_rows
from inference_atlas.catalog_v2.schema import CanonicalPricingRow


def fetch_fal_ai_rows() -> list[CanonicalPricingRow]:
    """Fetch fal.ai pricing rows from configured API endpoint."""
    return fetch_api_rows(
        provider_id="fal_ai",
        endpoint_env="FAL_AI_PRICING_API_URL",
        token_env="FAL_KEY",
        source_kind="provider_api",
    )
