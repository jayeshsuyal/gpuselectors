"""Google Cloud pricing connector for catalog_v2."""

from __future__ import annotations

from inference_atlas.catalog_v2.connectors.api_provider import fetch_api_rows
from inference_atlas.catalog_v2.schema import CanonicalPricingRow


def fetch_google_cloud_rows() -> list[CanonicalPricingRow]:
    """Fetch Google Cloud pricing rows from configured API endpoint."""
    return fetch_api_rows(
        provider_id="google_cloud",
        endpoint_env="GOOGLE_CLOUD_PRICING_API_URL",
        token_env="GOOGLE_CLOUD_PRICING_API_TOKEN",
        source_kind="provider_api",
    )
