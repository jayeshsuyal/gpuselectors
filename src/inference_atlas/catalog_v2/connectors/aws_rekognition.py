"""AWS Rekognition pricing connector for catalog_v2."""

from __future__ import annotations

from inference_atlas.catalog_v2.connectors.api_provider import fetch_api_rows
from inference_atlas.catalog_v2.schema import CanonicalPricingRow


def fetch_aws_rekognition_rows() -> list[CanonicalPricingRow]:
    """Fetch AWS Rekognition pricing rows from configured API endpoint."""
    return fetch_api_rows(
        provider_id="aws_rekognition",
        endpoint_env="AWS_REKOGNITION_PRICING_API_URL",
        token_env=None,
        source_kind="provider_api",
    )
