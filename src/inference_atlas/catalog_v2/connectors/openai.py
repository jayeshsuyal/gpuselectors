"""OpenAI connector for catalog v2."""

from __future__ import annotations

from inference_atlas.catalog_v2.schema import CanonicalPricingRow
from inference_atlas.catalog_v2.connectors.normalized_catalog import fetch_rows_for_provider


def fetch_openai_rows() -> list[CanonicalPricingRow]:
    """Fetch OpenAI pricing rows mapped to canonical v2 rows."""
    return fetch_rows_for_provider("openai")
