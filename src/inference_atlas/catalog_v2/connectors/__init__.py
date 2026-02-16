"""Provider connectors for catalog v2."""

from inference_atlas.catalog_v2.connectors.aws_rekognition import (
    fetch_aws_rekognition_rows,
)
from inference_atlas.catalog_v2.connectors.fal_ai import fetch_fal_ai_rows
from inference_atlas.catalog_v2.connectors.google_cloud import fetch_google_cloud_rows
from inference_atlas.catalog_v2.connectors.normalized_catalog import (
    fetch_rows_for_provider as fetch_rows_from_normalized_catalog,
    list_available_providers as list_normalized_providers,
)
from inference_atlas.catalog_v2.connectors.provider_csv import (
    fetch_rows_for_provider as fetch_rows_from_provider_csv,
    list_csv_providers,
)
from inference_atlas.catalog_v2.connectors.openai import fetch_openai_rows

API_CONNECTORS = {
    "aws_rekognition": fetch_aws_rekognition_rows,
    "fal_ai": fetch_fal_ai_rows,
    "google_cloud": fetch_google_cloud_rows,
}


def fetch_rows_for_provider(provider_id: str):
    """Fetch provider rows with API -> provider CSV -> normalized fallback."""
    api_fetcher = API_CONNECTORS.get(provider_id)
    if api_fetcher is not None:
        api_rows = api_fetcher()
        if api_rows:
            return api_rows
    csv_rows = fetch_rows_from_provider_csv(provider_id)
    if csv_rows:
        return csv_rows
    return fetch_rows_from_normalized_catalog(provider_id)


def list_available_providers() -> list[str]:
    """Return all providers available across API, provider CSV, and normalized sources."""
    providers = set(list_normalized_providers())
    providers.update(list(API_CONNECTORS))
    providers.update(list_csv_providers())
    return sorted(providers)


__all__ = [
    "API_CONNECTORS",
    "fetch_aws_rekognition_rows",
    "fetch_fal_ai_rows",
    "fetch_google_cloud_rows",
    "fetch_openai_rows",
    "list_csv_providers",
    "list_available_providers",
    "fetch_rows_for_provider",
]
