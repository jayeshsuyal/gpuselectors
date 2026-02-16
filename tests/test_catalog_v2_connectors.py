from __future__ import annotations

from inference_atlas.catalog_v2.connectors import fetch_rows_for_provider
from inference_atlas.catalog_v2.connectors import normalized_catalog
from inference_atlas.catalog_v2.connectors import API_CONNECTORS
from inference_atlas.catalog_v2.schema import CanonicalPricingRow
from inference_atlas.catalog_v2.connectors import list_csv_providers


def test_fetch_rows_for_provider_uses_api_when_available(monkeypatch) -> None:
    def _fake_api_rows() -> list[CanonicalPricingRow]:
        return [
            CanonicalPricingRow(
                provider="fal_ai",
                workload_type="llm",
                sku_key="api_row",
                sku_name="API Row",
                model_key="llama_70b",
                billing_mode="per_token",
                unit_price_usd=1.0,
                unit_name="1m_tokens",
                region="global",
                source_url="api:fal_ai",
                source_date="2026-02-16",
                confidence="official",
                source_kind="provider_api",
            )
        ]

    monkeypatch.setitem(API_CONNECTORS, "fal_ai", _fake_api_rows)
    rows = fetch_rows_for_provider("fal_ai")
    assert rows
    assert rows[0].source_kind == "provider_api"
    assert rows[0].sku_key == "api_row"


def test_fetch_rows_for_provider_falls_back_when_api_empty(monkeypatch) -> None:
    monkeypatch.setitem(API_CONNECTORS, "fal_ai", lambda: [])
    expected = normalized_catalog.fetch_rows_for_provider("fal_ai")
    rows = fetch_rows_for_provider("fal_ai")
    assert rows
    assert rows == expected


def test_fetch_rows_for_provider_prefers_provider_csv_over_normalized() -> None:
    rows = fetch_rows_for_provider("fireworks")
    assert rows
    assert rows[0].source_kind == "provider_csv"


def test_list_csv_providers_detects_provider_files() -> None:
    providers = list_csv_providers()
    assert "openai" in providers
    assert "fireworks" in providers
