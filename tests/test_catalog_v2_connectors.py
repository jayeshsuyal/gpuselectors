from __future__ import annotations

from inference_atlas.catalog_v2.connectors import fetch_rows_for_provider
from inference_atlas.catalog_v2.connectors import normalized_catalog
from inference_atlas.catalog_v2.connectors import API_CONNECTORS
from inference_atlas.catalog_v2.schema import CanonicalPricingRow
from inference_atlas.catalog_v2.connectors import list_csv_providers
from inference_atlas.catalog_v2.connectors import provider_csv


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


def test_fetch_rows_for_provider_falls_back_to_provider_csv_when_api_empty(monkeypatch) -> None:
    monkeypatch.setitem(API_CONNECTORS, "fal_ai", lambda: [])
    expected = provider_csv.fetch_rows_for_provider("fal_ai")
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


def test_provider_csv_connector_maps_throughput_fields(tmp_path, monkeypatch) -> None:
    csv_path = tmp_path / "demo.csv"
    csv_path.write_text(
        "workload_type,provider,billing_type,sku_key,sku_name,model_key,unit_price_usd,unit_name,throughput_value,throughput_unit,memory_gb,latency_p50_ms,latency_p95_ms,region,notes,source_url,source_date,confidence\n"
        "speech_to_text,demo,per_minute,demo_1,Demo SKU,demo_model,0.01,audio_min,120,per_minute,,,,global,,https://example.com,2026-02-18,high\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(provider_csv, "PROVIDERS_CSV_DIR", tmp_path)
    rows = provider_csv.fetch_rows_for_provider("demo")
    assert len(rows) == 1
    assert rows[0].throughput_value == 120.0
    assert rows[0].throughput_unit == "per_minute"
