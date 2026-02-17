from __future__ import annotations

from types import SimpleNamespace

import pytest

from inference_atlas.catalog_ranking import (
    build_provider_diagnostics,
    normalize_unit_price_for_workload,
    rank_catalog_offers,
)


def test_normalize_audio_min_to_audio_hour() -> None:
    price = normalize_unit_price_for_workload(
        unit_price_usd=0.2,
        unit_name="audio_min",
        workload_type="speech_to_text",
    )
    assert price == pytest.approx(12.0)


def test_normalize_1k_chars_to_1m_chars() -> None:
    price = normalize_unit_price_for_workload(
        unit_price_usd=0.03,
        unit_name="1k_chars",
        workload_type="tts",
    )
    assert price == pytest.approx(30.0)


def test_normalize_image_unit_to_1k_images() -> None:
    price = normalize_unit_price_for_workload(
        unit_price_usd=0.02,
        unit_name="image",
        workload_type="image_generation",
    )
    assert price == pytest.approx(20.0)


def test_normalize_unknown_unit_returns_none() -> None:
    price = normalize_unit_price_for_workload(
        unit_price_usd=1.0,
        unit_name="gpu_hour",
        workload_type="speech_to_text",
    )
    assert price is None


def test_rank_catalog_offers_counts_exclusions_and_sorts() -> None:
    rows = [
        SimpleNamespace(
            provider="a",
            unit_name="audio_min",
            unit_price_usd=0.25,
            confidence="high",
            sku_name="a-stt",
            billing_mode="per_unit",
        ),
        SimpleNamespace(
            provider="a",
            unit_name="gpu_hour",
            unit_price_usd=2.0,
            confidence="high",
            sku_name="a-gpu",
            billing_mode="hourly",
        ),
        SimpleNamespace(
            provider="b",
            unit_name="audio_min",
            unit_price_usd=0.20,
            confidence="estimated",
            sku_name="b-stt",
            billing_mode="per_unit",
        ),
    ]
    ranked, reasons, excluded = rank_catalog_offers(
        rows=rows,
        allowed_providers={"a", "b"},
        unit_name=None,
        top_k=5,
        monthly_budget_max_usd=0.0,
        comparator_mode="normalized",
        confidence_weighted=True,
        workload_type="speech_to_text",
        monthly_usage=10.0,
    )
    assert excluded == 1  # gpu_hour row excluded as non-comparable
    assert reasons["a"].startswith("Included")
    assert reasons["b"].startswith("Included")
    assert [row.provider for row in ranked] == ["b", "a"]  # lower weighted comparator first
    assert ranked[0].monthly_estimate_usd is not None


def test_build_provider_diagnostics_included_excluded() -> None:
    diagnostics = build_provider_diagnostics(
        workload_provider_ids=["a", "b", "c"],
        selected_global_providers=["a", "c"],
        provider_reasons={"a": "Included (1 rankable offers).", "c": "No rankable offers after filters."},
    )
    by_provider = {row["provider"]: row for row in diagnostics}
    assert by_provider["a"]["status"] == "included"
    assert by_provider["b"]["reason"] == "Not selected by user."
    assert by_provider["c"]["status"] == "excluded"

