from __future__ import annotations

from types import SimpleNamespace

import pytest

from inference_atlas.catalog_ranking import (
    build_provider_diagnostics,
    get_catalog_tuning_preset,
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
    assert ranked[0].score <= ranked[1].score


def test_rank_catalog_offers_throughput_aware_with_replicas() -> None:
    rows = [
        SimpleNamespace(
            provider="a",
            unit_name="audio_hour",
            unit_price_usd=1.0,
            confidence="high",
            sku_name="a-stt",
            billing_mode="per_unit",
            throughput_value=1.0,
            throughput_unit="per_hour",
        ),
        SimpleNamespace(
            provider="b",
            unit_name="audio_hour",
            unit_price_usd=0.8,
            confidence="high",
            sku_name="b-stt",
            billing_mode="per_unit",
            throughput_value=20.0,
            throughput_unit="per_hour",
        ),
    ]
    ranked, _, _ = rank_catalog_offers(
        rows=rows,
        allowed_providers={"a", "b"},
        unit_name="audio_hour",
        top_k=5,
        monthly_budget_max_usd=0.0,
        comparator_mode="normalized",
        confidence_weighted=False,
        workload_type="speech_to_text",
        monthly_usage=720.0,
        throughput_aware=True,
        peak_to_avg=2.5,
        util_target=0.75,
    )
    assert ranked[0].provider == "b"
    assert ranked[0].required_replicas == 1
    assert ranked[1].required_replicas is not None
    assert ranked[1].required_replicas > ranked[0].required_replicas


def test_rank_catalog_offers_strict_capacity_check_excludes_missing_throughput() -> None:
    rows = [
        SimpleNamespace(
            provider="a",
            unit_name="audio_hour",
            unit_price_usd=1.0,
            confidence="high",
            sku_name="a-stt",
            billing_mode="per_unit",
            throughput_value=None,
            throughput_unit=None,
        ),
        SimpleNamespace(
            provider="b",
            unit_name="audio_hour",
            unit_price_usd=1.2,
            confidence="high",
            sku_name="b-stt",
            billing_mode="per_unit",
            throughput_value=50.0,
            throughput_unit="per_hour",
        ),
    ]
    ranked, reasons, excluded = rank_catalog_offers(
        rows=rows,
        allowed_providers={"a", "b"},
        unit_name="audio_hour",
        top_k=5,
        monthly_budget_max_usd=0.0,
        comparator_mode="normalized",
        confidence_weighted=False,
        workload_type="speech_to_text",
        monthly_usage=100.0,
        throughput_aware=True,
        peak_to_avg=2.5,
        util_target=0.75,
        strict_capacity_check=True,
    )
    assert excluded >= 1
    assert [row.provider for row in ranked] == ["b"]
    assert reasons["a"].startswith("Excluded by strict capacity check")


def test_monthly_budget_filter_uses_listed_unit_price_not_normalized_comparator() -> None:
    rows = [
        SimpleNamespace(
            provider="a",
            unit_name="audio_min",
            unit_price_usd=0.2,
            confidence="high",
            sku_name="a-stt",
            billing_mode="per_unit",
            throughput_value=None,
            throughput_unit=None,
        ),
    ]
    ranked, _, _ = rank_catalog_offers(
        rows=rows,
        allowed_providers={"a"},
        unit_name="audio_min",
        top_k=5,
        monthly_budget_max_usd=5.0,
        comparator_mode="normalized",
        confidence_weighted=False,
        workload_type="speech_to_text",
        monthly_usage=10.0,
    )
    assert len(ranked) == 1
    assert ranked[0].monthly_estimate_usd == pytest.approx(2.0)

    ranked_tight_budget, _, _ = rank_catalog_offers(
        rows=rows,
        allowed_providers={"a"},
        unit_name="audio_min",
        top_k=5,
        monthly_budget_max_usd=1.0,
        comparator_mode="normalized",
        confidence_weighted=False,
        workload_type="speech_to_text",
        monthly_usage=10.0,
    )
    assert ranked_tight_budget == []


def test_normalized_mode_with_specific_unit_falls_back_to_same_unit_price() -> None:
    rows = [
        SimpleNamespace(
            provider="vision_a",
            unit_name="video_min",
            unit_price_usd=20.0,
            confidence="high",
            sku_name="vision-a",
            billing_mode="per_minute",
            throughput_value=None,
            throughput_unit=None,
        ),
        SimpleNamespace(
            provider="vision_b",
            unit_name="video_min",
            unit_price_usd=30.0,
            confidence="high",
            sku_name="vision-b",
            billing_mode="per_minute",
            throughput_value=None,
            throughput_unit=None,
        ),
    ]
    ranked, reasons, excluded = rank_catalog_offers(
        rows=rows,
        allowed_providers={"vision_a", "vision_b"},
        unit_name="video_min",
        top_k=5,
        monthly_budget_max_usd=500.0,
        comparator_mode="normalized",
        confidence_weighted=False,
        workload_type="vision",
        monthly_usage=4.0,
        throughput_aware=True,
        strict_capacity_check=False,
    )
    assert excluded == 0
    assert [r.provider for r in ranked] == ["vision_a", "vision_b"]
    assert reasons["vision_a"].startswith("Included")


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


def test_rank_catalog_offers_validates_parameters() -> None:
    rows = [
        SimpleNamespace(
            provider="a",
            unit_name="audio_min",
            unit_price_usd=0.2,
            confidence="high",
            sku_name="a-stt",
            billing_mode="per_unit",
            throughput_value=None,
            throughput_unit=None,
        ),
    ]
    with pytest.raises(ValueError, match="top_k"):
        rank_catalog_offers(
            rows=rows,
            allowed_providers={"a"},
            unit_name=None,
            top_k=0,
            monthly_budget_max_usd=0.0,
            comparator_mode="normalized",
            confidence_weighted=False,
            workload_type="speech_to_text",
        )
    with pytest.raises(ValueError, match="comparator_mode"):
        rank_catalog_offers(
            rows=rows,
            allowed_providers={"a"},
            unit_name=None,
            top_k=1,
            monthly_budget_max_usd=0.0,
            comparator_mode="invalid",
            confidence_weighted=False,
            workload_type="speech_to_text",
        )
    with pytest.raises(ValueError, match="alpha"):
        rank_catalog_offers(
            rows=rows,
            allowed_providers={"a"},
            unit_name=None,
            top_k=1,
            monthly_budget_max_usd=0.0,
            comparator_mode="normalized",
            confidence_weighted=False,
            workload_type="speech_to_text",
            alpha=-0.1,
        )


def test_catalog_tuning_preset_balanced_default() -> None:
    preset = get_catalog_tuning_preset("vision", "balanced")
    assert preset["peak_to_avg"] == pytest.approx(2.5)
    assert preset["util_target"] == pytest.approx(0.75)
    assert preset["alpha"] == pytest.approx(1.0)


def test_catalog_tuning_preset_workload_specific() -> None:
    stt = get_catalog_tuning_preset("speech_to_text", "conservative")
    assert stt["peak_to_avg"] == pytest.approx(3.5)
    assert stt["util_target"] == pytest.approx(0.65)
    assert stt["alpha"] == pytest.approx(1.5)


def test_catalog_tuning_preset_invalid_name_raises() -> None:
    with pytest.raises(ValueError, match="Unknown preset"):
        get_catalog_tuning_preset("speech_to_text", "not-a-preset")


def test_catalog_tuning_preset_unknown_workload_uses_default_group() -> None:
    preset = get_catalog_tuning_preset("unknown_workload", "balanced")
    assert preset["peak_to_avg"] == pytest.approx(2.5)
    assert preset["util_target"] == pytest.approx(0.75)
    assert preset["alpha"] == pytest.approx(1.0)


def test_rank_catalog_offers_accepts_raw_as_listed_mode() -> None:
    rows = [
        SimpleNamespace(
            provider="a",
            unit_name="audio_min",
            unit_price_usd=0.2,
            confidence="high",
            sku_name="a-stt",
            billing_mode="per_unit",
            throughput_value=None,
            throughput_unit=None,
        ),
        SimpleNamespace(
            provider="b",
            unit_name="audio_min",
            unit_price_usd=0.3,
            confidence="high",
            sku_name="b-stt",
            billing_mode="per_unit",
            throughput_value=None,
            throughput_unit=None,
        ),
    ]
    listed, _, _ = rank_catalog_offers(
        rows=rows,
        allowed_providers={"a", "b"},
        unit_name=None,
        top_k=5,
        monthly_budget_max_usd=0.0,
        comparator_mode="listed",
        confidence_weighted=False,
        workload_type="speech_to_text",
    )
    raw, _, _ = rank_catalog_offers(
        rows=rows,
        allowed_providers={"a", "b"},
        unit_name=None,
        top_k=5,
        monthly_budget_max_usd=0.0,
        comparator_mode="raw",
        confidence_weighted=False,
        workload_type="speech_to_text",
    )
    assert [r.provider for r in listed] == [r.provider for r in raw]


def test_rank_catalog_offers_tie_break_is_deterministic() -> None:
    rows = [
        SimpleNamespace(
            provider="b",
            unit_name="audio_min",
            unit_price_usd=0.2,
            confidence="high",
            sku_name="same",
            billing_mode="per_unit",
            throughput_value=None,
            throughput_unit=None,
        ),
        SimpleNamespace(
            provider="a",
            unit_name="audio_min",
            unit_price_usd=0.2,
            confidence="high",
            sku_name="same",
            billing_mode="per_unit",
            throughput_value=None,
            throughput_unit=None,
        ),
    ]
    ranked, _, _ = rank_catalog_offers(
        rows=rows,
        allowed_providers={"a", "b"},
        unit_name=None,
        top_k=5,
        monthly_budget_max_usd=0.0,
        comparator_mode="listed",
        confidence_weighted=False,
        workload_type="speech_to_text",
    )
    assert [r.provider for r in ranked] == ["a", "b"]


def test_rank_catalog_offers_carries_price_delta_fields() -> None:
    rows = [
        SimpleNamespace(
            provider="a",
            unit_name="audio_hour",
            unit_price_usd=1.0,
            confidence="high",
            sku_name="a-stt",
            billing_mode="per_unit",
            throughput_value=None,
            throughput_unit=None,
            previous_unit_price_usd=1.2,
            price_change_abs_usd=-0.2,
            price_change_pct=-16.6667,
        ),
    ]
    ranked, _, _ = rank_catalog_offers(
        rows=rows,
        allowed_providers={"a"},
        unit_name="audio_hour",
        top_k=3,
        monthly_budget_max_usd=0.0,
        comparator_mode="normalized",
        confidence_weighted=False,
        workload_type="speech_to_text",
    )
    assert ranked
    assert ranked[0].previous_unit_price_usd == pytest.approx(1.2)
    assert ranked[0].price_change_abs_usd == pytest.approx(-0.2)
    assert ranked[0].price_change_pct == pytest.approx(-16.6667)
