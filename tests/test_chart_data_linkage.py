from __future__ import annotations

from inference_atlas import get_catalog_v2_rows, rank_catalog_offers, rank_configs


def test_llm_ranked_plans_expose_chart_core_fields() -> None:
    plans = rank_configs(
        tokens_per_day=2_000_000,
        model_bucket="7b",
        top_k=3,
    )
    assert plans, "Expected at least one LLM plan for chart regression test"
    for plan in plans:
        assert isinstance(plan.monthly_cost_usd, float)
        assert plan.monthly_cost_usd >= 0
        assert isinstance(plan.risk.total_risk, float)
        assert 0 <= plan.risk.total_risk <= 1


def test_catalog_rows_expose_price_delta_fields() -> None:
    rows = get_catalog_v2_rows()
    assert rows, "Expected catalog rows to exist"
    sample = rows[0]
    assert hasattr(sample, "previous_unit_price_usd")
    assert hasattr(sample, "price_change_abs_usd")
    assert hasattr(sample, "price_change_pct")

    # Ensure at least one row has baseline pricing metadata for change charts.
    assert any(row.previous_unit_price_usd is not None for row in rows)


def test_non_llm_ranked_offers_keep_delta_fields_for_charts() -> None:
    rows = get_catalog_v2_rows("speech_to_text")
    assert rows, "Expected speech_to_text rows for ranking"
    providers = sorted({row.provider for row in rows})[:5]
    ranked, _, _ = rank_catalog_offers(
        rows=rows,
        allowed_providers=set(providers),
        unit_name=None,
        top_k=10,
        monthly_budget_max_usd=0.0,
        comparator_mode="normalized",
        confidence_weighted=True,
        workload_type="speech_to_text",
        monthly_usage=1000.0,
    )
    assert ranked, "Expected ranked non-LLM offers for chart regression test"
    for offer in ranked:
        assert isinstance(offer.provider, str) and offer.provider
        assert hasattr(offer, "monthly_estimate_usd")
        assert hasattr(offer, "price_change_abs_usd")
        assert hasattr(offer, "price_change_pct")
