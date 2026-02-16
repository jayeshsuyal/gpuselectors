from __future__ import annotations

import pytest

from inference_atlas.mvp_planner import (
    CapacityEstimate,
    PlannerConfig,
    capacity,
    compute_monthly_cost,
    enumerate_configs,
    enumerate_configs_for_providers,
    get_provider_compatibility,
    normalize_workload,
    rank_configs,
    risk_score,
)


def test_normalize_workload_math() -> None:
    wl = normalize_workload(tokens_per_day=8_640_000, peak_to_avg=3.0, util_target=0.75)
    assert wl.avg_tok_s == pytest.approx(100.0)
    assert wl.peak_tok_s == pytest.approx(300.0)
    assert wl.required_capacity_tok_s == pytest.approx(400.0)
    assert wl.tokens_per_month == pytest.approx(259_200_000.0)


def test_capacity_scaling_efficiency() -> None:
    cap = capacity(model_bucket="70b", gpu_type="a100_80gb", gpus=4, beta=0.08)
    expected_eff = 1 / (1 + 0.08 * 3)
    assert cap.efficiency == pytest.approx(expected_eff)
    assert cap.tok_s_per_gpu == pytest.approx(cap.tok_s_total / 4)


def test_compute_monthly_cost_per_token() -> None:
    wl = normalize_workload(tokens_per_day=1_000_000)
    cfg = PlannerConfig(
        provider_id="x",
        provider_name="X",
        offering_id="x_per_token",
        billing_mode="per_token",
        gpu_type=None,
        gpu_count=0,
        price_per_gpu_hour_usd=None,
        price_per_1m_tokens_usd=2.0,
        tps_cap=None,
        region="global",
        confidence="estimated",
        notes="",
    )
    assert compute_monthly_cost(cfg, wl, cap=None) == pytest.approx(60.0)


def test_compute_monthly_cost_autoscale() -> None:
    wl = normalize_workload(tokens_per_day=8_640_000)
    cfg = PlannerConfig(
        provider_id="x",
        provider_name="X",
        offering_id="x_auto",
        billing_mode="autoscale_hourly",
        gpu_type="a100_80gb",
        gpu_count=2,
        price_per_gpu_hour_usd=2.0,
        price_per_1m_tokens_usd=None,
        tps_cap=None,
        region="global",
        confidence="estimated",
        notes="",
    )
    cap_est = CapacityEstimate(
        tok_s_total=200.0,
        tok_s_per_gpu=100.0,
        efficiency=1.0,
        p95_latency_est_ms=None,
        mem_ok=True,
    )
    monthly = compute_monthly_cost(cfg, wl, cap=cap_est, autoscale_inefficiency=1.2)
    expected = (((wl.tokens_per_month / 200.0) / 3600.0) * 2) * 2.0 * 1.2
    assert monthly == pytest.approx(expected)


def test_risk_score_range() -> None:
    cfg = PlannerConfig(
        provider_id="x",
        provider_name="X",
        offering_id="x_ded",
        billing_mode="dedicated_hourly",
        gpu_type="a100_80gb",
        gpu_count=8,
        price_per_gpu_hour_usd=1.0,
        price_per_1m_tokens_usd=None,
        tps_cap=None,
        region="global",
        confidence="estimated",
        notes="",
    )
    risk = risk_score(cfg, required_capacity_tok_s=1000.0, provided_capacity_tok_s=1200.0)
    assert 0 <= risk.risk_overload <= 1
    assert 0 <= risk.risk_complexity <= 1
    assert 0 <= risk.total_risk <= 1


def test_enumerate_configs_has_matches() -> None:
    rows = enumerate_configs(model_bucket="70b")
    assert rows
    assert any(row.billing_mode in {"dedicated_hourly", "autoscale_hourly"} for row in rows)


def test_enumerate_configs_blends_io_pairs_for_llm_rows() -> None:
    rows = enumerate_configs(model_bucket="7b")
    ids = {row.offering_id for row in rows}
    assert "fireworks_deepseek_v3_blended_io" in ids
    assert "fireworks_deepseek_v3_input" not in ids
    assert "fireworks_deepseek_v3_output" not in ids


def test_output_token_ratio_changes_blended_price() -> None:
    low_ratio_rows = enumerate_configs(model_bucket="7b", output_token_ratio=0.1)
    high_ratio_rows = enumerate_configs(model_bucket="7b", output_token_ratio=0.9)

    def _price(rows: list[PlannerConfig], offering_id: str) -> float:
        row = next(item for item in rows if item.offering_id == offering_id)
        assert row.price_per_1m_tokens_usd is not None
        return row.price_per_1m_tokens_usd

    target = "fireworks_deepseek_v3_blended_io"
    assert _price(low_ratio_rows, target) < _price(high_ratio_rows, target)


def test_enumerate_configs_provider_subset_filters() -> None:
    rows = enumerate_configs_for_providers(
        model_bucket="70b",
        provider_ids={"baseten", "together_ai"},
    )
    assert rows
    assert all(row.provider_id in {"baseten", "together_ai"} for row in rows)


def test_provider_compatibility_reports_supported_and_unsupported() -> None:
    diag = get_provider_compatibility(
        model_bucket="70b",
        provider_ids={"baseten", "cohere"},
    )
    by_id = {row.provider_id: row for row in diag}
    assert by_id["baseten"].compatible is True
    assert by_id["cohere"].compatible is False
    assert "No offering for selected model bucket" in by_id["cohere"].reason


def test_rank_configs_returns_sorted_results() -> None:
    plans = rank_configs(tokens_per_day=8_000_000, model_bucket="70b", top_k=5)
    assert 1 <= len(plans) <= 5
    scores = [row.score for row in plans]
    assert scores == sorted(scores)
    assert all(
        ("required" in row.why) or ("throughput cap unspecified" in row.why)
        for row in plans
    )
