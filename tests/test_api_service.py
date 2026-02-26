from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from inference_atlas.api_models import (
    AIAssistContext,
    AIAssistRequest,
    CatalogRankingRequest,
    CopilotTurnRequest,
    LLMPlanningRequest,
    ReportGenerateRequest,
)
from inference_atlas.api_service import (
    run_ai_assist,
    run_browse_catalog,
    run_copilot_turn,
    run_generate_report,
    run_invoice_analyze,
    run_plan_llm,
    run_rank_catalog,
)


def test_run_copilot_turn_returns_valid_response() -> None:
    payload = CopilotTurnRequest(user_text="Need llm with 5 million tokens/day and $300 budget")
    response = run_copilot_turn(payload)
    assert response.extracted_spec["workload_type"] == "llm"
    assert response.extracted_spec["tokens_per_day"] == 5_000_000
    assert response.extracted_spec["monthly_budget_max_usd"] == 300
    assert isinstance(response.follow_up_questions, list)
    assert response.apply_payload is None or isinstance(response.apply_payload, dict)


def test_run_copilot_turn_merges_prior_state() -> None:
    first = run_copilot_turn(CopilotTurnRequest(user_text="Need speech to text under $50"))
    second = run_copilot_turn(
        CopilotTurnRequest(
            user_text="monthly usage 4000 and strict latency",
            state={
                "messages": [{"role": "user", "content": "Need speech to text under $50"}],
                "extracted_spec": first.extracted_spec,
            },
        )
    )
    assert second.extracted_spec["workload_type"] == "speech_to_text"
    assert second.extracted_spec["monthly_budget_max_usd"] == 50
    assert second.extracted_spec["monthly_usage"] == 4000
    assert second.extracted_spec["latency_priority"] == "strict"


def test_copilot_turn_request_validates_input() -> None:
    with pytest.raises(ValidationError):
        CopilotTurnRequest()


def test_run_copilot_turn_accepts_frontend_shape() -> None:
    payload = CopilotTurnRequest(
        message="Need vision with monthly usage 2000",
        history=[],
        workload_type="vision",
    )
    response = run_copilot_turn(payload)
    assert response.extracted_spec["workload_type"] == "vision"
    assert response.extracted_spec["monthly_usage"] == 2000


def test_run_plan_llm_returns_ranked_plans() -> None:
    response = run_plan_llm(
        LLMPlanningRequest(
            tokens_per_day=5_000_000,
            model_bucket="7b",
            provider_ids=[],
            top_k=3,
        )
    )
    assert response.plans
    assert response.plans[0].rank == 1
    assert response.plans[0].monthly_cost_usd > 0


def test_run_rank_catalog_returns_rows_for_non_llm() -> None:
    response = run_rank_catalog(
        CatalogRankingRequest(
            workload_type="speech_to_text",
            allowed_providers=[],
            unit_name=None,
            monthly_usage=100.0,
            top_k=5,
            confidence_weighted=True,
            comparator_mode="normalized",
            throughput_aware=False,
        )
    )
    assert response.excluded_count >= 0
    assert len(response.provider_diagnostics) >= 1
    assert isinstance(response.relaxation_steps, list)
    assert isinstance(response.exclusion_breakdown, dict)


def test_run_browse_catalog_filters_workload() -> None:
    response = run_browse_catalog(workload_type="llm")
    assert response.total >= 1
    assert all(row["workload_type"] == "llm" for row in response.rows)


def test_run_invoice_analyze_returns_line_items() -> None:
    csv_bytes = (
        "provider,workload_type,usage_qty,usage_unit,amount_usd\n"
        "openai,speech_to_text,120,audio_min,12.0\n"
    ).encode("utf-8")
    response = run_invoice_analyze(csv_bytes)
    assert response.grand_total > 0
    assert response.line_items


def test_run_ai_assist_returns_grounded_reply() -> None:
    response = run_ai_assist(
        AIAssistRequest(
            message="cheapest speech to text provider",
            context=AIAssistContext(workload_type="speech_to_text", providers=[]),
        )
    )
    assert "lowest current unit prices" in response.reply


def test_run_ai_assist_returns_alternatives_when_workload_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        SimpleNamespace(
            provider="openai",
            workload_type="llm",
            sku_name="gpt-4o-mini",
            model_key="gpt-4o-mini",
            billing_mode="per_token",
            unit_price_usd=0.15,
            unit_name="1m_tokens",
            region="global",
            confidence="official",
            source_kind="provider_csv",
        ),
        SimpleNamespace(
            provider="deepgram",
            workload_type="speech_to_text",
            sku_name="nova-2",
            model_key="nova-2",
            billing_mode="per_unit",
            unit_price_usd=0.0043,
            unit_name="audio_min",
            region="global",
            confidence="official",
            source_kind="provider_csv",
        ),
    ]

    monkeypatch.setattr("inference_atlas.api_service.get_catalog_v2_rows", lambda: rows)

    response = run_ai_assist(
        AIAssistRequest(
            message="give me text to speech options",
            context=AIAssistContext(workload_type="text_to_speech", providers=[]),
        )
    )
    assert "consider these alternatives" in response.reply.lower()
    assert "next best actions" in response.reply.lower()
    assert response.suggested_action is None


def test_run_ai_assist_returns_actionable_message_when_catalog_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("inference_atlas.api_service.get_catalog_v2_rows", lambda: [])

    response = run_ai_assist(
        AIAssistRequest(
            message="cheapest provider?",
            context=AIAssistContext(workload_type="llm", providers=[]),
        )
    )
    assert "don't have pricing rows loaded" in response.reply.lower()
    assert "broadening provider scope" in response.reply.lower()
    assert response.suggested_action is None


def test_run_rank_catalog_sets_relaxation_metadata_on_fallback() -> None:
    response = run_rank_catalog(
        CatalogRankingRequest(
            workload_type="vision",
            allowed_providers=[],
            unit_name="audio_min",
            monthly_usage=10.0,
            monthly_budget_max_usd=0.0,
            top_k=5,
            confidence_weighted=True,
            comparator_mode="listed",
            throughput_aware=False,
        )
    )
    assert response.relaxation_applied is True
    assert any(step["step"] == "relax_unit" for step in response.relaxation_steps)
    assert any("Applied fallback step '" in warning for warning in response.warnings)


def test_run_rank_catalog_response_contract_shape_is_stable() -> None:
    response = run_rank_catalog(
        CatalogRankingRequest(
            workload_type="speech_to_text",
            allowed_providers=[],
            unit_name=None,
            monthly_usage=0.0,
            monthly_budget_max_usd=0.0,
            top_k=3,
            confidence_weighted=True,
            comparator_mode="normalized",
            throughput_aware=False,
        )
    )
    payload = response.model_dump()
    assert set(payload) == {
        "offers",
        "provider_diagnostics",
        "excluded_count",
        "warnings",
        "relaxation_applied",
        "relaxation_steps",
        "exclusion_breakdown",
    }
    assert set(payload["exclusion_breakdown"]) >= {
        "provider_filtered_out",
        "unit_mismatch",
        "non_comparable_normalization",
        "missing_throughput",
        "budget",
    }
    if payload["offers"]:
        assert set(payload["offers"][0]) == {
            "rank",
            "provider",
            "sku_name",
            "billing_mode",
            "unit_price_usd",
            "normalized_price",
            "unit_name",
            "confidence",
            "monthly_estimate_usd",
            "required_replicas",
            "capacity_check",
            "previous_unit_price_usd",
            "price_change_abs_usd",
            "price_change_pct",
        }


def test_run_rank_catalog_relaxation_order_is_progressive() -> None:
    response = run_rank_catalog(
        CatalogRankingRequest(
            workload_type="vision",
            allowed_providers=[],
            unit_name="audio_min",
            monthly_usage=2.0,
            monthly_budget_max_usd=1.0,
            top_k=5,
            confidence_weighted=True,
            comparator_mode="listed",
            throughput_aware=False,
        )
    )
    attempted_steps = [step["step"] for step in response.relaxation_steps if step["attempted"]]
    assert attempted_steps[0] == "strict"
    assert attempted_steps == sorted(
        attempted_steps,
        key={"strict": 0, "relax_unit": 1, "relax_budget": 2, "relax_provider": 3}.get,
    )


def test_run_rank_catalog_includes_exclusion_summary_warning() -> None:
    response = run_rank_catalog(
        CatalogRankingRequest(
            workload_type="vision",
            allowed_providers=[],
            unit_name="audio_min",
            monthly_usage=10.0,
            monthly_budget_max_usd=0.0,
            top_k=5,
            confidence_weighted=True,
            comparator_mode="normalized",
            throughput_aware=True,
            strict_capacity_check=True,
        )
    )
    assert any(warning.startswith("Top exclusion reasons: ") for warning in response.warnings)


def test_run_rank_catalog_exclusion_summary_is_bounded() -> None:
    response = run_rank_catalog(
        CatalogRankingRequest(
            workload_type="vision",
            allowed_providers=[],
            unit_name=None,
            monthly_usage=1.0,
            monthly_budget_max_usd=1.0,
            top_k=5,
            confidence_weighted=True,
            comparator_mode="normalized",
            throughput_aware=True,
            strict_capacity_check=True,
        )
    )
    summary = next((w for w in response.warnings if w.startswith("Top exclusion reasons: ")), "")
    assert summary
    assert summary.count("(") <= 3


def test_run_generate_report_llm_mode_produces_markdown() -> None:
    llm = run_plan_llm(
        LLMPlanningRequest(
            tokens_per_day=3_000_000,
            model_bucket="7b",
            provider_ids=[],
            top_k=3,
        )
    )
    report = run_generate_report(
        ReportGenerateRequest(
            mode="llm",
            title="LLM Test Report",
            llm_planning=llm,
        )
    )
    assert report.mode == "llm"
    assert report.report_id.startswith("rep_")
    assert "LLM Test Report" in report.markdown
    assert "## Executive Summary" in report.markdown
    assert set(report.chart_data) >= {"cost_by_rank", "risk_breakdown", "confidence_distribution"}
    assert isinstance(report.chart_data["cost_by_rank"], list)
    assert set(report.metadata) >= {
        "catalog_generated_at_utc",
        "catalog_schema_version",
        "catalog_row_count",
        "catalog_providers_synced_count",
    }
    if report.chart_data["cost_by_rank"]:
        first = report.chart_data["cost_by_rank"][0]
        assert set(first) == {
            "rank",
            "provider_id",
            "provider_name",
            "monthly_cost_usd",
            "score",
            "total_risk",
        }
    if report.chart_data["risk_breakdown"]:
        first = report.chart_data["risk_breakdown"][0]
        assert set(first) == {
            "rank",
            "provider_id",
            "risk_overload",
            "risk_complexity",
            "total_risk",
        }


def test_run_generate_report_catalog_mode_produces_markdown() -> None:
    catalog = run_rank_catalog(
        CatalogRankingRequest(
            workload_type="speech_to_text",
            allowed_providers=[],
            unit_name=None,
            monthly_usage=20.0,
            top_k=3,
            confidence_weighted=True,
            comparator_mode="normalized",
            throughput_aware=False,
        )
    )
    report = run_generate_report(
        ReportGenerateRequest(
            mode="catalog",
            title="Catalog Test Report",
            catalog_ranking=catalog,
        )
    )
    assert report.mode == "catalog"
    assert "Catalog Test Report" in report.markdown
    assert "## Top Recommendations" in report.markdown
    assert set(report.chart_data) >= {
        "cost_by_rank",
        "exclusion_breakdown",
        "relaxation_trace",
        "confidence_distribution",
    }
    assert set(report.metadata) >= {
        "catalog_generated_at_utc",
        "catalog_schema_version",
        "catalog_row_count",
        "catalog_providers_synced_count",
    }
    if report.chart_data["cost_by_rank"]:
        first = report.chart_data["cost_by_rank"][0]
        assert set(first) == {
            "rank",
            "provider",
            "sku_name",
            "monthly_estimate_usd",
            "unit_price_usd",
            "unit_name",
        }
    assert isinstance(report.chart_data["exclusion_breakdown"], dict)
    assert isinstance(report.chart_data["relaxation_trace"], list)
