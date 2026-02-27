from __future__ import annotations

import pytest
from pydantic import ValidationError

from inference_atlas.api_models import (
    AIAssistContext,
    AIAssistRequest,
    CatalogRankingRequest,
    CopilotTurnRequest,
    LLMPlanningRequest,
    ReportGenerateRequest,
    ScalingPlanRequest,
)
from inference_atlas.api_service import (
    run_ai_assist,
    run_browse_catalog,
    run_copilot_turn,
    run_generate_report,
    run_invoice_analyze,
    run_plan_llm,
    run_plan_scaling,
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
    assert [chart.id for chart in report.charts] == [
        "cost_comparison",
        "risk_comparison",
        "confidence_distribution",
        "fallback_trace",
    ]
    assert set(report.chart_data) >= {"cost_by_rank", "risk_breakdown", "confidence_distribution"}
    assert isinstance(report.chart_data["cost_by_rank"], list)
    assert set(report.metadata) >= {
        "chart_schema_version",
        "catalog_generated_at_utc",
        "catalog_schema_version",
        "catalog_row_count",
        "catalog_providers_synced_count",
    }
    assert report.output_format == "markdown"
    assert report.narrative is None
    assert "ranked_results.csv" in report.csv_exports
    assert "provider_diagnostics.csv" in report.csv_exports
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
    assert [chart.id for chart in report.charts] == [
        "cost_comparison",
        "risk_comparison",
        "confidence_distribution",
        "fallback_trace",
    ]
    assert set(report.chart_data) >= {
        "cost_by_rank",
        "exclusion_breakdown",
        "relaxation_trace",
        "confidence_distribution",
    }
    assert set(report.metadata) >= {
        "chart_schema_version",
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
    assert report.output_format == "markdown"
    assert report.narrative is None
    assert "ranked_results.csv" in report.csv_exports
    assert "provider_diagnostics.csv" in report.csv_exports


def test_run_generate_report_can_disable_charts_and_csv_exports() -> None:
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
            title="No Chart Report",
            include_charts=False,
            include_csv_exports=False,
            catalog_ranking=catalog,
        )
    )
    assert report.charts == []
    assert report.chart_data == {}
    assert report.csv_exports == {}


def test_run_generate_report_supports_html_and_pdf_output() -> None:
    llm = run_plan_llm(
        LLMPlanningRequest(
            tokens_per_day=3_000_000,
            model_bucket="7b",
            provider_ids=[],
            top_k=2,
        )
    )
    html_report = run_generate_report(
        ReportGenerateRequest(
            mode="llm",
            title="HTML Report",
            output_format="html",
            llm_planning=llm,
        )
    )
    assert html_report.output_format == "html"
    assert html_report.html is not None
    assert "<html" in html_report.html.lower()
    assert html_report.pdf_base64 is None

    pdf_report = run_generate_report(
        ReportGenerateRequest(
            mode="llm",
            title="PDF Report",
            output_format="pdf",
            llm_planning=llm,
        )
    )
    assert pdf_report.output_format == "pdf"
    assert pdf_report.pdf_base64 is not None
    assert len(pdf_report.pdf_base64) > 20


def test_run_generate_report_can_include_narrative() -> None:
    llm = run_plan_llm(
        LLMPlanningRequest(
            tokens_per_day=3_000_000,
            model_bucket="7b",
            provider_ids=[],
            top_k=2,
        )
    )
    report = run_generate_report(
        ReportGenerateRequest(
            mode="llm",
            title="Narrative Report",
            include_narrative=True,
            llm_planning=llm,
        )
    )
    assert report.narrative is not None
    assert "Primary recommendation:" in report.narrative


def test_run_plan_scaling_llm_returns_mode_and_gpu_estimate() -> None:
    llm = run_plan_llm(
        LLMPlanningRequest(
            tokens_per_day=3_000_000,
            model_bucket="7b",
            provider_ids=[],
            top_k=3,
        )
    )
    response = run_plan_scaling(ScalingPlanRequest(mode="llm", llm_planning=llm))
    assert response.mode == "llm"
    assert response.deployment_mode in {"serverless", "autoscale", "dedicated", "unknown"}
    assert response.estimated_gpu_count >= 0
    assert response.risk_band in {"low", "medium", "high", "unknown"}


def test_run_plan_scaling_catalog_returns_capacity_guidance() -> None:
    catalog = run_rank_catalog(
        CatalogRankingRequest(
            workload_type="speech_to_text",
            allowed_providers=[],
            unit_name=None,
            monthly_usage=20.0,
            top_k=3,
            confidence_weighted=True,
            comparator_mode="normalized",
            throughput_aware=True,
        )
    )
    response = run_plan_scaling(ScalingPlanRequest(mode="catalog", catalog_ranking=catalog))
    assert response.mode == "catalog"
    assert response.capacity_check in {"ok", "insufficient", "unknown"}
    assert response.estimated_gpu_count >= 0


def test_run_generate_report_includes_scaling_summary_section() -> None:
    llm = run_plan_llm(
        LLMPlanningRequest(
            tokens_per_day=3_000_000,
            model_bucket="7b",
            provider_ids=[],
            top_k=2,
        )
    )
    report = run_generate_report(
        ReportGenerateRequest(
            mode="llm",
            title="Scaling Summary Report",
            llm_planning=llm,
        )
    )
    titles = [section.title for section in report.sections]
    assert "Scaling Summary" in titles
