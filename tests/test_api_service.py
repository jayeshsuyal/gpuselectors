from __future__ import annotations

import pytest
from pydantic import ValidationError

from inference_atlas.api_models import (
    AIAssistContext,
    AIAssistRequest,
    CatalogRankingRequest,
    CopilotTurnRequest,
    LLMPlanningRequest,
)
from inference_atlas.api_service import (
    run_ai_assist,
    run_browse_catalog,
    run_copilot_turn,
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
