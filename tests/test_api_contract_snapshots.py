from __future__ import annotations

import json
from pathlib import Path

from inference_atlas.api_models import CatalogRankingRequest, LLMPlanningRequest, ReportGenerateRequest
from inference_atlas.api_service import (
    run_browse_catalog,
    run_generate_report,
    run_invoice_analyze,
    run_plan_llm,
    run_rank_catalog,
)

GOLDEN_DIR = Path(__file__).parent / "golden" / "api"


def _golden(name: str) -> dict[str, object]:
    path = GOLDEN_DIR / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_catalog(body: dict[str, object]) -> dict[str, object]:
    rows = body.get("rows") or []
    first = rows[0] if rows else {}
    return {
        "keys": sorted(body.keys()),
        "row_keys": sorted(first.keys()),
    }


def _normalize_rank_catalog(body: dict[str, object]) -> dict[str, object]:
    offers = body.get("offers") or []
    diagnostics = body.get("provider_diagnostics") or []
    trace = body.get("relaxation_steps") or []
    exclusion_breakdown = body.get("exclusion_breakdown") or {}
    first_offer = offers[0] if offers else {}
    first_diag = diagnostics[0] if diagnostics else {}
    first_trace = trace[0] if trace else {}
    return {
        "keys": sorted(body.keys()),
        "offer_keys": sorted(first_offer.keys()),
        "provider_diag_keys": sorted(first_diag.keys()),
        "relaxation_step_keys": sorted(first_trace.keys()),
        "exclusion_breakdown_keys": sorted(exclusion_breakdown.keys()),
    }


def _normalize_plan_llm(body: dict[str, object]) -> dict[str, object]:
    plans = body.get("plans") or []
    first_plan = plans[0] if plans else {}
    assumptions = first_plan.get("assumptions") if first_plan else {}
    return {
        "keys": sorted(body.keys()),
        "plan_keys": sorted(first_plan.keys()),
        "assumption_keys": sorted(assumptions.keys()) if assumptions else [],
    }


def _normalize_report(body: dict[str, object]) -> dict[str, object]:
    chart_data = body.get("chart_data") or {}
    metadata = body.get("metadata") or {}
    return {
        "keys": sorted(body.keys()),
        "mode": body.get("mode"),
        "chart_data_keys": sorted(chart_data.keys()),
        "metadata_keys": sorted(metadata.keys()),
    }


def _normalize_invoice(body: dict[str, object]) -> dict[str, object]:
    items = body.get("line_items") or []
    opportunities = body.get("savings_opportunities") or []
    first_item = items[0] if items else {}
    first_opportunity = opportunities[0] if opportunities else {}
    return {
        "keys": sorted(body.keys()),
        "line_item_keys": sorted(first_item.keys()),
        "savings_opportunity_keys": sorted(first_opportunity.keys()),
    }


def test_catalog_contract_snapshot() -> None:
    body = run_browse_catalog(workload_type="llm").model_dump(mode="json")
    assert _normalize_catalog(body) == _golden("catalog_llm")


def test_rank_catalog_contract_snapshot() -> None:
    body = run_rank_catalog(
        CatalogRankingRequest(
            workload_type="speech_to_text",
            allowed_providers=[],
            unit_name=None,
            monthly_usage=100.0,
            monthly_budget_max_usd=0.0,
            top_k=5,
            confidence_weighted=True,
            comparator_mode="normalized",
            throughput_aware=False,
            peak_to_avg=2.5,
            util_target=0.75,
            strict_capacity_check=False,
        )
    ).model_dump(mode="json")
    assert _normalize_rank_catalog(body) == _golden("rank_catalog")


def test_plan_llm_contract_snapshot() -> None:
    body = run_plan_llm(
        LLMPlanningRequest(
            tokens_per_day=5_000_000,
            model_bucket="7b",
            provider_ids=[],
            peak_to_avg=2.5,
            util_target=0.75,
            beta=0.08,
            alpha=1.0,
            autoscale_inefficiency=1.15,
            monthly_budget_max_usd=0.0,
            output_token_ratio=0.3,
            top_k=3,
        )
    ).model_dump(mode="json")
    assert _normalize_plan_llm(body) == _golden("plan_llm")


def test_report_generate_contract_snapshot() -> None:
    ranking = run_rank_catalog(
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
    body = run_generate_report(
        ReportGenerateRequest(mode="catalog", title="API Report", catalog_ranking=ranking)
    ).model_dump(mode="json")
    assert _normalize_report(body) == _golden("report_generate")


def test_invoice_analyze_contract_snapshot() -> None:
    body = run_invoice_analyze(
        (
            "provider,workload_type,usage_qty,usage_unit,amount_usd\n"
            "openai,speech_to_text,120,audio_min,12.0\n"
        ).encode("utf-8")
    ).model_dump(mode="json")
    assert _normalize_invoice(body) == _golden("invoice_analyze")
