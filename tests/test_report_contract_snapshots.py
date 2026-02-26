from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from inference_atlas.api_models import CatalogRankingRequest, LLMPlanningRequest, ReportGenerateRequest
from inference_atlas.api_service import run_generate_report, run_plan_llm, run_rank_catalog


SNAPSHOT_DIR = Path(__file__).parent / "snapshots"


def _load_snapshot(name: str) -> dict[str, Any]:
    path = SNAPSHOT_DIR / name
    return json.loads(path.read_text(encoding="utf-8"))


def _snapshotify_report_payload(payload: dict[str, Any]) -> dict[str, Any]:
    sections = payload.get("sections") or []
    chart_data = payload.get("chart_data") or {}
    metadata = payload.get("metadata") or {}
    csv_exports = payload.get("csv_exports") or {}

    first_section = sections[0] if sections else {}
    first_cost_row = None
    if isinstance(chart_data.get("cost_by_rank"), list) and chart_data["cost_by_rank"]:
        first_cost_row = chart_data["cost_by_rank"][0]

    first_risk_row = None
    if isinstance(chart_data.get("risk_breakdown"), list) and chart_data["risk_breakdown"]:
        first_risk_row = chart_data["risk_breakdown"][0]

    return {
        "top_keys": sorted(payload.keys()),
        "mode": payload.get("mode"),
        "output_format": payload.get("output_format"),
        "section_count": len(sections),
        "section_keys": sorted(first_section.keys()) if isinstance(first_section, dict) else [],
        "chart_data_keys": sorted(chart_data.keys()) if isinstance(chart_data, dict) else [],
        "first_cost_row_keys": sorted(first_cost_row.keys()) if isinstance(first_cost_row, dict) else [],
        "first_risk_row_keys": sorted(first_risk_row.keys()) if isinstance(first_risk_row, dict) else [],
        "metadata_keys": sorted(metadata.keys()) if isinstance(metadata, dict) else [],
        "csv_export_keys": sorted(csv_exports.keys()) if isinstance(csv_exports, dict) else [],
        "has_markdown": bool(payload.get("markdown")),
        "has_html": payload.get("html") is not None,
        "has_pdf_base64": payload.get("pdf_base64") is not None,
        "has_narrative": payload.get("narrative") is not None,
    }


def test_report_contract_snapshot_llm_markdown() -> None:
    llm = run_plan_llm(
        LLMPlanningRequest(
            tokens_per_day=3_000_000,
            model_bucket="7b",
            provider_ids=[],
            top_k=3,
        )
    )
    response = run_generate_report(
        ReportGenerateRequest(
            mode="llm",
            output_format="markdown",
            include_charts=True,
            include_csv_exports=True,
            include_narrative=True,
            llm_planning=llm,
        )
    )
    snap = _snapshotify_report_payload(response.model_dump())
    assert snap == _load_snapshot("report_llm_markdown_contract.json")


def test_report_contract_snapshot_catalog_pdf() -> None:
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
    response = run_generate_report(
        ReportGenerateRequest(
            mode="catalog",
            output_format="pdf",
            include_charts=True,
            include_csv_exports=True,
            include_narrative=True,
            catalog_ranking=catalog,
        )
    )
    snap = _snapshotify_report_payload(response.model_dump())
    assert snap == _load_snapshot("report_catalog_pdf_contract.json")
