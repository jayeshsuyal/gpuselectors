"""Service-layer handlers for API endpoints."""

from __future__ import annotations

import csv
import base64
import hashlib
import html
import io
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

from inference_atlas.ai_copilot import next_copilot_turn
from inference_atlas.ai_inference import build_catalog_context, resolve_ai_scope
from inference_atlas.api_models import (
    AIAssistRequest,
    AIAssistResponse,
    CatalogBrowseResponse,
    CatalogRankingRequest,
    CatalogRankingResponse,
    CopilotTurnRequest,
    CopilotTurnResponse,
    InvoiceAnalysisResponse,
    InvoiceLineItem,
    LLMPlanningRequest,
    LLMPlanningResponse,
    ProviderDiagnostic,
    ReportGenerateRequest,
    ReportGenerateResponse,
    ScalingPlanRequest,
    ScalingPlanResponse,
    ReportChart,
    ReportChartSeries,
    ReportSection,
    RankedCatalogOffer,
    RankedPlan,
    RiskBreakdown,
)
from inference_atlas.catalog_ranking import (
    build_provider_diagnostics,
    run_catalog_ranking_with_relaxation,
)
from inference_atlas.data_loader import get_catalog_v2_metadata, get_catalog_v2_rows
from inference_atlas.invoice_analyzer import analyze_invoice_csv
from inference_atlas.mvp_planner import get_provider_compatibility, rank_configs


EXCLUSION_REASON_LABELS: dict[str, str] = {
    "provider_filtered_out": "provider filter",
    "unit_mismatch": "unit mismatch",
    "non_comparable_normalization": "non-comparable normalization",
    "missing_throughput": "missing throughput metadata",
    "budget": "budget filter",
}

CONFIDENCE_ORDER = [
    "official",
    "high",
    "medium",
    "estimated",
    "low",
    "vendor_list",
]


def _build_exclusion_summary_warnings(exclusion_breakdown: dict[str, int]) -> list[str]:
    ranked_reasons = sorted(
        (
            (reason, count)
            for reason, count in exclusion_breakdown.items()
            if count > 0
        ),
        key=lambda item: (-item[1], item[0]),
    )
    if not ranked_reasons:
        return []
    top_items = ranked_reasons[:3]
    rendered = ", ".join(
        f"{EXCLUSION_REASON_LABELS.get(reason, reason)} ({count})"
        for reason, count in top_items
    )
    return [f"Top exclusion reasons: {rendered}."]


def _normalize_state(payload: CopilotTurnRequest) -> tuple[str, dict[str, Any]]:
    if payload.user_text:
        return payload.user_text, dict(payload.state or {})

    user_text = str(payload.message or "")
    state = dict(payload.state or {})
    if not state:
        state["messages"] = [
            {"role": message.role, "content": message.content}
            for message in payload.history
        ]
        extracted = dict(state.get("extracted_spec") or {})
        if payload.workload_type:
            extracted["workload_type"] = payload.workload_type
        state["extracted_spec"] = extracted
    return user_text, state


def _to_frontend_apply_payload(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not payload:
        return None
    values = payload.get("values")
    if isinstance(values, dict):
        return dict(values)
    if isinstance(payload, dict):
        return dict(payload)
    return None


def _build_reply(result: dict[str, Any]) -> str:
    missing_fields = list(result.get("missing_fields") or [])
    if not missing_fields:
        return "Configuration is ready. Click Apply to Config and run optimization."
    follow_ups = list(result.get("next_questions") or [])
    if follow_ups:
        return "Got it. Quick follow-ups:\n- " + "\n- ".join(follow_ups)
    return "I need a bit more detail to proceed."


def run_copilot_turn(payload: CopilotTurnRequest) -> CopilotTurnResponse:
    """Run one IA copilot turn with validated input/output contracts."""
    user_text, state = _normalize_state(payload)
    result = next_copilot_turn(user_text=user_text, state=state)
    response = {
        "reply": _build_reply(result),
        "extracted_spec": result.get("extracted_spec", {}),
        "missing_fields": result.get("missing_fields", []),
        "follow_up_questions": result.get("next_questions", []),
        "apply_payload": _to_frontend_apply_payload(result.get("apply_payload")),
        "is_ready": bool(result.get("ready_to_rank", False)),
    }
    return CopilotTurnResponse.model_validate(response)


def run_plan_llm(payload: LLMPlanningRequest) -> LLMPlanningResponse:
    provider_ids = set(payload.provider_ids) if payload.provider_ids else None
    plans = rank_configs(
        tokens_per_day=payload.tokens_per_day,
        model_bucket=payload.model_bucket,
        peak_to_avg=payload.peak_to_avg,
        util_target=payload.util_target,
        beta=payload.beta,
        alpha=payload.alpha,
        autoscale_inefficiency=payload.autoscale_inefficiency,
        monthly_budget_max_usd=payload.monthly_budget_max_usd,
        top_k=payload.top_k,
        provider_ids=provider_ids,
        output_token_ratio=payload.output_token_ratio,
    )

    compat = get_provider_compatibility(
        model_bucket=payload.model_bucket,
        provider_ids=provider_ids,
        output_token_ratio=payload.output_token_ratio,
    )
    diagnostics = [
        ProviderDiagnostic(
            provider=row.provider_id,
            status="included" if row.compatible else "excluded",
            reason=row.reason,
        )
        for row in compat
    ]
    plan_rows = [
        RankedPlan(
            rank=row.rank,
            provider_id=row.provider_id,
            provider_name=row.provider_name,
            offering_id=row.offering_id,
            billing_mode=row.billing_mode,
            confidence=row.confidence,
            monthly_cost_usd=row.monthly_cost_usd,
            score=row.score,
            utilization_at_peak=row.utilization_at_peak,
            risk=RiskBreakdown(
                risk_overload=row.risk.risk_overload,
                risk_complexity=row.risk.risk_complexity,
                total_risk=row.risk.total_risk,
            ),
            assumptions=dict(row.assumptions),
            why=row.why,
        )
        for row in plans
    ]
    return LLMPlanningResponse(
        plans=plan_rows,
        provider_diagnostics=diagnostics,
        excluded_count=0,
        warnings=[],
    )


def run_rank_catalog(payload: CatalogRankingRequest) -> CatalogRankingResponse:
    rows = get_catalog_v2_rows(payload.workload_type)
    if not rows:
        return CatalogRankingResponse(
            offers=[],
            provider_diagnostics=[],
            excluded_count=0,
            warnings=[f"No catalog rows available for workload '{payload.workload_type}'."],
        )

    allowed = set(payload.allowed_providers) if payload.allowed_providers else {r.provider for r in rows}
    run = run_catalog_ranking_with_relaxation(
        rows=rows,
        allowed_providers=allowed,
        unit_name=payload.unit_name,
        top_k=payload.top_k,
        monthly_budget_max_usd=payload.monthly_budget_max_usd,
        comparator_mode=payload.comparator_mode,
        confidence_weighted=payload.confidence_weighted,
        workload_type=payload.workload_type,
        monthly_usage=payload.monthly_usage,
        throughput_aware=payload.throughput_aware,
        peak_to_avg=payload.peak_to_avg,
        util_target=payload.util_target,
        strict_capacity_check=payload.strict_capacity_check,
        alpha=1.0,
    )
    workload_provider_ids = sorted({r.provider for r in rows})
    selected_global_providers = sorted(allowed)
    diagnostics = build_provider_diagnostics(
        workload_provider_ids=workload_provider_ids,
        selected_global_providers=selected_global_providers,
        provider_reasons=run.provider_reasons,
    )
    offers = [
        RankedCatalogOffer(
            rank=idx,
            provider=row.provider,
            sku_name=row.offering,
            billing_mode=row.billing,
            unit_price_usd=row.listed_unit_price,
            normalized_price=(row.comparator_price if payload.comparator_mode == "normalized" else None),
            unit_name=row.unit_name,
            confidence=row.confidence,
            monthly_estimate_usd=row.monthly_estimate_usd,
            required_replicas=row.required_replicas,
            capacity_check=(
                "ok"
                if row.capacity_check and row.capacity_check.startswith("pass")
                else "unknown"
            ),
            previous_unit_price_usd=row.previous_unit_price_usd,
            price_change_abs_usd=row.price_change_abs_usd,
            price_change_pct=row.price_change_pct,
        )
        for idx, row in enumerate(run.ranked, start=1)
    ]
    warnings: list[str] = []
    if not offers:
        warnings.append("No offers matched the selected providers/unit/budget filter.")
    if run.selected_step != "strict" and offers:
        warnings.append(
            f"Applied fallback step '{run.selected_step}' to return best available matches."
        )
    warnings.extend(_build_exclusion_summary_warnings(run.exclusion_breakdown))
    return CatalogRankingResponse(
        offers=offers,
        provider_diagnostics=diagnostics,
        excluded_count=run.excluded_offer_count,
        warnings=warnings,
        relaxation_applied=run.selected_step != "strict",
        relaxation_steps=run.relaxation_trace,
        exclusion_breakdown=run.exclusion_breakdown,
    )


def run_browse_catalog(
    workload_type: str | None = None,
    provider: str | None = None,
    unit_name: str | None = None,
) -> CatalogBrowseResponse:
    rows = get_catalog_v2_rows(workload_type)
    filtered = [
        row
        for row in rows
        if (provider is None or row.provider == provider)
        and (unit_name is None or row.unit_name == unit_name)
    ]
    payload_rows: list[dict[str, Any]] = []
    for row in filtered:
        payload_rows.append(
            {
                "provider": row.provider,
                "workload_type": row.workload_type,
                "sku_name": row.sku_name,
                "unit_name": row.unit_name,
                "unit_price_usd": row.unit_price_usd,
                "billing_mode": row.billing_mode,
                "confidence": row.confidence,
                "region": row.region,
                "model_name": row.model_key,
                "throughput_value": row.throughput_value,
                "throughput_unit": row.throughput_unit,
                "previous_unit_price_usd": row.previous_unit_price_usd,
                "price_change_abs_usd": row.price_change_abs_usd,
                "price_change_pct": row.price_change_pct,
            }
        )
    return CatalogBrowseResponse(rows=payload_rows, total=len(payload_rows))


def run_invoice_analyze(file_bytes: bytes) -> InvoiceAnalysisResponse:
    rows = get_catalog_v2_rows()
    suggestions, _summary = analyze_invoice_csv(file_bytes, rows)
    parsed_rows, parse_warnings = parse_invoice_upload(file_bytes)
    line_items: list[InvoiceLineItem] = []
    totals_by_provider: dict[str, float] = defaultdict(float)
    workloads: set[str] = set()
    warnings: list[str] = list(parse_warnings)

    for row in parsed_rows:
        provider = str(row.get("provider", "unknown")).strip() or "unknown"
        workload = str(row.get("workload_type", "unknown")).strip() or "unknown"
        try:
            quantity = float(str(row.get("usage_qty", "0")).strip() or "0")
            total = float(str(row.get("amount_usd", "0")).strip() or "0")
        except ValueError:
            continue
        if quantity <= 0 or total < 0:
            continue
        unit = str(row.get("usage_unit", "")).strip()
        unit_price = (total / quantity) if quantity > 0 else 0.0
        line_items.append(
            InvoiceLineItem(
                provider=provider,
                workload_type=workload,
                line_item=str(row.get("line_item", "")).strip() or "Invoice line",
                quantity=quantity,
                unit=unit,
                unit_price=unit_price,
                total=total,
            )
        )
        totals_by_provider[provider] += total
        workloads.add(workload)

    if suggestions:
        warnings.append(
            f"Found {len(suggestions)} potential savings opportunities versus current catalog."
        )

    if not line_items:
        warnings.append("No valid invoice rows were parsed from the uploaded CSV.")
    grand_total = float(sum(totals_by_provider.values()))
    return InvoiceAnalysisResponse(
        line_items=line_items,
        totals_by_provider=dict(totals_by_provider),
        grand_total=grand_total,
        detected_workloads=sorted(workloads),
        warnings=warnings,
    )


def _build_assist_reply(message: str, workload: str | None, providers: list[str]) -> str:
    rows = get_catalog_v2_rows()
    scoped_workload, scoped_providers = resolve_ai_scope(
        ai_text=message,
        selected_workload=workload or "llm",
        selected_providers=providers,
        rows=rows,
    )
    context = build_catalog_context(
        selected_workload=scoped_workload,
        selected_providers=scoped_providers,
        rows=rows,
        max_rows=25,
    )
    if context.startswith("No matching catalog rows"):
        return "I don't have matching catalog rows for that request yet. Try another workload or provider set."

    filtered = [
        row
        for row in rows
        if (scoped_workload is None or row.workload_type == scoped_workload)
        and row.provider in set(scoped_providers)
    ]
    if not filtered:
        return "No matching rows found in current catalog scope."

    filtered.sort(key=lambda row: row.unit_price_usd)
    top = filtered[:3]
    lines = [f"For workload `{scoped_workload or 'all'}`, lowest current unit prices are:"]
    for idx, row in enumerate(top, start=1):
        lines.append(
            f"{idx}. {row.provider}: {row.sku_name} at {row.unit_price_usd:.6g} USD/{row.unit_name}"
        )
    lines.append("These are list-price comparisons from the current catalog snapshot.")
    return "\n".join(lines)


def run_ai_assist(payload: AIAssistRequest) -> AIAssistResponse:
    text = payload.message.strip()
    workload = payload.context.workload_type
    providers = payload.context.providers
    if not text:
        return AIAssistResponse(reply="Ask me about provider tradeoffs, cheapest options, or risk posture.")
    reply = _build_assist_reply(text, workload, providers)
    suggested_action = "run_optimize" if "lowest current unit prices" in reply else None
    return AIAssistResponse(reply=reply, suggested_action=suggested_action)


def _format_money(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"${value:,.2f}"


def _llm_report_sections(response: LLMPlanningResponse) -> list[ReportSection]:
    if not response.plans:
        return [
            ReportSection(
                title="Executive Summary",
                bullets=["No feasible LLM plans were returned for the current constraints."],
            )
        ]
    winner = response.plans[0]
    return [
        ReportSection(
            title="Executive Summary",
            bullets=[
                f"Top plan: {winner.provider_name} ({winner.offering_id}).",
                f"Estimated monthly cost: {_format_money(winner.monthly_cost_usd)}.",
                f"Total risk score: {winner.risk.total_risk:.3f}.",
            ],
        ),
        ReportSection(
            title="Top Recommendations",
            bullets=[
                f"#{plan.rank} {plan.provider_name} · {_format_money(plan.monthly_cost_usd)} · {plan.confidence}"
                for plan in response.plans[:5]
            ],
        ),
        ReportSection(
            title="Diagnostics",
            bullets=[
                f"Included/compatible providers: {sum(1 for d in response.provider_diagnostics if d.status == 'included')}.",
                f"Excluded providers: {sum(1 for d in response.provider_diagnostics if d.status != 'included')}.",
                *response.warnings[:3],
            ],
        ),
    ]


def _catalog_report_sections(response: CatalogRankingResponse) -> list[ReportSection]:
    if not response.offers:
        return [
            ReportSection(
                title="Executive Summary",
                bullets=["No matching catalog offers were returned for the selected constraints."],
            )
        ]
    winner = response.offers[0]
    return [
        ReportSection(
            title="Executive Summary",
            bullets=[
                f"Top offer: {winner.provider} ({winner.sku_name}).",
                f"Estimated monthly cost: {_format_money(winner.monthly_estimate_usd)}.",
                f"Unit price: {_format_money(winner.unit_price_usd)} per {winner.unit_name}.",
            ],
        ),
        ReportSection(
            title="Top Recommendations",
            bullets=[
                f"#{offer.rank} {offer.provider} · {_format_money(offer.monthly_estimate_usd)} · {offer.confidence}"
                for offer in response.offers[:10]
            ],
        ),
        ReportSection(
            title="Filter Diagnostics",
            bullets=[
                f"Excluded offers count: {response.excluded_count}.",
                *response.warnings[:4],
            ],
        ),
    ]


def _sections_to_markdown(title: str, mode: str, generated_at_utc: str, sections: list[ReportSection]) -> str:
    lines = [f"# {title}", "", f"- Mode: `{mode}`", f"- Generated at (UTC): `{generated_at_utc}`", ""]
    for section in sections:
        lines.append(f"## {section.title}")
        if section.bullets:
            lines.extend([f"- {item}" for item in section.bullets])
        else:
            lines.append("- n/a")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _sections_to_html(title: str, mode: str, generated_at_utc: str, sections: list[ReportSection]) -> str:
    escaped_title = html.escape(title)
    escaped_mode = html.escape(mode)
    escaped_generated = html.escape(generated_at_utc)
    parts = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8" />'
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />"
        f"<title>{escaped_title}</title>",
        "<style>"
        "body{font-family:Inter,Arial,sans-serif;background:#0b0b0f;color:#e5e7eb;padding:24px;line-height:1.5}"
        "h1{margin:0 0 12px 0;font-size:28px} h2{margin:22px 0 8px 0;font-size:20px}"
        "ul{margin:0 0 0 18px;padding:0} li{margin:4px 0}"
        ".meta{color:#9ca3af;margin-bottom:18px}"
        ".card{background:#111218;border:1px solid #2b2f3a;border-radius:10px;padding:14px;margin:12px 0}"
        "</style></head><body>",
        f"<h1>{escaped_title}</h1>",
        f"<div class=\"meta\">Mode: <code>{escaped_mode}</code><br/>Generated at (UTC): <code>{escaped_generated}</code></div>",
    ]
    for section in sections:
        parts.append("<section class=\"card\">")
        parts.append(f"<h2>{html.escape(section.title)}</h2>")
        if section.bullets:
            parts.append("<ul>")
            parts.extend(f"<li>{html.escape(item)}</li>" for item in section.bullets)
            parts.append("</ul>")
        else:
            parts.append("<p>n/a</p>")
        parts.append("</section>")
    parts.append("</body></html>")
    return "".join(parts)


def _text_to_minimal_pdf_bytes(text: str) -> bytes:
    # Minimal deterministic PDF encoder for report export without external dependencies.
    safe_text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = safe_text.split("\n")
    max_lines_per_page = 42
    pages: list[list[str]] = [
        lines[i : i + max_lines_per_page] for i in range(0, len(lines), max_lines_per_page)
    ] or [[]]

    def _escape_pdf_text(value: str) -> str:
        return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    contents: list[str] = []
    for page_lines in pages:
        stream_lines = ["BT", "/F1 10 Tf", "50 780 Td", "12 TL"]
        if page_lines:
            stream_lines.append(f"({_escape_pdf_text(page_lines[0])}) Tj")
            for line in page_lines[1:]:
                stream_lines.append("T*")
                stream_lines.append(f"({_escape_pdf_text(line)}) Tj")
        stream_lines.append("ET")
        contents.append("\n".join(stream_lines))

    objects: list[bytes] = []
    # 1: Catalog, 2: Pages, 3: Font
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{4 + idx * 2} 0 R" for idx in range(len(contents)))
    objects.append(f"<< /Type /Pages /Count {len(contents)} /Kids [{kids}] >>".encode("utf-8"))
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for idx, content in enumerate(contents):
        page_obj = f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents {5 + idx * 2} 0 R >>"
        content_bytes = content.encode("utf-8")
        content_obj = (
            b"<< /Length "
            + str(len(content_bytes)).encode("utf-8")
            + b" >>\nstream\n"
            + content_bytes
            + b"\nendstream"
        )
        objects.append(page_obj.encode("utf-8"))
        objects.append(content_obj)

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{idx} 0 obj\n".encode("utf-8"))
        out.extend(obj)
        out.extend(b"\nendobj\n")
    xref_start = len(out)
    out.extend(f"xref\n0 {len(objects)+1}\n".encode("utf-8"))
    out.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.extend(f"{off:010d} 00000 n \n".encode("utf-8"))
    out.extend(
        (
            f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\n"
            f"startxref\n{xref_start}\n%%EOF\n"
        ).encode("utf-8")
    )
    return bytes(out)


def _build_report_narrative(
    mode: str,
    sections: list[ReportSection],
    chart_data: dict[str, Any],
) -> str:
    summary = next((section for section in sections if section.title == "Executive Summary"), None)
    summary_bullet = summary.bullets[0] if summary and summary.bullets else "No summary available."
    if mode == "llm":
        top_cost = chart_data.get("cost_by_rank", [])
        risk = chart_data.get("risk_breakdown", [])
        risk_lead = risk[0]["total_risk"] if risk else None
        return (
            f"Primary recommendation: {summary_bullet} "
            f"This result is grounded to the current run payload and catalog snapshot. "
            f"Returned plans: {len(top_cost)}; top-plan risk={risk_lead if risk_lead is not None else 'n/a'}."
        )
    top_cost = chart_data.get("cost_by_rank", [])
    trace = chart_data.get("relaxation_trace", [])
    return (
        f"Primary recommendation: {summary_bullet} "
        f"This ranking uses current catalog rows only; fallback steps attempted={len(trace)}; "
        f"offers returned={len(top_cost)}."
    )


def _risk_band_from_total_risk(total_risk: float | None) -> str:
    if total_risk is None:
        return "unknown"
    if total_risk < 0.3:
        return "low"
    if total_risk < 0.6:
        return "medium"
    return "high"


def _build_scaling_summary_for_llm(response: LLMPlanningResponse) -> ScalingPlanResponse:
    if not response.plans:
        return ScalingPlanResponse(
            mode="llm",
            deployment_mode="unknown",
            estimated_gpu_count=0,
            projected_utilization=None,
            utilization_target=None,
            risk_band="unknown",
            capacity_check="unknown",
            rationale="No ranked LLM plans available to estimate scaling.",
            assumptions=["No feasible LLM plans were returned for current constraints."],
        )

    top = response.plans[0]
    plan_assumptions = dict(top.assumptions or {})
    util_target = (
        float(plan_assumptions["util_target"])
        if "util_target" in plan_assumptions
        else None
    )

    deployment_mode = (
        "serverless"
        if top.billing_mode == "per_token"
        else "autoscale"
        if top.billing_mode == "autoscale_hourly"
        else "dedicated"
        if top.billing_mode == "dedicated_hourly"
        else "unknown"
    )
    capacity_check = (
        "ok"
        if top.utilization_at_peak is not None
        else "unknown"
    )
    projected_util = top.utilization_at_peak
    assumptions = [
        f"Top ranked plan #{top.rank} ({top.provider_name}, {top.offering_id}) was used for scaling guidance.",
        "Guidance is risk-aware and based on current assumptions in the planning payload.",
    ]
    if util_target is not None:
        assumptions.append(f"Utilization target used in planning: {util_target:.2f}.")
    if projected_util is None:
        assumptions.append("Projected utilization unavailable for this billing mode.")

    return ScalingPlanResponse(
        mode="llm",
        deployment_mode=deployment_mode,
        estimated_gpu_count=max(int(getattr(top, "gpu_count", 0) or 0), 0),
        suggested_gpu_type=getattr(top, "gpu_type", None),
        projected_utilization=projected_util,
        utilization_target=util_target,
        risk_band=_risk_band_from_total_risk(top.risk.total_risk),
        capacity_check=capacity_check,
        rationale=(
            f"{top.provider_name} rank #{top.rank} is the current best-value plan with "
            f"risk={top.risk.total_risk:.2f} and billing mode '{top.billing_mode}'."
        ),
        assumptions=assumptions,
    )


def _build_scaling_summary_for_catalog(response: CatalogRankingResponse) -> ScalingPlanResponse:
    if not response.offers:
        return ScalingPlanResponse(
            mode="catalog",
            deployment_mode="unknown",
            estimated_gpu_count=0,
            projected_utilization=None,
            utilization_target=None,
            risk_band="unknown",
            capacity_check="unknown",
            rationale="No ranked catalog offers available to estimate scaling.",
            assumptions=["No matching offers were returned after current filters and fallback steps."],
        )

    top = response.offers[0]
    deployment_mode = (
        "autoscale"
        if top.billing_mode == "autoscale_hourly"
        else "dedicated"
        if top.billing_mode == "dedicated_hourly"
        else "serverless"
        if top.billing_mode == "per_token"
        else "unknown"
    )
    capacity_check = top.capacity_check
    if capacity_check == "insufficient":
        risk_band = "high"
    elif capacity_check == "ok":
        risk_band = "low"
    else:
        risk_band = "unknown"
    assumptions = [
        f"Top ranked offer #{top.rank} ({top.provider}, {top.sku_name}) was used for scaling guidance.",
        "Catalog scaling guidance uses required_replicas and capacity_check when present.",
    ]
    if top.required_replicas is None:
        assumptions.append("Throughput metadata was missing, so GPU replica estimate may be unavailable.")

    return ScalingPlanResponse(
        mode="catalog",
        deployment_mode=deployment_mode,
        estimated_gpu_count=max(int(top.required_replicas or 0), 0),
        suggested_gpu_type=None,
        projected_utilization=None,
        utilization_target=None,
        risk_band=risk_band,
        capacity_check=capacity_check,
        rationale=(
            f"{top.provider} rank #{top.rank} is the current best-value offer under active "
            f"filters with billing mode '{top.billing_mode}'."
        ),
        assumptions=assumptions,
    )


def run_plan_scaling(payload: ScalingPlanRequest) -> ScalingPlanResponse:
    if payload.mode == "llm":
        assert payload.llm_planning is not None
        return _build_scaling_summary_for_llm(payload.llm_planning)
    assert payload.catalog_ranking is not None
    return _build_scaling_summary_for_catalog(payload.catalog_ranking)


def _rows_to_csv_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    ordered_keys = list(rows[0].keys())
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=ordered_keys)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key) for key in ordered_keys})
    return output.getvalue()


def _build_report_csv_exports(payload: ReportGenerateRequest) -> dict[str, str]:
    if payload.mode == "llm":
        assert payload.llm_planning is not None
        plans_rows = [
            {
                "rank": plan.rank,
                "provider_id": plan.provider_id,
                "provider_name": plan.provider_name,
                "offering_id": plan.offering_id,
                "billing_mode": plan.billing_mode,
                "confidence": plan.confidence,
                "monthly_cost_usd": plan.monthly_cost_usd,
                "score": plan.score,
                "utilization_at_peak": plan.utilization_at_peak,
                "risk_overload": plan.risk.risk_overload,
                "risk_complexity": plan.risk.risk_complexity,
                "total_risk": plan.risk.total_risk,
                "why": plan.why,
            }
            for plan in payload.llm_planning.plans
        ]
        diagnostics_rows = [
            {"provider": d.provider, "status": d.status, "reason": d.reason}
            for d in payload.llm_planning.provider_diagnostics
        ]
        return {
            "ranked_results.csv": _rows_to_csv_text(plans_rows),
            "provider_diagnostics.csv": _rows_to_csv_text(diagnostics_rows),
        }

    assert payload.catalog_ranking is not None
    offer_rows = [
        {
            "rank": offer.rank,
            "provider": offer.provider,
            "sku_name": offer.sku_name,
            "billing_mode": offer.billing_mode,
            "unit_name": offer.unit_name,
            "unit_price_usd": offer.unit_price_usd,
            "normalized_price": offer.normalized_price,
            "monthly_estimate_usd": offer.monthly_estimate_usd,
            "confidence": offer.confidence,
            "required_replicas": offer.required_replicas,
            "capacity_check": offer.capacity_check,
            "previous_unit_price_usd": offer.previous_unit_price_usd,
            "price_change_abs_usd": offer.price_change_abs_usd,
            "price_change_pct": offer.price_change_pct,
        }
        for offer in payload.catalog_ranking.offers
    ]
    diagnostics_rows = [
        {"provider": d.provider, "status": d.status, "reason": d.reason}
        for d in payload.catalog_ranking.provider_diagnostics
    ]
    return {
        "ranked_results.csv": _rows_to_csv_text(offer_rows),
        "provider_diagnostics.csv": _rows_to_csv_text(diagnostics_rows),
    }


def _build_llm_report_chart_data(response: LLMPlanningResponse) -> dict[str, Any]:
    confidence_counts = Counter(plan.confidence for plan in response.plans)
    return {
        "cost_by_rank": [
            {
                "rank": plan.rank,
                "provider_id": plan.provider_id,
                "provider_name": plan.provider_name,
                "monthly_cost_usd": plan.monthly_cost_usd,
                "score": plan.score,
                "total_risk": plan.risk.total_risk,
            }
            for plan in response.plans
        ],
        "risk_breakdown": [
            {
                "rank": plan.rank,
                "provider_id": plan.provider_id,
                "risk_overload": plan.risk.risk_overload,
                "risk_complexity": plan.risk.risk_complexity,
                "total_risk": plan.risk.total_risk,
            }
            for plan in response.plans
        ],
        "confidence_distribution": dict(sorted(confidence_counts.items())),
    }


def _build_catalog_report_chart_data(response: CatalogRankingResponse) -> dict[str, Any]:
    confidence_counts = Counter(offer.confidence for offer in response.offers)
    return {
        "cost_by_rank": [
            {
                "rank": offer.rank,
                "provider": offer.provider,
                "sku_name": offer.sku_name,
                "monthly_estimate_usd": offer.monthly_estimate_usd,
                "unit_price_usd": offer.unit_price_usd,
                "unit_name": offer.unit_name,
            }
            for offer in response.offers
        ],
        "exclusion_breakdown": dict(response.exclusion_breakdown),
        "relaxation_trace": list(response.relaxation_steps),
        "confidence_distribution": dict(sorted(confidence_counts.items())),
    }


def _sorted_confidence_distribution(confidence_counts: Counter[str]) -> list[dict[str, Any]]:
    ordered_keys = [key for key in CONFIDENCE_ORDER if key in confidence_counts]
    unordered_keys = sorted(key for key in confidence_counts.keys() if key not in set(CONFIDENCE_ORDER))
    keys = ordered_keys + unordered_keys
    return [{"confidence": key, "count": int(confidence_counts[key])} for key in keys]


def _build_llm_report_charts(response: LLMPlanningResponse) -> list[ReportChart]:
    plans = sorted(response.plans, key=lambda plan: plan.rank)
    confidence_counts = Counter(plan.confidence for plan in plans)
    confidence_points = _sorted_confidence_distribution(confidence_counts)
    return [
        ReportChart(
            id="cost_comparison",
            title="Cost Comparison",
            type="bar",
            x_label="Rank",
            y_label="Monthly Cost (USD)",
            series=[
                ReportChartSeries(
                    id="monthly_cost_usd",
                    label="Monthly Cost",
                    unit="usd",
                    points=[
                        {
                            "rank": plan.rank,
                            "provider_id": plan.provider_id,
                            "provider_name": plan.provider_name,
                            "value": plan.monthly_cost_usd,
                        }
                        for plan in plans
                    ],
                )
            ],
            legend=["Monthly Cost"],
            meta={"sort": "rank_asc"},
        ),
        ReportChart(
            id="risk_comparison",
            title="Risk Comparison",
            type="stacked_bar",
            x_label="Rank",
            y_label="Risk Score",
            series=[
                ReportChartSeries(
                    id="risk_overload",
                    label="Overload Risk",
                    unit="risk_score",
                    points=[
                        {
                            "rank": plan.rank,
                            "provider_id": plan.provider_id,
                            "provider_name": plan.provider_name,
                            "value": plan.risk.risk_overload,
                        }
                        for plan in plans
                    ],
                ),
                ReportChartSeries(
                    id="risk_complexity",
                    label="Complexity Risk",
                    unit="risk_score",
                    points=[
                        {
                            "rank": plan.rank,
                            "provider_id": plan.provider_id,
                            "provider_name": plan.provider_name,
                            "value": plan.risk.risk_complexity,
                        }
                        for plan in plans
                    ],
                ),
            ],
            legend=["Overload Risk", "Complexity Risk"],
            meta={"sort": "rank_asc", "total_risk_available": True},
        ),
        ReportChart(
            id="confidence_distribution",
            title="Confidence Distribution",
            type="bar",
            x_label="Confidence Tier",
            y_label="Count",
            series=[
                ReportChartSeries(
                    id="confidence_count",
                    label="Count",
                    unit="count",
                    points=[{"confidence": row["confidence"], "value": row["count"]} for row in confidence_points],
                )
            ],
            legend=["Count"],
            meta={"confidence_order": CONFIDENCE_ORDER},
        ),
        ReportChart(
            id="fallback_trace",
            title="Fallback Trace",
            type="step_line",
            x_label="Step",
            y_label="Offers Returned",
            series=[
                ReportChartSeries(
                    id="offers_returned",
                    label="Offers Returned",
                    unit="count",
                    points=[
                        {
                            "step_index": 0,
                            "step": "strict",
                            "value": len(plans),
                            "selected": True,
                        }
                    ],
                )
            ],
            legend=["Offers Returned"],
            meta={"relaxation_applied": False},
        ),
    ]


def _build_catalog_report_charts(response: CatalogRankingResponse) -> list[ReportChart]:
    offers = sorted(response.offers, key=lambda offer: offer.rank)
    confidence_counts = Counter(offer.confidence for offer in offers)
    confidence_points = _sorted_confidence_distribution(confidence_counts)
    relaxation_trace = list(response.relaxation_steps)
    fallback_points: list[dict[str, Any]] = []
    for idx, step in enumerate(relaxation_trace):
        fallback_points.append(
            {
                "step_index": idx,
                "step": step.get("step", f"step_{idx}"),
                "attempted": bool(step.get("attempted", False)),
                "selected": bool(step.get("selected", False)),
                "value": int(step.get("offers_returned", 0) or 0),
            }
        )
    return [
        ReportChart(
            id="cost_comparison",
            title="Cost Comparison",
            type="bar",
            x_label="Rank",
            y_label="Monthly Estimate (USD)",
            series=[
                ReportChartSeries(
                    id="monthly_estimate_usd",
                    label="Monthly Estimate",
                    unit="usd",
                    points=[
                        {
                            "rank": offer.rank,
                            "provider": offer.provider,
                            "sku_name": offer.sku_name,
                            "value": offer.monthly_estimate_usd,
                        }
                        for offer in offers
                    ],
                )
            ],
            legend=["Monthly Estimate"],
            meta={"sort": "rank_asc"},
        ),
        ReportChart(
            id="risk_comparison",
            title="Risk Comparison",
            type="bar",
            x_label="Provider",
            y_label="Confidence Risk Proxy",
            series=[
                ReportChartSeries(
                    id="confidence_risk_proxy",
                    label="Risk Proxy",
                    unit="risk_score",
                    points=[
                        {
                            "rank": offer.rank,
                            "provider": offer.provider,
                            "confidence": offer.confidence,
                            "value": 0.0
                            if offer.confidence in {"official", "high"}
                            else 0.2
                            if offer.confidence in {"medium", "estimated"}
                            else 0.4,
                        }
                        for offer in offers
                    ],
                )
            ],
            legend=["Risk Proxy"],
            meta={"note": "Proxy from confidence tiers for non-LLM catalog offers."},
        ),
        ReportChart(
            id="confidence_distribution",
            title="Confidence Distribution",
            type="bar",
            x_label="Confidence Tier",
            y_label="Count",
            series=[
                ReportChartSeries(
                    id="confidence_count",
                    label="Count",
                    unit="count",
                    points=[{"confidence": row["confidence"], "value": row["count"]} for row in confidence_points],
                )
            ],
            legend=["Count"],
            meta={"confidence_order": CONFIDENCE_ORDER},
        ),
        ReportChart(
            id="fallback_trace",
            title="Fallback Trace",
            type="step_line",
            x_label="Relaxation Step",
            y_label="Offers Returned",
            series=[
                ReportChartSeries(
                    id="offers_returned",
                    label="Offers Returned",
                    unit="count",
                    points=fallback_points,
                )
            ],
            legend=["Offers Returned"],
            meta={"relaxation_applied": bool(response.relaxation_applied)},
        ),
    ]


def run_generate_report(payload: ReportGenerateRequest) -> ReportGenerateResponse:
    generated_at_utc = datetime.now(timezone.utc).isoformat()
    catalog_meta = get_catalog_v2_metadata()
    scaling_summary: ScalingPlanResponse | None = None
    if payload.mode == "llm":
        assert payload.llm_planning is not None
        sections = _llm_report_sections(payload.llm_planning)
        scaling_summary = run_plan_scaling(
            ScalingPlanRequest(mode="llm", llm_planning=payload.llm_planning)
        )
        chart_data = _build_llm_report_chart_data(payload.llm_planning)
        charts = _build_llm_report_charts(payload.llm_planning)
    else:
        assert payload.catalog_ranking is not None
        sections = _catalog_report_sections(payload.catalog_ranking)
        scaling_summary = run_plan_scaling(
            ScalingPlanRequest(mode="catalog", catalog_ranking=payload.catalog_ranking)
        )
        chart_data = _build_catalog_report_chart_data(payload.catalog_ranking)
        charts = _build_catalog_report_charts(payload.catalog_ranking)
    if scaling_summary is not None:
        gpu_label = (
            str(scaling_summary.estimated_gpu_count)
            if scaling_summary.estimated_gpu_count > 0
            else "n/a"
        )
        util_label = (
            f"{scaling_summary.projected_utilization:.2f}"
            if scaling_summary.projected_utilization is not None
            else "n/a"
        )
        sections.append(
            ReportSection(
                title="Scaling Summary",
                bullets=[
                    f"Recommended deployment mode: {scaling_summary.deployment_mode}.",
                    f"Estimated GPU count: {gpu_label}.",
                    f"Capacity check: {scaling_summary.capacity_check}; risk band: {scaling_summary.risk_band}.",
                    f"Projected utilization: {util_label}.",
                    scaling_summary.rationale,
                ],
            )
        )
    markdown = _sections_to_markdown(
        title=payload.title,
        mode=payload.mode,
        generated_at_utc=generated_at_utc,
        sections=sections,
    )
    html_report = _sections_to_html(
        title=payload.title,
        mode=payload.mode,
        generated_at_utc=generated_at_utc,
        sections=sections,
    )
    pdf_bytes = _text_to_minimal_pdf_bytes(markdown)
    report_hash = hashlib.sha1(markdown.encode("utf-8")).hexdigest()[:12]
    metadata = {
        "chart_schema_version": "v1.2",
        "catalog_generated_at_utc": catalog_meta.get("generated_at_utc"),
        "catalog_schema_version": catalog_meta.get("schema_version"),
        "catalog_row_count": catalog_meta.get("row_count"),
        "catalog_providers_synced_count": len(catalog_meta.get("providers_synced") or []),
    }
    narrative = (
        _build_report_narrative(payload.mode, sections=sections, chart_data=chart_data)
        if payload.include_narrative
        else None
    )
    csv_exports = _build_report_csv_exports(payload) if payload.include_csv_exports else {}
    return ReportGenerateResponse(
        report_id=f"rep_{report_hash}",
        generated_at_utc=generated_at_utc,
        title=payload.title,
        mode=payload.mode,
        sections=sections,
        charts=charts if payload.include_charts else [],
        chart_data=chart_data if payload.include_charts else {},
        metadata=metadata,
        output_format=payload.output_format,
        narrative=narrative,
        csv_exports=csv_exports,
        markdown=markdown,
        html=html_report if payload.output_format in {"html", "pdf"} else None,
        pdf_base64=base64.b64encode(pdf_bytes).decode("ascii") if payload.output_format == "pdf" else None,
    )


def parse_invoice_upload(file_bytes: bytes) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse raw invoice CSV bytes into normalized dictionaries for API fallback use."""
    warnings: list[str] = []
    try:
        decoded = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        decoded = file_bytes.decode("latin-1")
        warnings.append("Decoded invoice as latin-1 due to utf-8 decode failure.")
    reader = csv.DictReader(io.StringIO(decoded))
    rows = [dict(row) for row in reader if row]
    return rows, warnings
