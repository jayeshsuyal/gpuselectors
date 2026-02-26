"""Service-layer handlers for API endpoints."""

from __future__ import annotations

import csv
import hashlib
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
    if not rows:
        return (
            "I don't have pricing rows loaded in the current catalog snapshot yet. "
            "You can still proceed by selecting a workload and broadening provider scope, "
            "then retry optimization."
        )

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
    requested_workload = workload or scoped_workload
    requested_rows = [
        row for row in rows
        if requested_workload is not None and row.workload_type == requested_workload
    ]
    exact_workload_available = bool(requested_rows)

    filtered = [
        row
        for row in rows
        if (scoped_workload is None or row.workload_type == scoped_workload)
        and row.provider in set(scoped_providers)
    ]
    if context.startswith("No matching catalog rows") or not filtered or not exact_workload_available:
        # Fallback policy: never return a dead-end/null response.
        candidate_rows = filtered or requested_rows or rows
        workload_cheapest: dict[str, Any] = {}
        for row in candidate_rows:
            current = workload_cheapest.get(row.workload_type)
            if current is None or row.unit_price_usd < current.unit_price_usd:
                workload_cheapest[row.workload_type] = row

        alternatives = sorted(workload_cheapest.values(), key=lambda row: row.unit_price_usd)[:4]
        lines = []
        if requested_workload and not exact_workload_available:
            lines.append(
                f"I don't have direct rows for `{requested_workload}` in the current catalog scope yet."
            )
        elif not filtered:
            lines.append("I couldn't find direct matches with the current scope/filters.")
        else:
            lines.append("I couldn't find an exact direct match, but here are closest actionable options.")

        if alternatives:
            lines.append("You can consider these alternatives:")
            for idx, row in enumerate(alternatives, start=1):
                lines.append(
                    f"{idx}. {row.workload_type}: {row.provider} · {row.sku_name} "
                    f"at {row.unit_price_usd:.6g} USD/{row.unit_name}"
                )
        else:
            lines.append("No priced alternatives are currently available in the snapshot.")

        lines.append("Next best actions: use all providers, clear unit/budget filters, or pick a nearby workload category.")
        return "\n".join(lines)

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


def run_generate_report(payload: ReportGenerateRequest) -> ReportGenerateResponse:
    generated_at_utc = datetime.now(timezone.utc).isoformat()
    catalog_meta = get_catalog_v2_metadata()
    if payload.mode == "llm":
        assert payload.llm_planning is not None
        sections = _llm_report_sections(payload.llm_planning)
        chart_data = _build_llm_report_chart_data(payload.llm_planning)
    else:
        assert payload.catalog_ranking is not None
        sections = _catalog_report_sections(payload.catalog_ranking)
        chart_data = _build_catalog_report_chart_data(payload.catalog_ranking)
    markdown = _sections_to_markdown(
        title=payload.title,
        mode=payload.mode,
        generated_at_utc=generated_at_utc,
        sections=sections,
    )
    report_hash = hashlib.sha1(markdown.encode("utf-8")).hexdigest()[:12]
    metadata = {
        "catalog_generated_at_utc": catalog_meta.get("generated_at_utc"),
        "catalog_schema_version": catalog_meta.get("schema_version"),
        "catalog_row_count": catalog_meta.get("row_count"),
        "catalog_providers_synced_count": len(catalog_meta.get("providers_synced") or []),
    }
    return ReportGenerateResponse(
        report_id=f"rep_{report_hash}",
        generated_at_utc=generated_at_utc,
        title=payload.title,
        mode=payload.mode,
        sections=sections,
        chart_data=chart_data,
        metadata=metadata,
        markdown=markdown,
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
