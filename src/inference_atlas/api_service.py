"""Service-layer handlers for API endpoints."""

from __future__ import annotations

import csv
import base64
import hashlib
import html
import io
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from inference_atlas.ai_copilot import next_copilot_turn
from inference_atlas.ai_inference import build_catalog_context, resolve_ai_scope
from inference_atlas.api_models import (
    AIAssistRequest,
    AIAssistResponse,
    CatalogBrowseResponse,
    QualityCatalogResponse,
    QualityCatalogRow,
    QualityInsightPoint,
    QualityInsightsResponse,
    CatalogRankingRequest,
    CatalogRankingResponse,
    CostAuditDataGap,
    CostAuditLegAudit,
    CostAuditHardwareRecommendation,
    CostAuditPricingVerdict,
    CostAuditRecommendation,
    CostAuditRequest,
    CostAuditResponse,
    CostAuditScoreBreakdown,
    CostAuditSavingsEstimate,
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
from inference_atlas.llm.router import LLMRouter
from inference_atlas.llm.schema import WorkloadSpec
from inference_atlas.mvp_planner import get_provider_compatibility, rank_configs
from inference_atlas.quality_metrics import get_quality_scores_for_workload


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


def run_quality_catalog(
    workload_type: str | None = None,
    provider: str | None = None,
    model_key_query: str | None = None,
    mapped_only: bool = False,
) -> QualityCatalogResponse:
    scored = get_quality_scores_for_workload(workload_type=workload_type)
    query = (model_key_query or "").strip().lower()
    payload_rows: list[QualityCatalogRow] = []
    mapped_count = 0

    for row, quality in scored:
        if provider is not None and row.provider != provider:
            continue
        if query and query not in row.model_key.lower() and query not in row.sku_name.lower():
            continue
        if mapped_only and quality is None:
            continue

        quality_mapped = quality is not None
        if quality_mapped:
            mapped_count += 1

        payload_rows.append(
            QualityCatalogRow(
                provider=row.provider,
                workload_type=row.workload_type,
                model_key=row.model_key,
                sku_name=row.sku_name,
                billing_mode=row.billing_mode,
                unit_price_usd=row.unit_price_usd,
                unit_name=row.unit_name,
                quality_mapped=quality_mapped,
                quality_model_id=(quality.model_id if quality else None),
                quality_score_0_100=(quality.normalized_score if quality else None),
                quality_score_adjusted_0_100=(quality.adjusted_score if quality else None),
                quality_confidence=(quality.confidence if quality else None),
                quality_confidence_weight=(quality.confidence_weight if quality else None),
                quality_matched_by=(quality.matched_by if quality else None),
            )
        )

    payload_rows.sort(
        key=lambda r: (
            0 if r.quality_mapped else 1,
            -(r.quality_score_adjusted_0_100 or -1.0),
            r.unit_price_usd,
            r.provider,
            r.model_key,
        )
    )
    total = len(payload_rows)
    return QualityCatalogResponse(
        rows=payload_rows,
        total=total,
        mapped_count=mapped_count,
        unmapped_count=max(0, total - mapped_count),
    )


def _compute_pareto_frontier_flags(points: list[QualityInsightPoint]) -> list[bool]:
    """Pareto frontier for minimizing price and maximizing quality."""
    flags = [True] * len(points)
    for i, p in enumerate(points):
        for j, q in enumerate(points):
            if i == j:
                continue
            price_better_or_equal = q.unit_price_usd <= p.unit_price_usd
            quality_better_or_equal = q.quality_score_adjusted_0_100 >= p.quality_score_adjusted_0_100
            strictly_better = (
                q.unit_price_usd < p.unit_price_usd
                or q.quality_score_adjusted_0_100 > p.quality_score_adjusted_0_100
            )
            if price_better_or_equal and quality_better_or_equal and strictly_better:
                flags[i] = False
                break
    return flags


def run_quality_insights(
    workload_type: str | None = None,
    provider: str | None = None,
    model_key_query: str | None = None,
    mapped_only: bool = True,
) -> QualityInsightsResponse:
    catalog = run_quality_catalog(
        workload_type=workload_type,
        provider=provider,
        model_key_query=model_key_query,
        mapped_only=mapped_only,
    )
    points: list[QualityInsightPoint] = []
    for row in catalog.rows:
        if not row.quality_mapped:
            continue
        adjusted = row.quality_score_adjusted_0_100
        confidence = row.quality_confidence
        if adjusted is None or confidence is None:
            continue
        points.append(
            QualityInsightPoint(
                provider=row.provider,
                workload_type=row.workload_type,
                model_key=row.model_key,
                sku_name=row.sku_name,
                unit_name=row.unit_name,
                unit_price_usd=row.unit_price_usd,
                quality_score_adjusted_0_100=adjusted,
                quality_confidence=confidence,
                is_pareto_frontier=False,
            )
        )

    if points:
        flags = _compute_pareto_frontier_flags(points)
        points = [
            p.model_copy(update={"is_pareto_frontier": flags[idx]})
            for idx, p in enumerate(points)
        ]
        points.sort(
            key=lambda p: (
                0 if p.is_pareto_frontier else 1,
                p.unit_price_usd,
                -p.quality_score_adjusted_0_100,
                p.provider,
                p.model_key,
            )
        )

    frontier_count = sum(1 for p in points if p.is_pareto_frontier)
    return QualityInsightsResponse(
        points=points,
        total_points=len(points),
        frontier_count=frontier_count,
        mapped_count=catalog.mapped_count,
        unmapped_count=catalog.unmapped_count,
    )


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


def _estimate_token_api_monthly_cost(payload: CostAuditRequest) -> float | None:
    if payload.pricing_model != "token_api":
        return None
    in_tok = float(payload.monthly_input_tokens or 0.0)
    out_tok = float(payload.monthly_output_tokens or 0.0)
    total_m = (in_tok + out_tok) / 1_000_000.0
    # Conservative baseline when provider/model specific pricing isn't supplied.
    blended_per_1m = 2.0
    return total_m * blended_per_1m


GPU_HOURLY_USD: dict[str, float] = {
    "A100_80GB": 2.7,
    "H100_80GB": 4.0,
    "H200": 5.5,
    "B200": 6.2,
}

PROCUREMENT_DISCOUNT_FACTOR: dict[str, float] = {
    "on_demand": 1.0,
    "reserved": 0.6,
    "spot": 0.4,
    "mixed": 0.7,
    "unknown": 1.0,
}


_GPU_PRICING_ROWS_CACHE: list[dict[str, str]] | None = None
_GPU_PROVIDER_ALIASES: dict[str, str] = {
    "together": "together_ai",
    "together_ai": "together_ai",
    "aws_rekognition": "aws",
    "aws": "aws",
    "fal_ai": "fal",
    "fal": "fal",
    "voyage_ai": "voyage",
    "voyage": "voyage",
    "google_cloud": "google_cloud",
}


def _modality_to_workload_type(modality: str) -> str:
    mapping = {
        "llm": "llm",
        "asr": "speech_to_text",
        "tts": "text_to_speech",
        "embeddings": "embeddings",
        "image_gen": "image_generation",
        "video_gen": "video_generation",
    }
    return mapping.get(modality, "llm")


def _canonical_gpu_provider(provider: str) -> str:
    token = (provider or "").strip().lower()
    return _GPU_PROVIDER_ALIASES.get(token, token)


def _load_gpu_pricing_rows() -> list[dict[str, str]]:
    global _GPU_PRICING_ROWS_CACHE
    if _GPU_PRICING_ROWS_CACHE is not None:
        return _GPU_PRICING_ROWS_CACHE
    rows: list[dict[str, str]] = []
    gpu_dir = Path("data/providers_csv/gpu")
    if not gpu_dir.exists():
        _GPU_PRICING_ROWS_CACHE = rows
        return rows
    for csv_path in sorted(gpu_dir.glob("*.csv")):
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    provider = (row.get("provider") or "").strip().lower()
                    gpu_type = (row.get("gpu_type") or "").strip().upper()
                    price_raw = (row.get("price_per_gpu_hour_usd") or "").strip()
                    workload_type = (row.get("workload_type") or "").strip().lower()
                    billing_mode = (row.get("billing_mode") or "").strip().lower()
                    if not provider or not gpu_type or not price_raw:
                        continue
                    try:
                        price = float(price_raw)
                    except ValueError:
                        continue
                    if price <= 0:
                        continue
                    rows.append(
                        {
                            "provider": _canonical_gpu_provider(provider),
                            "gpu_type": gpu_type,
                            "workload_type": workload_type,
                            "billing_mode": billing_mode,
                            "price_per_gpu_hour_usd": f"{price:.10f}",
                            "confidence": (row.get("confidence") or "").strip().lower(),
                            "source_url": (row.get("source_url") or "").strip(),
                        }
                    )
        except (OSError, csv.Error):
            continue
    _GPU_PRICING_ROWS_CACHE = rows
    return rows


def _resolve_gpu_hourly_from_csv(
    payload: CostAuditRequest,
    preferred_gpu_type: str | None,
) -> tuple[float | None, str | None, str | None, str | None, str | None]:
    """Return hourly rate + metadata from provider GPU CSV when possible."""
    rows = _load_gpu_pricing_rows()
    if not rows:
        return None, None, None, None, None

    selected_providers = [_canonical_gpu_provider(p) for p in payload.providers if p]
    selected_provider_set = set(selected_providers)
    workload_type = _modality_to_workload_type(payload.modality)
    preferred_billing = "dedicated_hourly" if payload.traffic_pattern == "steady" else "autoscale_hourly"
    preferred_gpu = (preferred_gpu_type or "").strip().upper() if preferred_gpu_type else None

    candidates: list[tuple[int, int, float, dict[str, str]]] = []
    for row in rows:
        provider = row["provider"]
        if selected_provider_set and provider not in selected_provider_set:
            continue
        gpu_type = row["gpu_type"]
        if preferred_gpu and gpu_type != preferred_gpu:
            continue
        try:
            price = float(row["price_per_gpu_hour_usd"])
        except ValueError:
            continue
        if price <= 0:
            continue
        provider_rank = selected_providers.index(provider) if provider in selected_provider_set else 999
        match_score = 0
        if row.get("workload_type") == workload_type:
            match_score += 4
        if row.get("billing_mode") == preferred_billing:
            match_score += 2
        if preferred_gpu and gpu_type == preferred_gpu:
            match_score += 3
        if selected_provider_set:
            match_score += 3
        candidates.append((provider_rank, -match_score, price, row))

    if not candidates:
        return None, None, None, None, None

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    chosen = candidates[0][3]
    try:
        chosen_price = float(chosen["price_per_gpu_hour_usd"])
    except ValueError:
        return None, None, None, None, None
    return (
        chosen_price,
        chosen.get("provider"),
        chosen.get("gpu_type"),
        chosen.get("source_url") or None,
        chosen.get("confidence") or None,
    )


def _estimate_dedicated_gpu_monthly_cost(
    payload: CostAuditRequest,
) -> tuple[float, int, str, float, str, str, str | None, str]:
    total_tokens = float(payload.monthly_input_tokens or 0.0) + float(payload.monthly_output_tokens or 0.0)
    if payload.gpu_count is not None and payload.gpu_count > 0:
        gpu_count = int(payload.gpu_count)
    elif total_tokens >= 5_000_000_000:
        gpu_count = 8
    elif total_tokens >= 2_000_000_000:
        gpu_count = 4
    elif total_tokens >= 500_000_000:
        gpu_count = 2
    else:
        gpu_count = 1

    requested_gpu_type = (payload.gpu_type or "H100_80GB").strip().upper()
    csv_hourly, csv_provider, csv_gpu_type, csv_source_url, csv_confidence = _resolve_gpu_hourly_from_csv(
        payload=payload,
        preferred_gpu_type=requested_gpu_type,
    )
    if csv_hourly is None:
        fallback_hourly = GPU_HOURLY_USD.get(requested_gpu_type, GPU_HOURLY_USD["H100_80GB"])
        hourly = fallback_hourly
        gpu_type = requested_gpu_type
        pricing_basis = f"heuristic_prior:{gpu_type}"
        pricing_source = "heuristic_prior"
        pricing_source_provider = None
    else:
        hourly = csv_hourly
        gpu_type = csv_gpu_type or requested_gpu_type
        provider_label = csv_provider or "catalog"
        pricing_source = "provider_csv"
        pricing_source_provider = provider_label
        if csv_source_url:
            pricing_basis = f"provider_csv:{provider_label}:{csv_source_url}"
        else:
            confidence_label = csv_confidence or "unknown_confidence"
            pricing_basis = f"provider_csv:{provider_label}:{confidence_label}"

    procurement = payload.gpu_procurement_type
    factor = PROCUREMENT_DISCOUNT_FACTOR.get(procurement, 1.0)
    effective_hourly = hourly * factor
    monthly = gpu_count * effective_hourly * 730.0
    return (
        monthly,
        gpu_count,
        gpu_type,
        effective_hourly,
        pricing_basis,
        pricing_source,
        pricing_source_provider,
        gpu_type,
    )


def _gap(field: str, impact: str, why: str) -> CostAuditDataGap:
    return CostAuditDataGap(field=field, impact=impact, why_it_matters=why)


def _priority_rank(priority: str) -> int:
    return 0 if priority == "high" else 1 if priority == "medium" else 2


def _impact_rank(impact: str) -> int:
    return 0 if impact == "high" else 1 if impact == "medium" else 2


def _combine_savings_pct(pcts: list[float]) -> float:
    """Combine overlapping savings recommendations conservatively."""
    remaining = 1.0
    for pct in pcts:
        bounded = max(0.0, min(100.0, float(pct)))
        remaining *= (1.0 - (bounded / 100.0))
    return max(0.0, min(100.0, (1.0 - remaining) * 100.0))


def _infer_modality_for_model(model_name: str) -> str:
    token = model_name.strip().lower()
    if any(k in token for k in ("whisper", "stt", "asr", "transcribe", "nova")):
        return "asr"
    if any(k in token for k in ("tts", "voice", "speech", "eleven")):
        return "tts"
    if "embed" in token:
        return "embeddings"
    if any(k in token for k in ("image", "flux", "dall", "sdxl", "imagen")):
        return "image_gen"
    if any(k in token for k in ("video", "veo", "sora", "wan")):
        return "video_gen"
    return "llm"


def _build_mixed_leg_payloads(payload: CostAuditRequest) -> list[CostAuditRequest]:
    model_names = [m.strip() for m in payload.pipeline_models if m.strip()]
    if not model_names:
        return []
    leg_count = len(model_names)
    spend_total = float(payload.monthly_ai_spend_usd or 0.0)
    spend_per_leg = (spend_total / leg_count) if leg_count > 0 and spend_total > 0 else None
    in_total = float(payload.monthly_input_tokens or 0.0)
    out_total = float(payload.monthly_output_tokens or 0.0)
    in_per_leg = (in_total / leg_count) if leg_count > 0 and in_total > 0 else None
    out_per_leg = (out_total / leg_count) if leg_count > 0 and out_total > 0 else None

    legs: list[CostAuditRequest] = []
    for model in model_names:
        leg = payload.model_copy(
            update={
                "modality": _infer_modality_for_model(model),
                "model_name": model,
                "multi_model_pipeline": False,
                "pipeline_models": [],
                "monthly_ai_spend_usd": spend_per_leg,
                "monthly_input_tokens": in_per_leg,
                "monthly_output_tokens": out_per_leg,
            }
        )
        legs.append(leg)
    return legs


def run_cost_audit(payload: CostAuditRequest, _allow_mixed_split: bool = True) -> CostAuditResponse:
    base_score = 100
    score = base_score
    penalty_points = 0
    bonus_points = 0
    caps_applied: list[str] = []
    recommendations: list[CostAuditRecommendation] = []
    red_flags: list[str] = []
    assumptions: list[str] = []
    data_gaps_detailed: list[CostAuditDataGap] = []
    major_flags = 0
    pricing_source: str = "unknown"
    pricing_source_provider: str | None = None
    pricing_source_gpu: str | None = None

    if payload.quantization_applied == "no" and payload.model_precision in {"fp16", "bf16", "unknown"}:
        penalty_points += 14
        score -= 14
        major_flags += 1
        recommendations.append(
            CostAuditRecommendation(
                recommendation_type="quantization",
                title="Enable lower-precision inference where quality permits",
                rationale="Current precision likely leaves throughput on the table; FP8/INT8 can increase throughput materially.",
                estimated_savings_pct=28.0,
                priority="high",
            )
        )
    if payload.quantization_applied == "yes":
        bonus_points += 6
        score += 6

    if payload.traffic_pattern == "steady" and payload.gpu_procurement_type == "on_demand":
        penalty_points += 16
        score -= 16
        major_flags += 1
        red_flags.append("Steady traffic on on-demand GPUs suggests avoidable reservation waste.")
        if float(payload.monthly_ai_spend_usd or 0.0) <= 0 and payload.pricing_model != "token_api":
            (
                _estimated_dedicated_spend,
                _gpu_count,
                _gpu_type,
                _hourly,
                _pricing_basis,
                _pricing_source,
                _pricing_source_provider,
                _pricing_source_gpu,
            ) = _estimate_dedicated_gpu_monthly_cost(payload)
            assumptions.append("Dedicated spend baseline estimated from GPU type/count and on-demand hourly priors.")
        reserved_factor = PROCUREMENT_DISCOUNT_FACTOR["reserved"] / PROCUREMENT_DISCOUNT_FACTOR["on_demand"]
        procurement_savings_pct = max(0.0, min(100.0, (1.0 - reserved_factor) * 100.0))
        recommendations.append(
            CostAuditRecommendation(
                recommendation_type="procurement",
                title="Switch steady baseline to reserved capacity",
                rationale="Steady demand is typically cheaper on reservations; keep burst on spot/on-demand.",
                estimated_savings_pct=round(procurement_savings_pct, 1),
                priority="high",
            )
        )
    if payload.traffic_pattern == "steady" and payload.gpu_procurement_type == "reserved":
        bonus_points += 8
        score += 8

    if payload.traffic_pattern in {"batch_offline", "business_hours"} and payload.autoscaling == "no":
        penalty_points += 12
        score -= 12
        major_flags += 1
        red_flags.append("No autoscaling for non-steady traffic can create idle spend.")
        recommendations.append(
            CostAuditRecommendation(
                recommendation_type="autoscaling",
                title="Enable autoscaling / scale-to-zero for non-steady workloads",
                rationale="Batch and business-hours workloads benefit from reducing idle GPU hours.",
                estimated_savings_pct=20.0,
                priority="medium",
            )
        )
    if payload.autoscaling == "yes" and payload.traffic_pattern in {"batch_offline", "bursty", "business_hours"}:
        bonus_points += 6
        score += 6

    if payload.caching_enabled == "no" and payload.modality == "llm":
        penalty_points += 9
        score -= 9
        recommendations.append(
            CostAuditRecommendation(
                recommendation_type="caching",
                title="Turn on prompt/prefix caching for repeated context",
                rationale="Repeated prefixes without caching increase avoidable prefill spend.",
                estimated_savings_pct=12.0,
                priority="medium",
            )
        )
    if payload.caching_enabled == "yes" and payload.modality == "llm":
        bonus_points += 4
        score += 4

    token_cost_est = _estimate_token_api_monthly_cost(payload)
    current_spend = float(payload.monthly_ai_spend_usd or token_cost_est or 0.0)
    if payload.pricing_model == "token_api":
        if payload.monthly_input_tokens is None:
            data_gaps_detailed.append(
                _gap(
                    "monthly_input_tokens",
                    "high",
                    "Without token volume the API-to-dedicated break-even estimate is weak.",
                )
            )
        if payload.monthly_output_tokens is None:
            data_gaps_detailed.append(
                _gap(
                    "monthly_output_tokens",
                    "high",
                    "Output volume strongly impacts blended token spend and switch economics.",
                )
            )

        total_tokens = float(payload.monthly_input_tokens or 0.0) + float(payload.monthly_output_tokens or 0.0)
        (
            dedicated_cost,
            est_gpu_count,
            est_gpu_type,
            est_hourly,
            pricing_basis,
            pricing_source,
            pricing_source_provider,
            pricing_source_gpu,
        ) = _estimate_dedicated_gpu_monthly_cost(payload)
        denominator = current_spend if current_spend > 0 else float(token_cost_est or 0.0)
        switch_savings_pct = (
            max(0.0, min(100.0, ((denominator - dedicated_cost) / denominator) * 100.0))
            if denominator > 0
            else 0.0
        )
        if total_tokens >= 1_500_000_000:
            verdict = "consider_switch"
            reason = (
                "Token volume is high enough that dedicated/autoscale GPU may reduce inference-unit cost, "
                "subject to engineering overhead."
            )
            recommendations.append(
                CostAuditRecommendation(
                    recommendation_type="pricing_model_switch",
                    title="Evaluate dedicated/autoscale GPU for high token volume",
                    rationale=(
                        "Estimated dedicated monthly cost "
                        f"(${dedicated_cost:,.0f}) using {est_gpu_count}x {est_gpu_type} "
                        f"@ ${est_hourly:.2f}/hr effective may undercut current API spend "
                        f"(basis: {pricing_basis})."
                    ),
                    estimated_savings_pct=round(switch_savings_pct, 1),
                    priority="high",
                )
            )
            penalty_points += 12
            score -= 12
            major_flags += 1
        else:
            verdict = "appropriate"
            reason = "Token API is typically appropriate for low-to-mid volume and faster time-to-production."
    elif payload.pricing_model == "dedicated_gpu":
        (
            _dedicated_cost,
            _est_gpu_count,
            _est_gpu_type,
            _est_hourly,
            _pricing_basis,
            pricing_source,
            pricing_source_provider,
            pricing_source_gpu,
        ) = _estimate_dedicated_gpu_monthly_cost(payload)
        if payload.gpu_type is None:
            data_gaps_detailed.append(
                _gap(
                    "gpu_type",
                    "high",
                    "GPU family is required for hardware bottleneck fit and procurement comparisons.",
                )
            )
        if payload.gpu_count is None:
            data_gaps_detailed.append(
                _gap(
                    "gpu_count",
                    "high",
                    "Replica count determines dedicated monthly cost and utilization signals.",
                )
            )
        verdict = "appropriate"
        reason = "Dedicated GPU is appropriate when utilization is sustained and operational ownership is acceptable."
    else:
        verdict = "appropriate"
        reason = "Mixed model can be efficient when baseline and burst traffic are split intentionally."

    if payload.modality in {"image_gen", "video_gen"} and payload.gpu_type and "A10" in payload.gpu_type.upper():
        red_flags.append("Image/video workload on lower-tier GPU may be compute-constrained.")
        penalty_points += 8
        score -= 8
        recommendations.append(
            CostAuditRecommendation(
                recommendation_type="hardware_match",
                title="Upgrade compute tier for generation-heavy workloads",
                rationale="Image/video generation is commonly compute-bound and benefits from higher FLOPS tiers.",
                estimated_savings_pct=15.0,
                priority="medium",
            )
        )

    if payload.modality in {"embeddings", "llm"} and payload.workload_execution == "throughput_optimized":
        hw_tier = "multi_gpu" if (payload.gpu_count or 0) > 1 else "single_gpu"
        hw_family = "H200-class (memory bandwidth)"
        hw_reason = "Decode/embedding-heavy throughput workloads often benefit from higher memory bandwidth."
    elif payload.modality in {"image_gen", "video_gen"}:
        hw_tier = "multi_gpu" if (payload.gpu_count or 0) > 1 else "single_gpu"
        hw_family = "H100/B200-class (compute)"
        hw_reason = "Generation-heavy workloads are commonly compute-bound."
    elif payload.pricing_model == "token_api":
        hw_tier = "serverless"
        hw_family = None
        hw_reason = "Current setup is API-serverless; hardware selection is abstracted by provider."
    else:
        hw_tier = "hybrid" if payload.pricing_model == "mixed" else "unknown"
        hw_family = None
        hw_reason = "Insufficient workload-specific hardware signals; provide throughput and latency traces."

    if payload.pricing_model == "token_api":
        assumptions.append("Dedicated GPU break-even is workload-dependent and excludes migration engineering cost.")
    assumptions.append("Savings are deterministic directional estimates based on request inputs and pricing priors.")

    if payload.monthly_ai_spend_usd is None:
        data_gaps_detailed.append(
            _gap(
                "monthly_ai_spend_usd",
                "high",
                "Savings in USD cannot be calibrated without current monthly spend.",
            )
        )
    if payload.avg_input_tokens is None and payload.modality == "llm":
        data_gaps_detailed.append(
            _gap(
                "avg_input_tokens",
                "high",
                "Input length drives prefill cost and can change bottleneck/hardware guidance.",
            )
        )
    if payload.traffic_pattern == "unknown":
        data_gaps_detailed.append(
            _gap(
                "traffic_pattern",
                "medium",
                "Traffic shape affects procurement and autoscaling recommendations.",
            )
        )
    if payload.peak_concurrency is None:
        data_gaps_detailed.append(
            _gap(
                "peak_concurrency",
                "medium",
                "Concurrency helps estimate scaling headroom and autoscaling need.",
            )
        )

    if payload.modality == "mixed":
        data_gaps_detailed.append(
            _gap(
                "per_modality_usage_breakdown",
                "high",
                "Mixed workloads should be audited per modality leg to avoid masking expensive stages.",
            )
        )
        red_flags.append("Mixed modality should be audited per leg; aggregate-only inputs reduce recommendation precision.")

    if major_flags >= 3:
        caps_applied.append("major_flags_cap")
        score = min(score, 45)

    score = max(0, min(100, score))
    pre_cap_score = score

    recommendations_sorted = sorted(
        recommendations,
        key=lambda item: (
            _priority_rank(item.priority),
            -item.estimated_savings_pct,
        ),
    )[:3]

    top_savings_pcts = [item.estimated_savings_pct for item in recommendations_sorted]
    combined_savings_pct = min(85.0, _combine_savings_pct(top_savings_pcts)) if top_savings_pcts else 0.0
    # Align score with strong model-switch economics to avoid conflicting signals.
    if verdict == "consider_switch" and combined_savings_pct > 30.0:
        caps_applied.append("high_switch_savings_cap")
        score = min(score, 45)
    score = max(0, min(100, score))
    post_cap_score = score

    if current_spend > 0:
        high_savings = min(current_spend, current_spend * (combined_savings_pct / 100.0))
        low_savings = min(high_savings, high_savings * 0.5)
    else:
        low_savings = 0.0
        high_savings = 0.0

    if recommendations_sorted:
        top_title = recommendations_sorted[0].title
        assumptions.append(f"Top lever by modeled impact: {top_title}.")

    gap_field_order: dict[str, int] = {
        "monthly_ai_spend_usd": 0,
        "monthly_input_tokens": 1,
        "monthly_output_tokens": 2,
        "avg_input_tokens": 3,
        "gpu_type": 4,
        "gpu_count": 5,
        "per_modality_usage_breakdown": 6,
        "traffic_pattern": 7,
        "peak_concurrency": 8,
    }
    deduped_gaps: dict[str, CostAuditDataGap] = {}
    for gap in data_gaps_detailed:
        existing = deduped_gaps.get(gap.field)
        if existing is None or _impact_rank(gap.impact) < _impact_rank(existing.impact):
            deduped_gaps[gap.field] = gap

    recommendations_sorted = sorted(
        recommendations_sorted,
        key=lambda item: (_priority_rank(item.priority), -item.estimated_savings_pct),
    )
    data_gaps_sorted = sorted(
        deduped_gaps.values(),
        key=lambda g: (_impact_rank(g.impact), gap_field_order.get(g.field, 999), g.field),
    )

    per_modality_audits: list[CostAuditLegAudit] = []
    if payload.modality == "mixed" and _allow_mixed_split:
        leg_payloads = _build_mixed_leg_payloads(payload)
        if leg_payloads:
            leg_responses = [run_cost_audit(leg, _allow_mixed_split=False) for leg in leg_payloads]
            for leg_payload, leg_response in zip(leg_payloads, leg_responses):
                per_modality_audits.append(
                    CostAuditLegAudit(
                        modality=leg_payload.modality,
                        model_name=leg_payload.model_name,
                        estimated_spend_usd=float(leg_payload.monthly_ai_spend_usd or 0.0),
                        efficiency_score=leg_response.efficiency_score,
                        top_recommendation=(
                            leg_response.recommendations[0].title if leg_response.recommendations else None
                        ),
                        estimated_savings_high_usd=leg_response.estimated_monthly_savings.high_usd,
                    )
                )
            total_leg_spend = sum(leg.estimated_spend_usd for leg in per_modality_audits)
            if total_leg_spend > 0:
                weighted_score = sum(
                    leg.efficiency_score * leg.estimated_spend_usd for leg in per_modality_audits
                ) / total_leg_spend
            else:
                weighted_score = sum(leg.efficiency_score for leg in per_modality_audits) / len(per_modality_audits)
            score = max(0, min(100, int(round(weighted_score))))
            pre_cap_score = score
            post_cap_score = score
            summed_high = sum(leg.estimated_savings_high_usd for leg in per_modality_audits)
            if current_spend > 0:
                high_savings = min(current_spend, summed_high)
                low_savings = min(high_savings, high_savings * 0.5)
            else:
                high_savings = summed_high
                low_savings = max(0.0, summed_high * 0.5)
            # This gap is resolved when per-leg mini-audits are available.
            data_gaps_sorted = [g for g in data_gaps_sorted if g.field != "per_modality_usage_breakdown"]
            assumptions.append("Mixed workload score/savings aggregated from per-leg mini-audits.")

    return CostAuditResponse(
        efficiency_score=score,
        recommendations=recommendations_sorted,
        hardware_recommendation=CostAuditHardwareRecommendation(
            tier=hw_tier,
            gpu_family=hw_family,
            deployment_shape=payload.pricing_model,
            reasoning=hw_reason,
        ),
        pricing_model_verdict=CostAuditPricingVerdict(
            current_model=payload.pricing_model,
            verdict=verdict,
            reason=reason,
        ),
        pricing_source=pricing_source,
        pricing_source_provider=pricing_source_provider,
        pricing_source_gpu=pricing_source_gpu,
        red_flags=red_flags,
        estimated_monthly_savings=CostAuditSavingsEstimate(
            low_usd=round(low_savings, 2),
            high_usd=round(high_savings, 2),
            basis=(
                f"current_spend_usd={current_spend:.2f}; "
                f"combined_savings_pct={combined_savings_pct:.2f}; "
                f"top_recommendations={len(recommendations_sorted)}"
            ),
        ),
        score_breakdown=CostAuditScoreBreakdown(
            base_score=base_score,
            penalty_points=penalty_points,
            bonus_points=bonus_points,
            pre_cap_score=pre_cap_score,
            post_cap_score=post_cap_score,
            major_flags=major_flags,
            caps_applied=caps_applied,
            combined_savings_pct=round(combined_savings_pct, 2),
        ),
        per_modality_audits=per_modality_audits,
        assumptions=assumptions,
        data_gaps=[gap.field for gap in data_gaps_sorted],
        data_gaps_detailed=data_gaps_sorted,
    )


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
    llm_narrative = _build_llm_generated_report_narrative(
        mode=mode,
        sections=sections,
        chart_data=chart_data,
    )
    if llm_narrative:
        return llm_narrative

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


def _build_llm_generated_report_narrative(
    mode: str,
    sections: list[ReportSection],
    chart_data: dict[str, Any],
) -> str | None:
    # Keep LLM narrative optional and fail-safe.
    if mode != "llm":
        return None
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        return None
    try:
        router = LLMRouter()
        top_cost_rows = chart_data.get("cost_by_rank", [])
        top_summary = ""
        if isinstance(top_cost_rows, list) and top_cost_rows:
            top = top_cost_rows[0]
            if isinstance(top, dict):
                top_summary = (
                    f"Top plan rank={top.get('rank')}, "
                    f"provider={top.get('provider_name') or top.get('provider_id')}, "
                    f"monthly_cost_usd={top.get('monthly_cost_usd')}, "
                    f"risk={top.get('total_risk')}."
                )
        section_lines: list[str] = []
        for section in sections[:3]:
            section_lines.append(section.title)
            for bullet in section.bullets[:3]:
                section_lines.append(f"- {bullet}")
        recommendation_summary = (
            "Report context:\n"
            + "\n".join(section_lines)
            + ("\n" + top_summary if top_summary else "")
            + "\nProvide a concise, action-oriented recommendation narrative."
        )
        # Report generation does not yet carry full workload spec, so we pass a conservative default.
        workload = WorkloadSpec(
            tokens_per_day=1_000_000.0,
            pattern="steady",
            model_key="report_summary",
            latency_requirement_ms=None,
        )
        generated = router.explain(recommendation_summary=recommendation_summary, workload=workload)
        cleaned = generated.strip()
        return cleaned or None
    except Exception:  # noqa: BLE001 - narrative path must never break report generation
        return None


def _risk_band_from_total_risk(total_risk: float | None) -> str:
    if total_risk is None:
        return "unknown"
    if total_risk < 0.3:
        return "low"
    if total_risk < 0.6:
        return "medium"
    return "high"


def _deployment_mode_from_billing_mode(billing_mode: str | None) -> str:
    token = (billing_mode or "").strip().lower()
    if token == "autoscale_hourly":
        return "autoscale"
    if token == "dedicated_hourly":
        return "dedicated"
    # Serverless/API-style billing families.
    if token in {"per_token", "per_unit", "per_second", "per_minute", "per_request"}:
        return "serverless"
    return "unknown"


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

    deployment_mode = _deployment_mode_from_billing_mode(top.billing_mode)
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
    deployment_mode = _deployment_mode_from_billing_mode(top.billing_mode)
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
