"""Catalog ranking helpers for non-LLM workload offer comparisons."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from inference_atlas.contracts import ConfidenceLevel


@dataclass(frozen=True)
class RankedCatalogOffer:
    provider: str
    offering: str
    billing: str
    listed_unit_price: float
    comparator_price: float
    unit_name: str
    confidence: str
    monthly_estimate_usd: float | None
    score: float
    risk_overload: float
    risk_complexity: float
    total_risk: float
    required_replicas: int | None = None
    capacity_check: str | None = None
    exclusion_reason: str | None = None
    previous_unit_price_usd: float | None = None
    price_change_abs_usd: float | None = None
    price_change_pct: float | None = None


@dataclass(frozen=True)
class CatalogRankingRun:
    ranked: list[RankedCatalogOffer]
    provider_reasons: dict[str, str]
    excluded_offer_count: int
    relaxation_trace: list[dict[str, Any]]
    exclusion_breakdown: dict[str, int]
    selected_step: str


CATALOG_TUNING_PRESETS: dict[str, dict[str, dict[str, float]]] = {
    "default": {
        "conservative": {"peak_to_avg": 3.0, "util_target": 0.65, "alpha": 1.4},
        "balanced": {"peak_to_avg": 2.5, "util_target": 0.75, "alpha": 1.0},
        "aggressive": {"peak_to_avg": 2.0, "util_target": 0.85, "alpha": 0.7},
    },
    "speech_to_text": {
        "conservative": {"peak_to_avg": 3.5, "util_target": 0.65, "alpha": 1.5},
        "balanced": {"peak_to_avg": 2.8, "util_target": 0.75, "alpha": 1.0},
        "aggressive": {"peak_to_avg": 2.0, "util_target": 0.85, "alpha": 0.7},
    },
    "text_to_speech": {
        "conservative": {"peak_to_avg": 3.0, "util_target": 0.65, "alpha": 1.4},
        "balanced": {"peak_to_avg": 2.5, "util_target": 0.75, "alpha": 1.0},
        "aggressive": {"peak_to_avg": 2.0, "util_target": 0.85, "alpha": 0.7},
    },
    "video_generation": {
        "conservative": {"peak_to_avg": 2.5, "util_target": 0.60, "alpha": 1.6},
        "balanced": {"peak_to_avg": 2.0, "util_target": 0.70, "alpha": 1.2},
        "aggressive": {"peak_to_avg": 1.6, "util_target": 0.80, "alpha": 0.8},
    },
}


def get_catalog_tuning_preset(workload_type: str, name: str = "balanced") -> dict[str, float]:
    """Return workload-aware preset for non-LLM ranking controls."""
    workload_token = workload_type.strip().lower()
    preset_name = name.strip().lower()
    preset_group = CATALOG_TUNING_PRESETS.get(workload_token, CATALOG_TUNING_PRESETS["default"])
    if preset_name not in preset_group:
        valid = ", ".join(sorted(preset_group))
        raise ValueError(f"Unknown preset '{name}' for workload '{workload_type}'. Valid presets: {valid}")
    return dict(preset_group[preset_name])


def normalize_unit_price_for_workload(
    unit_price_usd: float,
    unit_name: str,
    workload_type: str,
) -> float | None:
    unit = unit_name.strip().lower()
    workload = workload_type.strip().lower()

    if workload in {"llm", "embedding", "embeddings", "moderation"}:
        return unit_price_usd if unit == "1m_tokens" else None

    if workload == "rerank":
        if unit == "per_1k_searches":
            return unit_price_usd * 1000.0
        if unit == "1m_tokens":
            return unit_price_usd
        return None

    if workload in {"speech_to_text", "transcription"}:
        if unit == "audio_hour":
            return unit_price_usd
        if unit in {"audio_min", "per_minute"}:
            return unit_price_usd * 60.0
        return None

    if workload in {"tts", "text_to_speech"}:
        if unit == "1m_chars":
            return unit_price_usd
        if unit == "1k_chars":
            return unit_price_usd * 1000.0
        return None

    if workload in {"image_generation", "image_gen"}:
        if unit in {"image", "per_image"}:
            return unit_price_usd * 1000.0
        return None

    if workload == "video_generation":
        if unit in {"per_second", "video_second"}:
            return unit_price_usd * 60.0
        return None

    if workload == "vision":
        return unit_price_usd if unit == "1k_images" else None

    return None


def confidence_multiplier(confidence: str) -> float:
    token = str(confidence).strip().lower()
    try:
        return ConfidenceLevel(token).price_penalty_multiplier
    except ValueError:
        return 1.30


def _throughput_to_per_hour(throughput_value: float, throughput_unit: str) -> float | None:
    token = str(throughput_unit).strip().lower()
    if token in {"per_hour", "hour", "units_per_hour", "requests_per_hour"}:
        return throughput_value
    if token in {"per_minute", "minute", "units_per_minute", "requests_per_minute"}:
        return throughput_value * 60.0
    if token in {"per_second", "second", "units_per_second", "requests_per_second"}:
        return throughput_value * 3600.0
    if token in {"audio_min_per_minute", "audio_minute_per_minute"}:
        return throughput_value * 60.0
    if token in {"audio_hour_per_hour"}:
        return throughput_value
    return None


def rank_catalog_offers(
    rows: list[object],
    allowed_providers: set[str],
    unit_name: str | None,
    top_k: int,
    monthly_budget_max_usd: float,
    comparator_mode: str,
    confidence_weighted: bool,
    workload_type: str,
    monthly_usage: float = 0.0,
    throughput_aware: bool = False,
    peak_to_avg: float = 2.5,
    util_target: float = 0.75,
    strict_capacity_check: bool = False,
    alpha: float = 1.0,
) -> tuple[list[RankedCatalogOffer], dict[str, str], int]:
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if monthly_budget_max_usd < 0:
        raise ValueError("monthly_budget_max_usd must be >= 0")
    if monthly_usage < 0:
        raise ValueError("monthly_usage must be >= 0")
    if comparator_mode not in {"normalized", "listed", "raw"}:
        raise ValueError("comparator_mode must be one of: normalized, listed, raw")
    if not 0 < util_target < 1:
        raise ValueError("util_target must be between 0 and 1")
    if peak_to_avg <= 0:
        raise ValueError("peak_to_avg must be > 0")
    if alpha < 0:
        raise ValueError("alpha must be >= 0")

    # Backward compatibility: treat "raw" as "listed".
    if comparator_mode == "raw":
        comparator_mode = "listed"

    provider_reasons: dict[str, str] = {}
    ranked_rows: list[RankedCatalogOffer] = []
    excluded_offer_count = 0

    for provider in sorted(allowed_providers):
        provider_rows = [row for row in rows if row.provider == provider]
        if not provider_rows:
            provider_reasons[provider] = "No offers for selected workload."
            continue

        if unit_name:
            provider_rows = [row for row in provider_rows if row.unit_name == unit_name]
            if not provider_rows:
                provider_reasons[provider] = "No offers for selected unit filter."
                continue

        provider_rankable = 0
        excluded_for_capacity = 0
        for row in provider_rows:
            normalized_price = normalize_unit_price_for_workload(
                unit_price_usd=row.unit_price_usd,
                unit_name=row.unit_name,
                workload_type=workload_type,
            )
            if comparator_mode == "normalized" and normalized_price is None and unit_name is not None:
                # If user already filtered to a single unit, direct same-unit comparison is valid.
                normalized_price = row.unit_price_usd
            if comparator_mode == "normalized" and normalized_price is None:
                excluded_offer_count += 1
                continue

            effective_price = (
                normalized_price if comparator_mode == "normalized" else row.unit_price_usd
            )
            if effective_price is None:
                excluded_offer_count += 1
                continue
            if confidence_weighted:
                effective_price *= confidence_multiplier(row.confidence)

            # Monthly estimate/budget checks should stay in the row's listed billing unit.
            # Comparator price is for ranking order only.
            monthly_estimate: float | None = (
                (monthly_usage * row.unit_price_usd) if monthly_usage > 0 else None
            )
            required_replicas: int | None = None
            capacity_check: str | None = None
            risk_overload = 0.20
            risk_complexity = 0.0

            if throughput_aware and monthly_usage > 0:
                required_peak_rate_per_hour = (monthly_usage / (30.0 * 24.0)) * max(peak_to_avg, 1.0)
                required_capacity_per_hour = required_peak_rate_per_hour / max(util_target, 1e-6)
                throughput_value = getattr(row, "throughput_value", None)
                throughput_unit = getattr(row, "throughput_unit", None)
                if throughput_value is not None and throughput_unit:
                    row_capacity_per_hour = _throughput_to_per_hour(
                        float(throughput_value), str(throughput_unit)
                    )
                    if row_capacity_per_hour and row_capacity_per_hour > 0:
                        required_replicas = max(
                            1,
                            int(math.ceil(required_capacity_per_hour / row_capacity_per_hour)),
                        )
                        provided_capacity = row_capacity_per_hour * required_replicas
                        margin = (provided_capacity / required_capacity_per_hour) - 1.0
                        risk_overload = min(1.0, max(0.0, math.exp(-4.0 * margin)))
                        replica_count = max(1, required_replicas)
                        risk_complexity = (
                            0.0
                            if replica_count <= 1
                            else min(1.0, 0.15 * math.log2(replica_count))
                        )
                        capacity_check = (
                            f"pass ({required_replicas} replica{'s' if required_replicas != 1 else ''})"
                        )
                        if monthly_estimate is not None:
                            monthly_estimate *= required_replicas
                    else:
                        capacity_check = "unknown throughput unit"
                        risk_overload = 0.35
                        risk_complexity = 0.10
                else:
                    capacity_check = "missing throughput metadata"
                    risk_overload = 0.35
                    risk_complexity = 0.10

                if strict_capacity_check and capacity_check and not str(capacity_check).startswith("pass"):
                    excluded_offer_count += 1
                    excluded_for_capacity += 1
                    continue
            elif throughput_aware:
                # Throughput-aware requested but usage is missing; keep a small uncertainty penalty.
                risk_overload = 0.25
                risk_complexity = 0.05
            else:
                # Cost-only ranking still carries a small uncertainty risk component.
                risk_overload = 0.20
                risk_complexity = 0.05

            total_risk = (0.7 * risk_overload) + (0.3 * risk_complexity)
            score_basis = monthly_estimate if monthly_estimate is not None else effective_price
            score = score_basis * (1.0 + alpha * total_risk)

            if monthly_budget_max_usd > 0:
                budget_value = monthly_estimate if monthly_estimate is not None else effective_price
                if budget_value > monthly_budget_max_usd:
                    excluded_offer_count += 1
                    continue

            provider_rankable += 1
            ranked_rows.append(
                RankedCatalogOffer(
                    provider=row.provider,
                    offering=row.sku_name,
                    billing=row.billing_mode,
                    listed_unit_price=row.unit_price_usd,
                    comparator_price=effective_price,
                    unit_name=row.unit_name,
                    confidence=row.confidence,
                    monthly_estimate_usd=monthly_estimate,
                    score=score,
                    risk_overload=risk_overload,
                    risk_complexity=risk_complexity,
                    total_risk=total_risk,
                    required_replicas=required_replicas,
                    capacity_check=capacity_check,
                    previous_unit_price_usd=getattr(row, "previous_unit_price_usd", None),
                    price_change_abs_usd=getattr(row, "price_change_abs_usd", None),
                    price_change_pct=getattr(row, "price_change_pct", None),
                )
            )

        if provider_rankable == 0:
            if excluded_for_capacity > 0:
                provider_reasons[provider] = "Excluded by strict capacity check (missing throughput metadata)."
            elif comparator_mode == "normalized":
                provider_reasons[provider] = "No comparable normalized unit for selected workload."
            elif monthly_budget_max_usd > 0:
                provider_reasons[provider] = "All matching offers exceed budget filter."
            else:
                provider_reasons[provider] = "No rankable offers after filters."
            continue

        provider_reasons[provider] = f"Included ({provider_rankable} rankable offers)."

    if monthly_usage > 0:
        ranked_rows.sort(
            key=lambda row: (
                row.score,
                row.monthly_estimate_usd if row.monthly_estimate_usd is not None else float("inf"),
                row.comparator_price,
                row.provider,
                row.offering,
            )
        )
    else:
        ranked_rows.sort(key=lambda row: (row.score, row.comparator_price, row.provider, row.offering))
    return ranked_rows[:top_k], provider_reasons, excluded_offer_count


def build_provider_diagnostics(
    workload_provider_ids: list[str],
    selected_global_providers: list[str],
    provider_reasons: dict[str, str],
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    selected_set = set(selected_global_providers)
    for provider_id in workload_provider_ids:
        if provider_id not in selected_set:
            diagnostics.append(
                {"provider": provider_id, "status": "excluded", "reason": "Not selected by user."}
            )
            continue
        reason = provider_reasons.get(provider_id, "No matching offers after filters.")
        status = "included" if reason.startswith("Included") else "excluded"
        diagnostics.append({"provider": provider_id, "status": status, "reason": reason})
    return diagnostics


def _build_exclusion_breakdown(
    *,
    rows: list[object],
    allowed_providers: set[str],
    unit_name: str | None,
    comparator_mode: str,
    workload_type: str,
    monthly_usage: float,
    throughput_aware: bool,
    strict_capacity_check: bool,
    peak_to_avg: float,
    util_target: float,
    monthly_budget_max_usd: float,
    confidence_weighted: bool,
) -> dict[str, int]:
    breakdown = {
        "provider_filtered_out": 0,
        "unit_mismatch": 0,
        "non_comparable_normalization": 0,
        "missing_throughput": 0,
        "budget": 0,
    }
    for row in rows:
        if row.provider not in allowed_providers:
            breakdown["provider_filtered_out"] += 1
            continue
        if unit_name and row.unit_name != unit_name:
            breakdown["unit_mismatch"] += 1
            continue

        normalized_price = normalize_unit_price_for_workload(
            unit_price_usd=row.unit_price_usd,
            unit_name=row.unit_name,
            workload_type=workload_type,
        )
        if comparator_mode == "normalized" and normalized_price is None and unit_name is not None:
            normalized_price = row.unit_price_usd
        if comparator_mode == "normalized" and normalized_price is None:
            breakdown["non_comparable_normalization"] += 1
            continue
        effective_price = normalized_price if comparator_mode == "normalized" else row.unit_price_usd
        if effective_price is None:
            breakdown["non_comparable_normalization"] += 1
            continue
        if confidence_weighted:
            effective_price *= confidence_multiplier(row.confidence)

        monthly_estimate: float | None = (
            (monthly_usage * row.unit_price_usd) if monthly_usage > 0 else None
        )

        if throughput_aware and monthly_usage > 0 and strict_capacity_check:
            throughput_value = getattr(row, "throughput_value", None)
            throughput_unit = getattr(row, "throughput_unit", None)
            if throughput_value is None or not throughput_unit:
                breakdown["missing_throughput"] += 1
                continue
            row_capacity_per_hour = _throughput_to_per_hour(
                float(throughput_value), str(throughput_unit)
            )
            if row_capacity_per_hour is None or row_capacity_per_hour <= 0:
                breakdown["missing_throughput"] += 1
                continue

            required_peak_rate_per_hour = (monthly_usage / (30.0 * 24.0)) * max(peak_to_avg, 1.0)
            required_capacity_per_hour = required_peak_rate_per_hour / max(util_target, 1e-6)
            required_replicas = max(1, int(math.ceil(required_capacity_per_hour / row_capacity_per_hour)))
            if monthly_estimate is not None:
                monthly_estimate *= required_replicas

        if monthly_budget_max_usd > 0:
            budget_value = monthly_estimate if monthly_estimate is not None else effective_price
            if budget_value > monthly_budget_max_usd:
                breakdown["budget"] += 1
                continue
    return breakdown


def run_catalog_ranking_with_relaxation(
    *,
    rows: list[object],
    allowed_providers: set[str],
    unit_name: str | None,
    top_k: int,
    monthly_budget_max_usd: float,
    comparator_mode: str,
    confidence_weighted: bool,
    workload_type: str,
    monthly_usage: float = 0.0,
    throughput_aware: bool = False,
    peak_to_avg: float = 2.5,
    util_target: float = 0.75,
    strict_capacity_check: bool = False,
    alpha: float = 1.0,
) -> CatalogRankingRun:
    """Run ranking with progressive filter relaxation for best-available results."""
    all_providers = {row.provider for row in rows}
    current_allowed = set(allowed_providers) if allowed_providers else set(all_providers)
    current_unit = unit_name
    current_budget = monthly_budget_max_usd

    steps: list[tuple[str, bool]] = [
        ("strict", True),
        ("relax_unit", bool(unit_name)),
        ("relax_budget", monthly_budget_max_usd > 0),
        ("relax_provider", current_allowed != all_providers),
    ]
    trace: list[dict[str, Any]] = []
    final_ranked: list[RankedCatalogOffer] = []
    final_provider_reasons: dict[str, str] = {}
    final_excluded = 0
    selected_step = "strict"

    for step_name, enabled in steps:
        if not enabled:
            trace.append({"step": step_name, "attempted": False, "count": 0})
            continue
        if step_name == "relax_unit":
            current_unit = None
        elif step_name == "relax_budget":
            current_budget = 0.0
        elif step_name == "relax_provider":
            current_allowed = set(all_providers)

        ranked, provider_reasons, excluded_count = rank_catalog_offers(
            rows=rows,
            allowed_providers=current_allowed,
            unit_name=current_unit,
            top_k=top_k,
            monthly_budget_max_usd=current_budget,
            comparator_mode=comparator_mode,
            confidence_weighted=confidence_weighted,
            workload_type=workload_type,
            monthly_usage=monthly_usage,
            throughput_aware=throughput_aware,
            peak_to_avg=peak_to_avg,
            util_target=util_target,
            strict_capacity_check=strict_capacity_check,
            alpha=alpha,
        )
        trace.append(
            {
                "step": step_name,
                "attempted": True,
                "count": len(ranked),
                "unit_name": current_unit,
                "monthly_budget_max_usd": current_budget,
                "provider_count": len(current_allowed),
            }
        )
        final_ranked = ranked
        final_provider_reasons = provider_reasons
        final_excluded = excluded_count
        selected_step = step_name
        if ranked:
            break

    breakdown = _build_exclusion_breakdown(
        rows=rows,
        allowed_providers=set(allowed_providers) if allowed_providers else set(all_providers),
        unit_name=unit_name,
        comparator_mode=("listed" if comparator_mode == "raw" else comparator_mode),
        workload_type=workload_type,
        monthly_usage=monthly_usage,
        throughput_aware=throughput_aware,
        strict_capacity_check=strict_capacity_check,
        peak_to_avg=peak_to_avg,
        util_target=util_target,
        monthly_budget_max_usd=monthly_budget_max_usd,
        confidence_weighted=confidence_weighted,
    )

    return CatalogRankingRun(
        ranked=final_ranked,
        provider_reasons=final_provider_reasons,
        excluded_offer_count=final_excluded,
        relaxation_trace=trace,
        exclusion_breakdown=breakdown,
        selected_step=selected_step,
    )
