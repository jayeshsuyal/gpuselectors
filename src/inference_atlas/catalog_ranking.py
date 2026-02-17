"""Catalog ranking helpers for non-LLM workload offer comparisons."""

from __future__ import annotations

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
    exclusion_reason: str | None = None


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
) -> tuple[list[RankedCatalogOffer], dict[str, str], int]:
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
        for row in provider_rows:
            normalized_price = normalize_unit_price_for_workload(
                unit_price_usd=row.unit_price_usd,
                unit_name=row.unit_name,
                workload_type=workload_type,
            )
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
            if monthly_budget_max_usd > 0 and effective_price > monthly_budget_max_usd:
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
                    monthly_estimate_usd=(
                        (monthly_usage * effective_price) if monthly_usage > 0 else None
                    ),
                )
            )

        if provider_rankable == 0:
            if comparator_mode == "normalized":
                provider_reasons[provider] = "No comparable normalized unit for selected workload."
            elif monthly_budget_max_usd > 0:
                provider_reasons[provider] = "All matching offers exceed budget filter."
            else:
                provider_reasons[provider] = "No rankable offers after filters."
            continue

        provider_reasons[provider] = f"Included ({provider_rankable} rankable offers)."

    ranked_rows.sort(key=lambda row: row.comparator_price)
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

