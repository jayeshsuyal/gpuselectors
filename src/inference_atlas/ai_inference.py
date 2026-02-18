"""Helpers for workload-aware AI context selection."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Iterable


WORKLOAD_KEYWORDS: dict[str, list[str]] = {
    "speech_to_text": [
        "speech to text",
        "speech-to-text",
        "speach to text",
        "voice to text",
        "stt",
        "transcription",
        "transcribe",
    ],
    "text_to_speech": ["text to speech", "text-to-speech", "tts", "voice synthesis"],
    "embeddings": ["embedding", "vector search", "semantic search", "retrieval"],
    "image_generation": ["image generation", "text to image", "text-to-image", "diffusion"],
    "vision": ["vision", "image understanding", "ocr", "visual qa", "multimodal"],
    "video_generation": ["video generation", "text to video", "text-to-video"],
    "moderation": ["moderation", "safety", "content filter"],
    "llm": ["llm", "chat", "completion", "text generation", "inference"],
}


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _tokenize(text: str) -> list[str]:
    normalized = _normalize_text(text)
    return [token for token in normalized.split() if token]


def _best_token_similarity(query_tokens: list[str], keyword_tokens: list[str]) -> float:
    if not query_tokens or not keyword_tokens:
        return 0.0
    best = 0.0
    for query_token in query_tokens:
        for keyword_token in keyword_tokens:
            score = SequenceMatcher(None, query_token, keyword_token).ratio()
            if score > best:
                best = score
    return best


def infer_workload_from_text(
    ai_text: str,
    default_workload: str,
    available_workloads: set[str] | None = None,
) -> str:
    """Infer workload intent from free-form text with fuzzy matching."""
    normalized_text = _normalize_text(ai_text)
    query_tokens = _tokenize(ai_text)
    candidates = (
        sorted(available_workloads)
        if available_workloads is not None
        else list(WORKLOAD_KEYWORDS.keys())
    )

    best_workload = default_workload
    best_score = 0.0

    for workload in candidates:
        keywords = WORKLOAD_KEYWORDS.get(workload, [])
        workload_score = 0.0
        for keyword in keywords:
            keyword_norm = _normalize_text(keyword)
            keyword_tokens = _tokenize(keyword)
            if not keyword_norm:
                continue

            if keyword_norm in normalized_text:
                workload_score = max(workload_score, 1.0)
                continue

            if keyword_tokens:
                token_overlap = len(set(query_tokens) & set(keyword_tokens)) / len(keyword_tokens)
                workload_score = max(workload_score, 0.8 * token_overlap)

                fuzzy = _best_token_similarity(query_tokens, keyword_tokens)
                if fuzzy >= 0.9:
                    workload_score = max(workload_score, 0.75)
                elif fuzzy >= 0.84:
                    workload_score = max(workload_score, 0.62)

        if workload_score > best_score:
            best_score = workload_score
            best_workload = workload

    if best_score < 0.55:
        return default_workload
    if available_workloads is not None and best_workload not in available_workloads:
        return default_workload
    return best_workload


def resolve_ai_scope(
    *,
    ai_text: str,
    selected_workload: str,
    selected_providers: Iterable[str],
    rows: list[object],
) -> tuple[str | None, list[str]]:
    """Resolve workload/providers for AI grounding with safe fallbacks."""
    selected_provider_set = set(selected_providers)
    available_workloads = {row.workload_type for row in rows}
    inferred_workload = infer_workload_from_text(
        ai_text,
        default_workload=selected_workload,
        available_workloads=available_workloads,
    )

    for workload in [inferred_workload, selected_workload]:
        workload_rows = [row for row in rows if row.workload_type == workload]
        if not workload_rows:
            continue
        workload_providers = sorted({row.provider for row in workload_rows})
        in_scope_providers = sorted(selected_provider_set & set(workload_providers))
        chosen_providers = in_scope_providers or workload_providers
        if chosen_providers:
            return workload, chosen_providers

    all_providers = sorted({row.provider for row in rows})
    in_scope_providers = sorted(selected_provider_set & set(all_providers))
    return None, in_scope_providers or all_providers


def build_catalog_context(
    *,
    selected_workload: str | None,
    selected_providers: list[str],
    rows: list[object],
    max_rows: int = 40,
) -> str:
    """Build compact, grounded catalog context for AI prompts."""
    selected_provider_set = set(selected_providers)
    filtered = [
        row
        for row in rows
        if (selected_workload is None or row.workload_type == selected_workload)
        and row.provider in selected_provider_set
    ]
    if not filtered:
        return "No matching catalog rows for current workload/provider filters."

    filtered.sort(key=lambda row: row.unit_price_usd)
    sample = filtered[:max_rows]
    providers = sorted({row.provider for row in filtered})
    units = sorted({row.unit_name for row in filtered})

    lines = [
        f"workload={selected_workload if selected_workload is not None else 'all'}",
        f"providers={','.join(providers)}",
        f"rows_total={len(filtered)}",
        f"units={','.join(units)}",
        "rows_sample_start",
    ]
    for row in sample:
        lines.append(
            "|".join(
                [
                    row.provider,
                    row.sku_name,
                    row.model_key,
                    row.billing_mode,
                    f"{row.unit_price_usd:.8g}",
                    row.unit_name,
                    row.region,
                    row.confidence,
                    row.source_kind,
                ]
            )
        )
    lines.append("rows_sample_end")
    return "\n".join(lines)
