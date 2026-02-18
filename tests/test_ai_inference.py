from __future__ import annotations

from dataclasses import dataclass

from inference_atlas.ai_inference import (
    build_catalog_context,
    infer_workload_from_text,
    resolve_ai_scope,
)


@dataclass(frozen=True)
class _Row:
    provider: str
    workload_type: str
    sku_name: str
    model_key: str
    billing_mode: str
    unit_price_usd: float
    unit_name: str
    region: str
    confidence: str
    source_kind: str


def test_infer_workload_from_text_detects_stt() -> None:
    inferred = infer_workload_from_text("Need low-cost speech-to-text under $50", "llm")
    assert inferred == "speech_to_text"


def test_infer_workload_from_text_detects_stt_with_common_typo() -> None:
    inferred = infer_workload_from_text("give top speach to text models", "llm")
    assert inferred == "speech_to_text"


def test_infer_workload_from_text_detects_stt_with_spich_typo() -> None:
    inferred = infer_workload_from_text(
        "give me a spich to text model for low cost personal project",
        "llm",
    )
    assert inferred == "speech_to_text"


def test_infer_workload_from_text_detects_embeddings_typo() -> None:
    inferred = infer_workload_from_text("need embeding model for retrieval", "llm")
    assert inferred == "embeddings"


def test_infer_workload_from_text_detects_tts() -> None:
    inferred = infer_workload_from_text("best tts for a small app", "llm")
    assert inferred == "text_to_speech"


def test_infer_workload_from_text_prefers_explicit_tts_over_llm_when_both_present() -> None:
    inferred = infer_workload_from_text(
        "so im not optimizing for llm im optimizing for tts can u suggest something",
        "llm",
    )
    assert inferred == "text_to_speech"


def test_infer_workload_from_text_prefers_embeddings_when_llm_is_negated() -> None:
    inferred = infer_workload_from_text(
        "not llm; optimizing for embeddings and vector search",
        "llm",
    )
    assert inferred == "embeddings"


def test_infer_workload_from_text_prefers_stt_when_tts_is_negated() -> None:
    inferred = infer_workload_from_text(
        "not text to speech, need speech to text for interviews",
        "llm",
    )
    assert inferred == "speech_to_text"


def test_infer_workload_from_text_prefers_vision_over_image_gen_when_negated() -> None:
    inferred = infer_workload_from_text(
        "not image generation, focused on vision OCR and understanding",
        "llm",
    )
    assert inferred == "vision"


def test_infer_workload_from_text_prefers_moderation_when_embeddings_negated() -> None:
    inferred = infer_workload_from_text(
        "do not use embeddings; need moderation safety filter",
        "llm",
    )
    assert inferred == "moderation"


def test_infer_workload_from_text_detects_vision_typo() -> None:
    inferred = infer_workload_from_text("need a visiom api for OCR", "llm")
    assert inferred == "vision"


def test_infer_workload_from_text_detects_image_generation() -> None:
    inferred = infer_workload_from_text("cheap text to image options", "llm")
    assert inferred == "image_generation"


def test_infer_workload_from_text_detects_moderation_typo() -> None:
    inferred = infer_workload_from_text("need moderatiom filter api", "llm")
    assert inferred == "moderation"


def test_infer_workload_from_text_ambiguous_defaults_to_selected() -> None:
    inferred = infer_workload_from_text("need low cost ai model", "llm")
    assert inferred == "llm"


def test_resolve_ai_scope_prefers_inferred_workload() -> None:
    rows = [
        _Row("openai", "llm", "gpt", "gpt", "per_token", 1.0, "per_1m_tokens", "global", "official", "vendor"),
        _Row("deepgram", "speech_to_text", "nova", "nova", "per_token", 0.5, "per_1m_chars", "global", "official", "vendor"),
    ]
    workload, providers = resolve_ai_scope(
        ai_text="speech to text options",
        selected_workload="llm",
        selected_providers=["openai"],
        rows=rows,
    )
    assert workload == "speech_to_text"
    assert providers == ["deepgram"]


def test_resolve_ai_scope_falls_back_to_selected_workload() -> None:
    rows = [
        _Row("openai", "llm", "gpt", "gpt", "per_token", 1.0, "per_1m_tokens", "global", "official", "vendor"),
    ]
    workload, providers = resolve_ai_scope(
        ai_text="unknown workload phrase",
        selected_workload="llm",
        selected_providers=[],
        rows=rows,
    )
    assert workload == "llm"
    assert providers == ["openai"]


def test_build_catalog_context_reports_empty() -> None:
    rows = [
        _Row("openai", "llm", "gpt", "gpt", "per_token", 1.0, "per_1m_tokens", "global", "official", "vendor"),
    ]
    context = build_catalog_context(
        selected_workload="speech_to_text",
        selected_providers=["openai"],
        rows=rows,
    )
    assert "No matching catalog rows" in context
