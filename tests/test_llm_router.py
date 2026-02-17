from __future__ import annotations

from typing import Any

import pytest

from inference_atlas.llm.base import LLMAdapter
from inference_atlas.llm.router import LLMRouter, RouterConfig
from inference_atlas.llm.schema import WorkloadSpec, validate_workload_payload


class _SuccessAdapter(LLMAdapter):
    def __init__(self, provider_name: str, payload: dict[str, Any], explanation: str = "ok") -> None:
        self.provider_name = provider_name
        self._payload = payload
        self._explanation = explanation

    def parse_workload(self, user_text: str) -> dict[str, Any]:
        return self._payload

    def explain(self, recommendation_summary: str, workload: WorkloadSpec) -> str:
        return self._explanation


class _FailAdapter(LLMAdapter):
    def __init__(self, provider_name: str) -> None:
        self.provider_name = provider_name

    def parse_workload(self, user_text: str) -> dict[str, Any]:
        raise RuntimeError("parse failed")

    def explain(self, recommendation_summary: str, workload: WorkloadSpec) -> str:
        raise RuntimeError("explain failed")


def test_validate_workload_payload_success_with_pattern_normalization() -> None:
    spec = validate_workload_payload(
        {
            "tokens_per_day": "5000000",
            "pattern": "Business Hours",
            "model_key": "llama_70b",
            "latency_requirement_ms": "250",
        }
    )
    assert spec.tokens_per_day == 5_000_000.0
    assert spec.pattern == "business_hours"
    assert spec.model_key == "llama_70b"
    assert spec.latency_requirement_ms == 250.0


def test_validate_workload_payload_rejects_invalid_tokens() -> None:
    with pytest.raises(ValueError, match="tokens_per_day must be > 0"):
        validate_workload_payload(
            {"tokens_per_day": 0, "pattern": "steady", "model_key": "llama_70b"}
        )


def test_validate_workload_payload_rejects_invalid_pattern() -> None:
    with pytest.raises(ValueError, match="Invalid pattern"):
        validate_workload_payload(
            {"tokens_per_day": 1, "pattern": "weekends", "model_key": "llama_70b"}
        )


def test_router_uses_fallback_for_parse() -> None:
    primary = _FailAdapter("gpt_5_2")
    fallback = _SuccessAdapter(
        "opus_4_6",
        {"tokens_per_day": 1234, "pattern": "steady", "model_key": "llama_8b"},
    )
    router = LLMRouter(
        adapters={"gpt_5_2": primary, "opus_4_6": fallback},
        config=RouterConfig(primary_provider="gpt_5_2", fallback_provider="opus_4_6"),
    )
    spec = router.parse_workload("test")
    assert spec.model_key == "llama_8b"


def test_router_raises_when_all_parse_providers_fail() -> None:
    router = LLMRouter(
        adapters={"gpt_5_2": _FailAdapter("gpt_5_2"), "opus_4_6": _FailAdapter("opus_4_6")},
        config=RouterConfig(primary_provider="gpt_5_2", fallback_provider="opus_4_6"),
    )
    with pytest.raises(RuntimeError, match="All LLM providers failed to parse workload"):
        router.parse_workload("test")


def test_router_parse_failure_message_includes_provider_error_types() -> None:
    router = LLMRouter(
        adapters={"gpt_5_2": _FailAdapter("gpt_5_2"), "opus_4_6": _FailAdapter("opus_4_6")},
        config=RouterConfig(primary_provider="gpt_5_2", fallback_provider="opus_4_6"),
    )
    with pytest.raises(RuntimeError) as exc_info:
        router.parse_workload("test")
    msg = str(exc_info.value)
    assert "gpt_5_2 [RuntimeError]" in msg
    assert "opus_4_6 [RuntimeError]" in msg


def test_router_uses_fallback_for_explain() -> None:
    primary = _FailAdapter("gpt_5_2")
    fallback = _SuccessAdapter(
        "opus_4_6",
        {"tokens_per_day": 1234, "pattern": "steady", "model_key": "llama_8b"},
        explanation="fallback explanation",
    )
    router = LLMRouter(
        adapters={"gpt_5_2": primary, "opus_4_6": fallback},
        config=RouterConfig(primary_provider="gpt_5_2", fallback_provider="opus_4_6"),
    )
    workload = WorkloadSpec(tokens_per_day=1000, pattern="steady", model_key="llama_8b")
    assert router.explain("summary", workload) == "fallback explanation"
