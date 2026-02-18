from __future__ import annotations

import os

import pytest

from inference_atlas.llm.gpt_5_2_adapter import GPT52Adapter
from inference_atlas.llm.opus_4_6_adapter import Opus46Adapter
from inference_atlas.llm.schema import WorkloadSpec


def test_gpt_parse_workload_extracts_json_from_text_wrapper() -> None:
    adapter = GPT52Adapter(api_key="test-key", client=object())
    adapter._generate_text = lambda _s, _u: (  # type: ignore[attr-defined]
        "Here is data:\n"
        '{"tokens_per_day": 5000000, "pattern": "steady", "model_key": "llama_70b", "latency_requirement_ms": null}'
    )
    payload = adapter.parse_workload("test")
    assert payload["tokens_per_day"] == 5_000_000
    assert payload["pattern"] == "steady"


def test_gpt_parser_prompt_not_hardcoded_to_legacy_model_list() -> None:
    adapter = GPT52Adapter(api_key="test-key", client=object())
    captured: dict[str, str] = {}

    def _fake_generate_text(_system: str, user_prompt: str) -> str:
        captured["prompt"] = user_prompt
        return '{"tokens_per_day": 5000000, "pattern": "steady", "model_key": "llama_70b", "latency_requirement_ms": null}'

    adapter._generate_text = _fake_generate_text  # type: ignore[attr-defined]
    adapter.parse_workload("test")
    assert "must be one of" not in captured["prompt"]


def test_opus_parse_workload_extracts_json_from_text_wrapper() -> None:
    adapter = Opus46Adapter(api_key="test-key", client=object())
    adapter._generate_text = lambda _s, _u: (  # type: ignore[attr-defined]
        '{"tokens_per_day": 2500000, "pattern": "business_hours", "model_key": "llama_8b"}'
    )
    payload = adapter.parse_workload("test")
    assert payload["tokens_per_day"] == 2_500_000
    assert payload["pattern"] == "business_hours"


def test_opus_parser_prompt_not_hardcoded_to_legacy_model_list() -> None:
    adapter = Opus46Adapter(api_key="test-key", client=object())
    captured: dict[str, str] = {}

    def _fake_generate_text(_system: str, user_prompt: str) -> str:
        captured["prompt"] = user_prompt
        return '{"tokens_per_day": 2500000, "pattern": "business_hours", "model_key": "llama_70b"}'

    adapter._generate_text = _fake_generate_text  # type: ignore[attr-defined]
    adapter.parse_workload("test")
    assert "must be one of" not in captured["prompt"]


def test_gpt_explain_returns_text() -> None:
    adapter = GPT52Adapter(api_key="test-key", client=object())
    adapter._generate_text = lambda _s, _u: "grounded explanation"  # type: ignore[attr-defined]
    workload = WorkloadSpec(tokens_per_day=1_000_000, pattern="steady", model_key="llama_8b")
    explanation = adapter.explain("summary", workload)
    assert explanation == "grounded explanation"


def test_opus_explain_returns_text() -> None:
    adapter = Opus46Adapter(api_key="test-key", client=object())
    adapter._generate_text = lambda _s, _u: "ops-grade explanation"  # type: ignore[attr-defined]
    workload = WorkloadSpec(tokens_per_day=1_000_000, pattern="steady", model_key="llama_8b")
    explanation = adapter.explain("summary", workload)
    assert explanation == "ops-grade explanation"


def test_gpt_missing_key_raises() -> None:
    with pytest.raises(ValueError, match="OPENAI_API_KEY not set"):
        GPT52Adapter(api_key="")


def test_opus_missing_key_raises() -> None:
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY not set"):
        Opus46Adapter(api_key="")


def test_gpt_constructor_validates_timeout() -> None:
    with pytest.raises(ValueError, match="timeout_sec must be > 0"):
        GPT52Adapter(api_key="x", timeout_sec=0, client=object())


def test_opus_constructor_validates_timeout() -> None:
    with pytest.raises(ValueError, match="timeout_sec must be > 0"):
        Opus46Adapter(api_key="x", timeout_sec=0, client=object())


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No API key")
def test_gpt_parse_workload_real() -> None:
    adapter = GPT52Adapter()
    result = adapter.parse_workload("Chat app with 5M tokens/day, steady traffic, Llama 70B")
    assert "tokens_per_day" in result
    assert result["pattern"] in ["steady", "business_hours", "bursty"]


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="No API key")
def test_opus_explain_real() -> None:
    adapter = Opus46Adapter()
    workload = WorkloadSpec(
        tokens_per_day=5_000_000,
        pattern="steady",
        model_key="llama_70b",
    )
    explanation = adapter.explain("fireworks - H100 80GB, $835/mo", workload)
    assert len(explanation) > 50
