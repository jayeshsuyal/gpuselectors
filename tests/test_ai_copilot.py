from __future__ import annotations

from inference_atlas.ai_copilot import (
    build_apply_payload,
    extract_spec_updates,
    get_follow_up_questions,
    get_missing_fields,
    get_suggested_presets,
    next_copilot_turn,
)


def test_extract_spec_updates_llm_budget_tokens_and_pattern() -> None:
    spec = extract_spec_updates("Need llm around 5 million tokens/day with $300 budget and bursty traffic")
    assert spec["workload_type"] == "llm"
    assert spec["tokens_per_day"] == 5_000_000
    assert spec["monthly_budget_max_usd"] == 300
    assert spec["traffic_pattern"] == "bursty"


def test_extract_spec_updates_stt_and_provider_mentions() -> None:
    spec = extract_spec_updates("Need speech to text with Deepgram and OpenAI")
    assert spec["workload_type"] == "speech_to_text"
    assert "deepgram" in spec["provider_ids"]
    assert "openai" in spec["provider_ids"]


def test_extract_spec_updates_filters_unknown_providers() -> None:
    spec = extract_spec_updates("Use openai and unknown_provider_xyz")
    assert spec["provider_ids"] == ["openai"]


def test_get_missing_fields_llm() -> None:
    missing = get_missing_fields({"workload_type": "llm"})
    assert "tokens_per_day" in missing
    assert "traffic_pattern" in missing
    assert "latency_priority" in missing


def test_get_follow_up_questions_are_targeted() -> None:
    questions = get_follow_up_questions({"workload_type": "llm"})
    assert questions
    assert any("daily token volume" in q.lower() for q in questions)


def test_get_suggested_presets_returns_three_options() -> None:
    presets = get_suggested_presets({"workload_type": "text_to_speech"})
    assert [preset["id"] for preset in presets] == ["cheap", "balanced", "reliable"]


def test_build_apply_payload_llm_defaults() -> None:
    payload = build_apply_payload({"workload_type": "llm"})
    assert payload["mode"] == "llm"
    assert payload["values"]["tokens_per_day"] == 5_000_000.0
    assert payload["values"]["traffic_pattern"] == "business_hours"


def test_build_apply_payload_non_llm_defaults() -> None:
    payload = build_apply_payload({"workload_type": "vision"})
    assert payload["mode"] == "catalog"
    assert payload["values"]["workload_type"] == "vision"
    assert payload["values"]["monthly_usage"] == 1000.0


def test_next_copilot_turn_merges_state_across_turns() -> None:
    state_1 = next_copilot_turn(user_text="Need low cost tts under $50")
    state_2 = next_copilot_turn(
        user_text="monthly usage 3000 and latency is strict",
        state=state_1,
    )
    assert len(state_2["messages"]) == 2
    assert state_2["extracted_spec"]["workload_type"] == "text_to_speech"
    assert state_2["extracted_spec"]["monthly_budget_max_usd"] == 50
    assert state_2["extracted_spec"]["monthly_usage"] == 3000
    assert state_2["extracted_spec"]["latency_priority"] == "strict"


def test_next_copilot_turn_sets_ready_flag() -> None:
    state = next_copilot_turn(
        user_text="Need llm with 5 million tokens/day, business hours, strict latency and budget $500",
    )
    assert state["ready_to_rank"] is True
    assert state["missing_fields"] == []
