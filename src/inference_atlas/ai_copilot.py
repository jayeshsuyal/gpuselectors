"""Stateful AI copilot helpers for multi-turn workload planning.

This module is deterministic and model-agnostic. It extracts structured planning
signals from user chat text and produces:
- updated conversation state
- missing field checklist
- targeted follow-up questions
- preset suggestions (cheap / balanced / reliable)
- one-click apply payload for frontend forms
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from inference_atlas.ai_inference import infer_workload_from_text

SUPPORTED_WORKLOADS = {
    "llm",
    "speech_to_text",
    "text_to_speech",
    "embeddings",
    "image_generation",
    "vision",
    "video_generation",
    "moderation",
}
SUPPORTED_PROVIDERS = {
    "anthropic",
    "assemblyai",
    "aws_rekognition",
    "baseten",
    "cohere",
    "deepgram",
    "elevenlabs",
    "fal_ai",
    "fireworks",
    "google_cloud",
    "modal",
    "openai",
    "replicate",
    "runpod",
    "together_ai",
    "voyage_ai",
}

TRAFFIC_PATTERNS = {"steady", "business_hours", "bursty"}
LATENCY_PRIORITIES = {"strict", "balanced", "flexible"}
RISK_PREFERENCES = {"cheap", "balanced", "reliable"}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _safe_float(value: str) -> float | None:
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return None


def _extract_budget_usd(text: str) -> float | None:
    lower = text.lower()
    # "$50", "$1,200/month", "budget 200 usd"
    for pattern in (
        r"\$+\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
        r"\b([0-9][0-9,]*(?:\.[0-9]+)?)\s*(?:usd|dollars?)\b",
        r"\bbudget(?:\s+is|\s+of|\s+under|\s+around)?\s*\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\b",
    ):
        match = re.search(pattern, lower)
        if not match:
            continue
        parsed = _safe_float(match.group(1))
        if parsed is not None and parsed >= 0:
            return parsed
    return None


def _extract_tokens_per_day(text: str) -> float | None:
    lower = text.lower()
    patterns = (
        r"([0-9][0-9,]*(?:\.[0-9]+)?)\s*(m|mn|million)\s*tokens?\s*(?:/|per)?\s*day",
        r"([0-9][0-9,]*(?:\.[0-9]+)?)\s*(b|bn|billion)\s*tokens?\s*(?:/|per)?\s*day",
        r"([0-9][0-9,]*(?:\.[0-9]+)?)\s*tokens?\s*(?:/|per)?\s*day",
    )
    for pattern in patterns:
        match = re.search(pattern, lower)
        if not match:
            continue
        base = _safe_float(match.group(1))
        if base is None:
            continue
        suffix = match.group(2) if match.lastindex and match.lastindex >= 2 else ""
        if suffix in {"m", "mn", "million"}:
            return base * 1_000_000
        if suffix in {"b", "bn", "billion"}:
            return base * 1_000_000_000
        return base
    return None


def _extract_monthly_usage(text: str) -> float | None:
    lower = text.lower()
    for pattern in (
        r"([0-9][0-9,]*(?:\.[0-9]+)?)\s*(?:/|per)?\s*month",
        r"monthly\s+usage(?:\s+is|\s+of)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
    ):
        match = re.search(pattern, lower)
        if not match:
            continue
        parsed = _safe_float(match.group(1))
        if parsed is not None and parsed >= 0:
            return parsed
    return None


def _extract_peak_to_avg(text: str) -> float | None:
    lower = text.lower()
    match = re.search(
        r"(?:peak(?:\s*to\s*avg|\s*\/\s*avg)|peak_to_avg)\s*(?:=|is|of)?\s*([0-9]+(?:\.[0-9]+)?)",
        lower,
    )
    if not match:
        return None
    value = _safe_float(match.group(1))
    if value is None or value <= 0:
        return None
    return value


def _extract_traffic_pattern(text: str) -> str | None:
    lower = text.lower()
    if "business" in lower and "hour" in lower:
        return "business_hours"
    if "bursty" in lower or "spiky" in lower or "spike" in lower:
        return "bursty"
    if "steady" in lower or "flat" in lower:
        return "steady"
    return None


def _extract_latency_priority(text: str) -> str | None:
    lower = text.lower()
    if (
        "low latency" in lower
        or "strict latency" in lower
        or "real-time" in lower
        or re.search(r"\blatency\b.{0,20}\bstrict\b", lower)
        or re.search(r"\bstrict\b.{0,20}\blatency\b", lower)
    ):
        return "strict"
    if "latency not important" in lower or "latency flexible" in lower:
        return "flexible"
    if "latency" in lower:
        return "balanced"
    return None


def _extract_risk_preference(text: str) -> str | None:
    lower = text.lower()
    if any(token in lower for token in ("cheapest", "low cost", "lowest cost", "budget first")):
        return "cheap"
    if any(token in lower for token in ("reliable", "stable", "production", "sla")):
        return "reliable"
    if "balanced" in lower:
        return "balanced"
    return None


def _extract_provider_mentions(text: str) -> list[str]:
    lower = text.lower()
    providers = []
    keywords = {
        "openai": ["openai", "gpt"],
        "anthropic": ["anthropic", "claude"],
        "fireworks": ["fireworks"],
        "together_ai": ["together", "together ai"],
        "deepgram": ["deepgram"],
        "assemblyai": ["assemblyai", "assembly ai"],
        "elevenlabs": ["elevenlabs", "eleven labs"],
        "cohere": ["cohere"],
        "voyage_ai": ["voyage", "voyage ai"],
        "google_cloud": ["google cloud", "gcp"],
        "aws_rekognition": ["aws rekognition", "rekognition"],
        "runpod": ["runpod"],
        "modal": ["modal"],
        "replicate": ["replicate"],
        "baseten": ["baseten"],
        "fal_ai": ["fal", "fal ai"],
    }
    for provider, terms in keywords.items():
        if any(term in lower for term in terms):
            providers.append(provider)
    return sorted(set(providers))


def _sanitize_spec(spec: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(spec)

    workload = str(out.get("workload_type") or "llm")
    out["workload_type"] = workload if workload in SUPPORTED_WORKLOADS else "llm"

    if out.get("tokens_per_day") is not None:
        value = float(out["tokens_per_day"])
        out["tokens_per_day"] = _clamp(value, 1.0, 1_000_000_000_000.0)

    if out.get("monthly_usage") is not None:
        value = float(out["monthly_usage"])
        out["monthly_usage"] = _clamp(value, 0.0, 1_000_000_000_000.0)

    if out.get("monthly_budget_max_usd") is not None:
        value = float(out["monthly_budget_max_usd"])
        out["monthly_budget_max_usd"] = _clamp(value, 0.0, 1_000_000_000.0)

    if out.get("peak_to_avg") is not None:
        out["peak_to_avg"] = _clamp(float(out["peak_to_avg"]), 1.0, 10.0)

    traffic_pattern = out.get("traffic_pattern")
    if traffic_pattern not in TRAFFIC_PATTERNS:
        out.pop("traffic_pattern", None)

    latency = out.get("latency_priority")
    if latency not in LATENCY_PRIORITIES:
        out.pop("latency_priority", None)

    risk_pref = out.get("risk_preference")
    if risk_pref not in RISK_PREFERENCES:
        out.pop("risk_preference", None)

    providers = out.get("provider_ids")
    if providers:
        normalized = sorted({str(provider).strip() for provider in providers if str(provider).strip()})
        out["provider_ids"] = [provider for provider in normalized if provider in SUPPORTED_PROVIDERS]
    else:
        out.pop("provider_ids", None)

    return out


def extract_spec_updates(user_text: str, prior_spec: dict[str, Any] | None = None) -> dict[str, Any]:
    """Extract structured updates from a user utterance."""
    spec = deepcopy(prior_spec) if prior_spec else {}

    inferred_workload = infer_workload_from_text(user_text, str(spec.get("workload_type", "llm")))
    if inferred_workload in SUPPORTED_WORKLOADS:
        spec["workload_type"] = inferred_workload

    budget = _extract_budget_usd(user_text)
    if budget is not None:
        spec["monthly_budget_max_usd"] = budget

    tokens_per_day = _extract_tokens_per_day(user_text)
    if tokens_per_day is not None:
        spec["tokens_per_day"] = tokens_per_day

    monthly_usage = _extract_monthly_usage(user_text)
    if monthly_usage is not None:
        spec["monthly_usage"] = monthly_usage

    peak_to_avg = _extract_peak_to_avg(user_text)
    if peak_to_avg is not None:
        spec["peak_to_avg"] = peak_to_avg

    traffic_pattern = _extract_traffic_pattern(user_text)
    if traffic_pattern in TRAFFIC_PATTERNS:
        spec["traffic_pattern"] = traffic_pattern

    latency = _extract_latency_priority(user_text)
    if latency in LATENCY_PRIORITIES:
        spec["latency_priority"] = latency

    risk_pref = _extract_risk_preference(user_text)
    if risk_pref in RISK_PREFERENCES:
        spec["risk_preference"] = risk_pref

    mentioned = _extract_provider_mentions(user_text)
    if mentioned:
        existing = spec.get("provider_ids") or []
        spec["provider_ids"] = sorted(set(existing) | set(mentioned))

    return _sanitize_spec(spec)


def get_missing_fields(spec: dict[str, Any]) -> list[str]:
    """Return high-value missing fields needed for confident recommendations."""
    workload = str(spec.get("workload_type") or "llm")
    missing: list[str] = []

    if workload == "llm":
        if not spec.get("tokens_per_day"):
            missing.append("tokens_per_day")
        if not spec.get("traffic_pattern"):
            missing.append("traffic_pattern")
    else:
        if not spec.get("monthly_usage"):
            missing.append("monthly_usage")

    if spec.get("monthly_budget_max_usd") is None:
        missing.append("monthly_budget_max_usd")
    if not spec.get("latency_priority"):
        missing.append("latency_priority")
    return missing


def get_follow_up_questions(spec: dict[str, Any], limit: int = 3) -> list[str]:
    """Generate targeted follow-up questions based on missing fields."""
    questions: list[str] = []
    for field in get_missing_fields(spec):
        if field == "tokens_per_day":
            questions.append("What is your expected daily token volume (tokens/day)?")
        elif field == "monthly_usage":
            questions.append("What is your expected monthly usage in this unit?")
        elif field == "traffic_pattern":
            questions.append("Is your traffic steady, business-hours heavy, or bursty?")
        elif field == "monthly_budget_max_usd":
            questions.append("Do you have a monthly budget cap in USD?")
        elif field == "latency_priority":
            questions.append("How latency-sensitive is this use case: strict, balanced, or flexible?")
    return questions[:limit]


def get_suggested_presets(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Return workload-aware preset suggestions for quick apply."""
    workload = str(spec.get("workload_type") or "llm")
    if workload == "llm":
        return [
            {
                "id": "cheap",
                "label": "Cheap",
                "rationale": "Minimize monthly cost with higher utilization.",
                "params": {"alpha": 0.7, "util_target": 0.85, "peak_to_avg": 2.0},
            },
            {
                "id": "balanced",
                "label": "Balanced",
                "rationale": "Balance cost and reliability for most production workloads.",
                "params": {"alpha": 1.0, "util_target": 0.75, "peak_to_avg": 2.5},
            },
            {
                "id": "reliable",
                "label": "Reliable",
                "rationale": "Prioritize headroom and operational stability.",
                "params": {"alpha": 1.4, "util_target": 0.65, "peak_to_avg": 3.0},
            },
        ]
    return [
        {
            "id": "cheap",
            "label": "Cheap",
            "rationale": "Rank by listed/normalized unit cost first.",
            "params": {"confidence_weighted": False, "throughput_aware": False},
        },
        {
            "id": "balanced",
            "label": "Balanced",
            "rationale": "Keep confidence penalties enabled and optional throughput checks.",
            "params": {"confidence_weighted": True, "throughput_aware": True, "strict_capacity_check": False},
        },
        {
            "id": "reliable",
            "label": "Reliable",
            "rationale": "Favor offers with known throughput and stricter capacity checks.",
            "params": {"confidence_weighted": True, "throughput_aware": True, "strict_capacity_check": True},
        },
    ]


def build_apply_payload(spec: dict[str, Any]) -> dict[str, Any]:
    """Build one-click payload for frontend forms using extracted state."""
    clean_spec = _sanitize_spec(spec)
    workload = str(clean_spec.get("workload_type") or "llm")
    provider_ids = list(clean_spec.get("provider_ids") or [])
    if workload == "llm":
        return {
            "mode": "llm",
            "values": {
                "tokens_per_day": float(clean_spec.get("tokens_per_day") or 5_000_000.0),
                "traffic_pattern": str(clean_spec.get("traffic_pattern") or "business_hours"),
                "monthly_budget_max_usd": float(clean_spec.get("monthly_budget_max_usd") or 0.0),
                "peak_to_avg": float(clean_spec.get("peak_to_avg") or 2.5),
                "provider_ids": provider_ids,
            },
        }
    return {
        "mode": "catalog",
        "values": {
            "workload_type": workload,
            "monthly_usage": float(clean_spec.get("monthly_usage") or 1000.0),
            "monthly_budget_max_usd": float(clean_spec.get("monthly_budget_max_usd") or 0.0),
            "provider_ids": provider_ids,
        },
    }


def next_copilot_turn(
    *,
    user_text: str,
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Advance copilot state by one user message."""
    current = deepcopy(state) if state else {}
    messages = list(current.get("messages") or [])
    messages.append({"role": "user", "content": user_text})

    prior_spec = dict(current.get("extracted_spec") or {})
    extracted_spec = extract_spec_updates(user_text, prior_spec)
    missing_fields = get_missing_fields(extracted_spec)
    follow_ups = get_follow_up_questions(extracted_spec)
    presets = get_suggested_presets(extracted_spec)
    apply_payload = build_apply_payload(extracted_spec)

    ready_to_rank = len(missing_fields) == 0

    return {
        "messages": messages,
        "extracted_spec": extracted_spec,
        "missing_fields": missing_fields,
        "next_questions": follow_ups,
        "suggested_presets": presets,
        "apply_payload": apply_payload,
        "ready_to_rank": ready_to_rank,
    }
