"""Prompt builders shared by LLM adapters."""

from __future__ import annotations


def build_workload_parser_prompt(user_text: str) -> str:
    """Build parser prompt for converting free text to workload JSON."""
    return f"""You are a workload parser for inference cost analysis.

Extract these fields from the user's description:
- tokens_per_day (float): Total tokens per day
- pattern (string): Traffic pattern, one of: "steady", "business_hours", "bursty"
- model_key (string): Model identifier used by the planner
- latency_requirement_ms (float, optional): Max latency in milliseconds, or null if not specified

Return ONLY valid JSON with exactly these keys:
tokens_per_day, pattern, model_key, latency_requirement_ms

Model key guidance:
- If the user names a known canonical key (e.g. "llama_70b"), return it as-is.
- If the user names a model family/size (e.g. "Llama 70B"), map it to a planner-friendly key.
- If uncertain, return a normalized non-empty model key string instead of inventing metadata.

Examples:
User: "Chat app with 1000 daily active users, steady traffic, Llama 70B"
Response: {{"tokens_per_day": 5000000, "pattern": "steady", "model_key": "llama_70b", "latency_requirement_ms": null}}

User: "API serving 10M tokens/day during business hours, need <200ms P99, using Mistral 7B"
Response: {{"tokens_per_day": 10000000, "pattern": "business_hours", "model_key": "mistral_7b", "latency_requirement_ms": 200}}

Now parse this:
{user_text}
"""

