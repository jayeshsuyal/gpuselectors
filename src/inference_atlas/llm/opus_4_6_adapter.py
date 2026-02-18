"""Opus adapter using the Anthropic API."""

from __future__ import annotations

import json
import os
from typing import Any, Optional

try:
    import anthropic
except ImportError:  # pragma: no cover - handled at runtime
    anthropic = None  # type: ignore[assignment]

from inference_atlas.llm.base import LLMAdapter
from inference_atlas.llm.prompting import build_workload_parser_prompt
from inference_atlas.llm.schema import WorkloadSpec

PRIMARY_MODEL = "claude-opus-4-6-20250514"
FALLBACK_MODEL = "claude-opus-4-20250514"


class Opus46Adapter(LLMAdapter):
    """Anthropic-backed adapter for parsing workloads and explanations."""

    provider_name = "opus_4_6"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_sec: int = 30,
        client: Any = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.model = model or os.getenv("ANTHROPIC_MODEL", PRIMARY_MODEL)
        self.timeout_sec = timeout_sec
        self.base_url = base_url or os.getenv("ANTHROPIC_BASE_URL")
        if self.timeout_sec <= 0:
            raise ValueError("timeout_sec must be > 0.")
        self._ensure_api_key()
        if client is not None:
            self.client = client
        else:
            if anthropic is None:
                raise RuntimeError(
                    "anthropic package is not installed. Install with: pip install anthropic>=0.40.0"
                )
            if self.base_url:
                self.client = anthropic.Anthropic(api_key=self.api_key, base_url=self.base_url)
            else:
                self.client = anthropic.Anthropic(api_key=self.api_key)
        self._model_candidates = [self.model]
        if self.model != FALLBACK_MODEL:
            self._model_candidates.append(FALLBACK_MODEL)

    def _ensure_api_key(self) -> None:
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

    def _messages_text(self, system_prompt: str, user_prompt: str, model: str) -> str:
        response = self.client.messages.create(
            model=model,
            system=system_prompt,
            max_tokens=600,
            messages=[{"role": "user", "content": user_prompt}],
            timeout=self.timeout_sec,
        )
        for item in getattr(response, "content", []):
            text = getattr(item, "text", None)
            if isinstance(text, str) and text.strip():
                return text.strip()
        raise RuntimeError("Anthropic response did not contain text output.")

    def _generate_text(self, system_prompt: str, user_prompt: str) -> str:
        last_error: Optional[Exception] = None
        for model_name in self._model_candidates:
            try:
                return self._messages_text(system_prompt, user_prompt, model_name)
            except Exception as exc:  # noqa: BLE001
                if anthropic is not None and isinstance(exc, anthropic.RateLimitError):
                    raise RuntimeError("Anthropic rate limit exceeded. Please retry shortly.") from exc
                last_error = exc
                continue
        raise RuntimeError(f"Anthropic request failed: {last_error}")

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any]:
        """Extract a JSON object from model output text."""
        candidate = text.strip()
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            snippet = candidate[start : end + 1]
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                return parsed
        raise RuntimeError("Opus parse response was not valid JSON object.")

    def parse_workload(self, user_text: str) -> dict[str, Any]:
        parser_prompt = build_workload_parser_prompt(user_text)
        text = self._generate_text("Return JSON only.", parser_prompt)
        try:
            return self._extract_json_object(text)
        except RuntimeError:
            retry_text = self._generate_text(
                "Return valid JSON only.",
                parser_prompt + "\n\nReturn valid JSON only.",
            )
            return self._extract_json_object(retry_text)

    def explain(self, recommendation_summary: str, workload: WorkloadSpec) -> str:
        explain_prompt = f"""You are an expert in LLM deployment cost optimization.

Given this workload:
- Tokens/day: {workload.tokens_per_day:,}
- Traffic pattern: {workload.pattern}
- Model: {workload.model_key}
- Latency requirement: {workload.latency_requirement_ms or "None"}

And this recommendation:
{recommendation_summary}

Explain in 2-3 sentences:
1. Why this option is cost-effective for this workload
2. Key trade-offs (utilization vs idle waste vs latency)
3. When to consider alternatives

Be technical but concise. Focus on actionable insights.
"""
        return self._generate_text(
            "Be concise and technical. Do not fabricate metrics.",
            explain_prompt,
        )
