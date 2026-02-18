"""GPT adapter using the OpenAI API."""

from __future__ import annotations

import json
import os
from typing import Any, Optional

try:
    from openai import OpenAI, RateLimitError
except ImportError:  # pragma: no cover - handled at runtime
    OpenAI = None  # type: ignore[assignment]
    RateLimitError = Exception  # type: ignore[assignment]

from inference_atlas.llm.base import LLMAdapter
from inference_atlas.llm.prompting import build_workload_parser_prompt
from inference_atlas.llm.schema import WorkloadSpec

PRIMARY_MODEL = "gpt-4-turbo"
FALLBACK_MODEL = "gpt-4"


class GPT52Adapter(LLMAdapter):
    """OpenAI-backed adapter for parsing workloads and explaining recommendations."""

    provider_name = "gpt_5_2"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_sec: int = 30,
        client: Any = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model or os.getenv("OPENAI_MODEL", PRIMARY_MODEL)
        self.timeout_sec = timeout_sec
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        if self.timeout_sec <= 0:
            raise ValueError("timeout_sec must be > 0.")
        self._ensure_api_key()
        if client is not None:
            self.client = client
        else:
            if OpenAI is None:
                raise RuntimeError(
                    "openai package is not installed. Install with: pip install openai>=1.0.0"
                )
            if self.base_url:
                self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            else:
                self.client = OpenAI(api_key=self.api_key)
        self._model_candidates = [self.model]
        if self.model != FALLBACK_MODEL:
            self._model_candidates.append(FALLBACK_MODEL)

    def _ensure_api_key(self) -> None:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")

    def _responses_text(self, system_prompt: str, user_prompt: str, model: str) -> str:
        response = self.client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
            ],
            timeout=self.timeout_sec,
        )
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()
        raise RuntimeError("OpenAI response did not contain text output.")

    def _generate_text(self, system_prompt: str, user_prompt: str) -> str:
        last_error: Optional[Exception] = None
        for model_name in self._model_candidates:
            try:
                return self._responses_text(system_prompt, user_prompt, model_name)
            except RateLimitError as exc:
                raise RuntimeError("OpenAI rate limit exceeded. Please retry shortly.") from exc
            except Exception as exc:  # noqa: BLE001 - fallback model probing
                last_error = exc
                continue
        raise RuntimeError(f"OpenAI request failed: {last_error}")

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
        raise RuntimeError("OpenAI parse response was not valid JSON object.")

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
