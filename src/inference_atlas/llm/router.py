"""Routing and fallback logic across multiple LLM adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from inference_atlas.llm.base import LLMAdapter
from inference_atlas.llm.gpt_5_2_adapter import GPT52Adapter
from inference_atlas.llm.opus_4_6_adapter import Opus46Adapter
from inference_atlas.llm.schema import WorkloadSpec, validate_workload_payload


@dataclass(frozen=True)
class RouterConfig:
    """Router configuration with primary/fallback provider names."""

    primary_provider: str = "gpt_5_2"
    fallback_provider: Optional[str] = "opus_4_6"


def build_default_adapters() -> dict[str, LLMAdapter]:
    """Return default adapter registry."""
    gpt = GPT52Adapter()
    opus = Opus46Adapter()
    return {
        gpt.provider_name: gpt,
        opus.provider_name: opus,
    }


class LLMRouter:
    """Provider-agnostic router with deterministic validation and fallback."""

    def __init__(
        self,
        adapters: Optional[Mapping[str, LLMAdapter]] = None,
        config: Optional[RouterConfig] = None,
    ) -> None:
        self.config = config or RouterConfig()
        self.adapters = dict(adapters) if adapters is not None else build_default_adapters()

    def _provider_order(self) -> list[str]:
        names: list[str] = [self.config.primary_provider]
        if (
            self.config.fallback_provider
            and self.config.fallback_provider != self.config.primary_provider
        ):
            names.append(self.config.fallback_provider)
        return names

    def parse_workload(self, user_text: str) -> WorkloadSpec:
        """Parse a free-form prompt into validated workload fields.

        Tries providers in order and returns first schema-valid result.
        """
        workload, _, _ = self.parse_workload_with_meta(user_text)
        return workload

    def parse_workload_with_meta(
        self,
        user_text: str,
    ) -> tuple[WorkloadSpec, str, dict[str, object]]:
        """Parse workload and return provider + raw payload metadata."""
        errors: list[str] = []
        last_exception: Exception | None = None
        for provider_name in self._provider_order():
            adapter = self.adapters.get(provider_name)
            if adapter is None:
                errors.append(f"{provider_name}: adapter missing")
                continue
            try:
                raw_payload = adapter.parse_workload(user_text)
                workload = validate_workload_payload(raw_payload)
                if not isinstance(raw_payload, dict):
                    raw_payload = dict(raw_payload)
                return workload, provider_name, raw_payload
            except Exception as exc:  # noqa: BLE001 - router must catch adapter errors
                last_exception = exc
                errors.append(f"{provider_name} [{type(exc).__name__}]: {exc}")

        raise RuntimeError(
            "All LLM providers failed to parse workload. Details: " + " | ".join(errors)
        ) from last_exception

    def explain(self, recommendation_summary: str, workload: WorkloadSpec) -> str:
        """Generate explanation via primary provider with fallback."""
        errors: list[str] = []
        last_exception: Exception | None = None
        for provider_name in self._provider_order():
            adapter = self.adapters.get(provider_name)
            if adapter is None:
                errors.append(f"{provider_name}: adapter missing")
                continue
            try:
                explanation = adapter.explain(recommendation_summary, workload)
                cleaned = explanation.strip()
                if cleaned:
                    return cleaned
                raise ValueError("empty explanation")
            except Exception as exc:  # noqa: BLE001 - router must catch adapter errors
                last_exception = exc
                errors.append(f"{provider_name} [{type(exc).__name__}]: {exc}")

        raise RuntimeError(
            "All LLM providers failed to generate explanation. Details: " + " | ".join(errors)
        ) from last_exception
