"""Canonical schema helpers for catalog v2 pricing rows."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CanonicalPricingRow:
    provider: str
    workload_type: str
    sku_key: str
    sku_name: str
    model_key: str
    billing_mode: str
    unit_price_usd: float
    unit_name: str
    region: str
    source_url: str
    source_date: str
    confidence: str
    source_kind: str
    last_verified_at: str | None = None
    throughput_value: float | None = None
    throughput_unit: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        if not payload.get("last_verified_at"):
            payload["last_verified_at"] = payload.get("source_date")
        if payload.get("throughput_value") is None:
            payload.pop("throughput_value", None)
        if payload.get("throughput_unit") is None:
            payload.pop("throughput_unit", None)
        return payload
