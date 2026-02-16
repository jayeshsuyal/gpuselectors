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

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
