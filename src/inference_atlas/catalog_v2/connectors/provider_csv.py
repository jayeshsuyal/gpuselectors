"""Per-provider CSV connector for catalog_v2."""

from __future__ import annotations

import csv
from pathlib import Path

from inference_atlas.catalog_v2.schema import CanonicalPricingRow

PROVIDERS_CSV_DIR = Path(__file__).resolve().parents[4] / "data" / "providers_csv"
REQUIRED_COLUMNS = {
    "workload_type",
    "provider",
    "billing_type",
    "sku_key",
    "sku_name",
    "model_key",
    "unit_price_usd",
    "unit_name",
    "region",
    "source_url",
    "source_date",
    "confidence",
}


def list_csv_providers() -> list[str]:
    """Return provider IDs from per-provider CSV files."""
    if not PROVIDERS_CSV_DIR.exists():
        return []
    return sorted(path.stem for path in PROVIDERS_CSV_DIR.glob("*.csv"))


def fetch_rows_for_provider(provider_id: str) -> list[CanonicalPricingRow]:
    """Load canonical rows for a provider from data/providers_csv/<provider>.csv."""
    path = PROVIDERS_CSV_DIR / f"{provider_id}.csv"
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []
        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} missing required columns: {', '.join(sorted(missing))}")

        out: list[CanonicalPricingRow] = []
        for idx, row in enumerate(reader, start=2):
            provider = (row.get("provider") or "").strip()
            if provider != provider_id:
                raise ValueError(
                    f"{path}:{idx} provider='{provider}' does not match filename '{provider_id}'"
                )
            try:
                unit_price_usd = float((row.get("unit_price_usd") or "").strip())
            except ValueError:
                continue
            if unit_price_usd <= 0:
                continue
            throughput_raw = (row.get("throughput_value") or "").strip()
            throughput_value: float | None = None
            if throughput_raw:
                try:
                    throughput_value = float(throughput_raw)
                except ValueError:
                    throughput_value = None
                else:
                    if throughput_value <= 0:
                        throughput_value = None

            out.append(
                CanonicalPricingRow(
                    provider=provider_id,
                    workload_type=(row.get("workload_type") or "").strip(),
                    sku_key=(row.get("sku_key") or "").strip(),
                    sku_name=(row.get("sku_name") or "").strip(),
                    model_key=(row.get("model_key") or "").strip(),
                    billing_mode=(row.get("billing_type") or "").strip(),
                    unit_price_usd=unit_price_usd,
                    unit_name=(row.get("unit_name") or "").strip(),
                    region=(row.get("region") or "").strip(),
                    source_url=(row.get("source_url") or "").strip(),
                    source_date=(row.get("source_date") or "").strip(),
                    last_verified_at=(
                        (row.get("last_verified_at") or "").strip()
                        or (row.get("source_date") or "").strip()
                        or None
                    ),
                    confidence=(row.get("confidence") or "").strip() or "estimated",
                    source_kind="provider_csv",
                    throughput_value=throughput_value,
                    throughput_unit=((row.get("throughput_unit") or "").strip() or None),
                )
            )
    return out
