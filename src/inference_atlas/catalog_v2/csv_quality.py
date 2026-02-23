"""Quality checks for per-provider CSV catalogs."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


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

CANONICAL_WORKLOADS = {
    "llm",
    "speech_to_text",
    "text_to_speech",
    "embeddings",
    "image_generation",
    "vision",
    "video_generation",
    "moderation",
}

WORKLOAD_ALIASES = {
    "transcription": "speech_to_text",
    "stt": "speech_to_text",
    "tts": "text_to_speech",
    "embedding": "embeddings",
    "rerank": "embeddings",
    "image_gen": "image_generation",
}


@dataclass(frozen=True)
class ProviderCsvAudit:
    provider_id: str
    file_path: str
    row_count: int
    bad_price_rows: int
    provider_mismatch_rows: int
    duplicate_key_rows: int
    unknown_workload_rows: int
    missing_required_cells: int
    missing_columns: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return (
            not self.missing_columns
            and self.bad_price_rows == 0
            and self.provider_mismatch_rows == 0
            and self.duplicate_key_rows == 0
            and self.unknown_workload_rows == 0
            and self.missing_required_cells == 0
        )


def _canonical_workload(raw: str) -> str:
    token = raw.strip().lower()
    if token in WORKLOAD_ALIASES:
        return WORKLOAD_ALIASES[token]
    return token


def audit_provider_csv(path: Path, provider_id: str) -> ProviderCsvAudit:
    """Run strict quality checks for one provider CSV file."""
    if not path.exists():
        return ProviderCsvAudit(
            provider_id=provider_id,
            file_path=str(path),
            row_count=0,
            bad_price_rows=0,
            provider_mismatch_rows=0,
            duplicate_key_rows=0,
            unknown_workload_rows=0,
            missing_required_cells=0,
            missing_columns=("file_missing",),
        )

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing_columns = tuple(sorted(REQUIRED_COLUMNS - set(fieldnames)))

        row_count = 0
        bad_price_rows = 0
        provider_mismatch_rows = 0
        duplicate_key_rows = 0
        unknown_workload_rows = 0
        missing_required_cells = 0
        seen_keys: set[tuple[str, str, str, str]] = set()

        for row in reader:
            row_count += 1
            if not row:
                continue

            for key in REQUIRED_COLUMNS:
                if not (row.get(key) or "").strip():
                    missing_required_cells += 1
                    break

            provider_value = (row.get("provider") or "").strip()
            if provider_value != provider_id:
                provider_mismatch_rows += 1

            workload_value = _canonical_workload((row.get("workload_type") or ""))
            if workload_value not in CANONICAL_WORKLOADS:
                unknown_workload_rows += 1

            price_raw = (row.get("unit_price_usd") or "").strip()
            try:
                price = float(price_raw)
            except ValueError:
                bad_price_rows += 1
            else:
                if price <= 0:
                    bad_price_rows += 1

            key = (
                provider_id,
                (row.get("sku_key") or "").strip(),
                (row.get("unit_name") or "").strip(),
                (row.get("region") or "").strip(),
            )
            if key in seen_keys:
                duplicate_key_rows += 1
            else:
                seen_keys.add(key)

    return ProviderCsvAudit(
        provider_id=provider_id,
        file_path=str(path),
        row_count=row_count,
        bad_price_rows=bad_price_rows,
        provider_mismatch_rows=provider_mismatch_rows,
        duplicate_key_rows=duplicate_key_rows,
        unknown_workload_rows=unknown_workload_rows,
        missing_required_cells=missing_required_cells,
        missing_columns=missing_columns,
    )
