#!/usr/bin/env python3
"""Merge incoming '* copy.csv' files into canonical provider CSV files.

This script:
1) reads data/providers_csv/incoming/<provider> copy.csv
2) normalizes workload_type, billing_type, unit_name
3) drops invalid rows (missing required fields or non-positive price)
4) deduplicates by (provider, sku_key, unit_name, region)
5) backs up current canonical file into data/providers_csv/archive/
6) writes canonical file to data/providers_csv/<provider>.csv
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROVIDERS_DIR = ROOT / "data" / "providers_csv"
INCOMING_DIR = PROVIDERS_DIR / "incoming"
ARCHIVE_DIR = PROVIDERS_DIR / "archive"

CANONICAL_COLUMNS = [
    "workload_type",
    "provider",
    "billing_type",
    "sku_key",
    "sku_name",
    "model_key",
    "unit_price_usd",
    "unit_name",
    "throughput_value",
    "throughput_unit",
    "memory_gb",
    "latency_p50_ms",
    "latency_p95_ms",
    "region",
    "notes",
    "source_url",
    "source_date",
    "confidence",
]

REQUIRED_VALUE_COLUMNS = {
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

WORKLOAD_MAP = {
    "llm": "llm",
    "stt": "speech_to_text",
    "transcription": "speech_to_text",
    "speech_to_text": "speech_to_text",
    "tts": "text_to_speech",
    "text_to_speech": "text_to_speech",
    "voice_agent": "speech_to_text",
    "audio_processing": "speech_to_text",
    "dubbing": "text_to_speech",
    "music_gen": "text_to_speech",
    "sfx_gen": "text_to_speech",
    "embeddings": "embeddings",
    "embedding": "embeddings",
    "rerank": "embeddings",
    "reranking": "embeddings",
    "image_gen": "image_generation",
    "image_generation": "image_generation",
    "vision": "vision",
    "video_gen": "video_generation",
    "video_generation": "video_generation",
    "moderation": "moderation",
}

BILLING_MAP = {
    "pay_per_token": "per_token",
    "pay_per_minute": "per_minute",
    "pay_per_image": "per_image",
    "pay_per_characters": "per_character",
    "pay_per_generation": "per_unit",
    "pay_per_hour": "per_hour",
    "pay_per_video": "per_unit",
    "pay_per_step": "per_unit",
}

UNIT_MAP = {
    "audio_minute": "audio_min",
    "per_generation": "generation",
    "per_video": "video",
}


def _normalize_workload(value: str) -> str:
    return WORKLOAD_MAP.get(value.strip().lower(), value.strip().lower())


def _normalize_billing(value: str) -> str:
    token = value.strip().lower()
    return BILLING_MAP.get(token, token)


def _normalize_unit(value: str) -> str:
    token = value.strip().lower()
    return UNIT_MAP.get(token, token)


def _iter_copy_files() -> list[Path]:
    if not INCOMING_DIR.exists():
        return []
    return sorted(INCOMING_DIR.glob("* copy.csv"))


def _target_provider(copy_file: Path) -> str:
    stem = copy_file.stem
    if not stem.endswith(" copy"):
        raise ValueError(f"Unexpected incoming filename format: {copy_file.name}")
    return stem[: -len(" copy")]


def _is_valid_row(row: dict[str, str]) -> bool:
    for key in REQUIRED_VALUE_COLUMNS:
        if not (row.get(key) or "").strip():
            return False
    try:
        price = float((row.get("unit_price_usd") or "").strip())
    except ValueError:
        return False
    return price > 0


def _normalize_row(row: dict[str, str], provider: str) -> dict[str, str]:
    out = {col: (row.get(col) or "").strip() for col in CANONICAL_COLUMNS}
    out["provider"] = provider
    out["workload_type"] = _normalize_workload(out["workload_type"])
    out["billing_type"] = _normalize_billing(out["billing_type"])
    out["unit_name"] = _normalize_unit(out["unit_name"])
    return out


def _dedupe(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str, str]] = set()
    out: list[dict[str, str]] = []
    for row in rows:
        key = (
            row["provider"],
            row["sku_key"],
            row["unit_name"],
            row["region"],
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _backup_current(path: Path) -> None:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        return
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup = ARCHIVE_DIR / f"{path.stem}.{ts}.csv"
    backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CANONICAL_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    copy_files = _iter_copy_files()
    if not copy_files:
        print("No '* copy.csv' files found in data/providers_csv/incoming")
        return 0

    for copy_file in copy_files:
        provider = _target_provider(copy_file)
        target = PROVIDERS_DIR / f"{provider}.csv"

        with copy_file.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            raw_rows = [row for row in reader if row]

        normalized = [_normalize_row(row, provider) for row in raw_rows]
        filtered = [row for row in normalized if _is_valid_row(row)]
        deduped = _dedupe(filtered)

        _backup_current(target)
        _write_rows(target, deduped)
        print(
            f"{provider}: incoming={len(raw_rows)} valid={len(filtered)} "
            f"deduped={len(deduped)} -> {target}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
