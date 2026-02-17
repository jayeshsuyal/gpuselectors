"""Invoice comparison helpers for catalog-backed savings analysis."""

from __future__ import annotations

import csv
import io
from typing import Any, Iterable

from inference_atlas.data_loader import CatalogV2Row


def canonical_workload_from_invoice(raw: str) -> str:
    """Normalize invoice workload aliases into catalog workload tokens."""
    token = raw.strip().lower()
    aliases = {
        "llm": "llm",
        "speech_to_text": "speech_to_text",
        "transcription": "speech_to_text",
        "stt": "speech_to_text",
        "text_to_speech": "text_to_speech",
        "tts": "text_to_speech",
        "embeddings": "embeddings",
        "embedding": "embeddings",
        "rerank": "embeddings",
        "image_generation": "image_generation",
        "image_gen": "image_generation",
        "vision": "vision",
        "video_generation": "video_generation",
        "moderation": "moderation",
    }
    return aliases.get(token, token)


def analyze_invoice_csv(
    csv_bytes: bytes,
    catalog_rows: Iterable[CatalogV2Row],
) -> tuple[list[dict[str, object]], dict[str, float]]:
    """Return per-line savings suggestions and a spend/savings summary."""
    text = csv_bytes.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    required = {"provider", "workload_type", "usage_qty", "usage_unit", "amount_usd"}
    if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
        missing = sorted(required - set(reader.fieldnames or []))
        raise ValueError("Invoice CSV missing required columns: " + ", ".join(missing))

    rows = list(catalog_rows)
    suggestions: list[dict[str, object]] = []
    total_spend = 0.0
    total_savings = 0.0

    for idx, row in enumerate(reader, start=2):
        try:
            provider = str(row["provider"]).strip()
            workload = canonical_workload_from_invoice(str(row["workload_type"]))
            usage_qty = float(str(row["usage_qty"]).strip())
            usage_unit = str(row["usage_unit"]).strip()
            amount_usd = float(str(row["amount_usd"]).strip())
        except (KeyError, ValueError):
            continue

        if usage_qty <= 0 or amount_usd <= 0:
            continue

        effective_unit_price = amount_usd / usage_qty
        total_spend += amount_usd
        pool = [
            candidate
            for candidate in rows
            if candidate.workload_type == workload and candidate.unit_name == usage_unit
        ]
        if not pool:
            continue
        best = min(pool, key=lambda candidate: candidate.unit_price_usd)
        savings_per_unit = max(0.0, effective_unit_price - best.unit_price_usd)
        savings_usd = savings_per_unit * usage_qty
        if savings_usd <= 0:
            continue

        total_savings += savings_usd
        suggestions.append(
            {
                "invoice_line": idx,
                "current_provider": provider,
                "workload_type": workload,
                "usage_qty": usage_qty,
                "usage_unit": usage_unit,
                "amount_usd": round(amount_usd, 2),
                "effective_unit_price": round(effective_unit_price, 8),
                "best_provider": best.provider,
                "best_offering": best.sku_name,
                "best_unit_price": round(best.unit_price_usd, 8),
                "estimated_savings_usd": round(savings_usd, 2),
                "savings_pct": round((savings_usd / amount_usd) * 100.0, 2),
                "source_kind": best.source_kind,
            }
        )

    suggestions.sort(key=lambda value: float(value["estimated_savings_usd"]), reverse=True)
    summary: dict[str, float] = {
        "invoice_lines_analyzed": float(len(suggestions)),
        "total_spend_usd": round(total_spend, 2),
        "total_estimated_savings_usd": round(total_savings, 2),
    }
    return suggestions, summary

