from __future__ import annotations

from inference_atlas.data_loader import CatalogV2Row, canonicalize_workload_token
from inference_atlas.invoice_analyzer import analyze_invoice_csv, canonical_workload_from_invoice


def test_canonical_workload_from_invoice_aliases() -> None:
    assert canonical_workload_from_invoice("tts") == "text_to_speech"
    assert canonical_workload_from_invoice("embedding") == "embeddings"
    assert canonical_workload_from_invoice("STT") == "speech_to_text"
    assert canonical_workload_from_invoice("transcription") == "speech_to_text"


def test_invoice_canonicalization_uses_data_loader_aliases() -> None:
    assert canonical_workload_from_invoice("image_gen") == canonicalize_workload_token("image_gen")
    assert canonical_workload_from_invoice("rerank") == canonicalize_workload_token("rerank")


def test_analyze_invoice_csv_finds_savings() -> None:
    csv_bytes = (
        "provider,workload_type,usage_qty,usage_unit,amount_usd\n"
        "expensive,llm,1000000,token,10.0\n"
    ).encode("utf-8")
    rows = [
        CatalogV2Row(
            provider="cheap",
            workload_type="llm",
            sku_key="cheap_llm",
            sku_name="cheap llm",
            model_key="llama",
            billing_mode="per_token",
            unit_price_usd=0.000005,
            unit_name="token",
            region="global",
            source_url="",
            source_date="2026-01-01",
            last_verified_at="2026-01-01",
            confidence="high",
            source_kind="provider_csv",
        )
    ]

    suggestions, summary = analyze_invoice_csv(csv_bytes, rows)
    assert len(suggestions) == 1
    assert suggestions[0]["best_provider"] == "cheap"
    assert summary["total_spend_usd"] == 10.0
    assert summary["total_estimated_savings_usd"] > 0


def test_analyze_invoice_csv_missing_columns() -> None:
    bad_csv = "provider,usage_qty,usage_unit,amount_usd\nx,1,token,2\n".encode("utf-8")
    try:
        analyze_invoice_csv(bad_csv, [])
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "Invoice CSV missing required columns" in str(exc)
