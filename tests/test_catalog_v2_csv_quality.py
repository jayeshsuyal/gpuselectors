from __future__ import annotations

from pathlib import Path

from inference_atlas.catalog_v2.csv_quality import audit_provider_csv


def test_audit_provider_csv_ok_with_alias_workload(tmp_path: Path) -> None:
    csv_path = tmp_path / "deepgram.csv"
    csv_path.write_text(
        "workload_type,provider,billing_type,sku_key,sku_name,model_key,unit_price_usd,unit_name,region,source_url,source_date,confidence\n"
        "transcription,deepgram,per_minute,nova_3,Nova 3,nova_3,0.007,audio_min,global,https://example.com,2026-02-18,high\n",
        encoding="utf-8",
    )
    audit = audit_provider_csv(csv_path, "deepgram")
    assert audit.ok is True
    assert audit.row_count == 1
    assert audit.unknown_workload_rows == 0


def test_audit_provider_csv_flags_bad_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "openai.csv"
    csv_path.write_text(
        "workload_type,provider,billing_type,sku_key,sku_name,model_key,unit_price_usd,unit_name,region,source_url,source_date,confidence\n"
        "llm,openai,per_token,gpt4o,GPT-4o,gpt-4o,0,1m_tokens,global,https://example.com,2026-02-18,official\n"
        "unknown_workload,wrong_provider,per_token,gpt4o,GPT-4o,gpt-4o,-1,1m_tokens,global,https://example.com,2026-02-18,official\n"
        "llm,openai,per_token,gpt4o,GPT-4o,gpt-4o,1,1m_tokens,global,https://example.com,2026-02-18,official\n",
        encoding="utf-8",
    )
    audit = audit_provider_csv(csv_path, "openai")
    assert audit.ok is False
    assert audit.bad_price_rows >= 2
    assert audit.provider_mismatch_rows >= 1
    assert audit.unknown_workload_rows >= 1
    assert audit.duplicate_key_rows >= 1
