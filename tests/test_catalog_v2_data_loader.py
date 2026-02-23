from __future__ import annotations

from inference_atlas.data_loader import get_catalog_v2_metadata, get_catalog_v2_rows


def test_get_catalog_v2_rows_loads_data() -> None:
    rows = get_catalog_v2_rows()
    assert rows
    assert any(row.provider == "openai" for row in rows)


def test_get_catalog_v2_rows_filters_by_workload() -> None:
    llm_rows = get_catalog_v2_rows("llm")
    assert llm_rows
    assert all(row.workload_type == "llm" for row in llm_rows)


def test_get_catalog_v2_rows_normalizes_workload_aliases() -> None:
    embeddings_rows = get_catalog_v2_rows("embeddings")
    tts_rows = get_catalog_v2_rows("text_to_speech")
    assert embeddings_rows
    assert tts_rows


def test_get_catalog_v2_metadata_has_core_fields() -> None:
    meta = get_catalog_v2_metadata()
    assert "generated_at_utc" in meta
    assert isinstance(meta.get("row_count"), int)
    assert isinstance(meta.get("providers_synced"), list)
    assert isinstance(meta.get("price_deltas_changed_rows"), int)
    assert isinstance(meta.get("price_deltas_matched_rows"), int)
