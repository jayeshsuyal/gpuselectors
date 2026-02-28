from __future__ import annotations

from inference_atlas.data_loader import get_catalog_v2_rows
from inference_atlas.quality_metrics import (
    CONFIDENCE_WEIGHT,
    build_quality_index,
    confidence_adjusted_quality_score,
    load_quality_metrics,
    normalize_quality_score,
    resolve_quality_score,
    score_catalog_quality,
)


def test_load_quality_metrics_payload() -> None:
    payload = load_quality_metrics()
    assert payload.schema_version == "v2.0"
    assert len(payload.rows) >= 5


def test_build_quality_index_and_exact_resolution() -> None:
    payload = load_quality_metrics()
    index = build_quality_index(payload.rows)
    score = resolve_quality_score("gpt-4o", index)
    assert score is not None
    assert score.matched_by == "exact"
    assert 0 <= score.normalized_score <= 100
    assert 0 <= score.adjusted_score <= 100


def test_alias_resolution_works() -> None:
    payload = load_quality_metrics()
    index = build_quality_index(payload.rows)
    score = resolve_quality_score("openai/whisper-large-v3", index)
    assert score is not None
    assert score.matched_by == "alias"
    assert score.model_id == "whisper-large"


def test_confidence_adjustment_moves_toward_neutral() -> None:
    normalized = 90.0
    adjusted_official = confidence_adjusted_quality_score(normalized, "official")
    adjusted_low = confidence_adjusted_quality_score(normalized, "low")
    assert adjusted_official > adjusted_low
    assert adjusted_official == normalized
    assert adjusted_low < normalized
    assert CONFIDENCE_WEIGHT["low"] < CONFIDENCE_WEIGHT["official"]


def test_normalization_clamps_to_bounds() -> None:
    assert normalize_quality_score(9999, 1200, 1400) == 100.0
    assert normalize_quality_score(0, 1200, 1400) == 0.0


def test_score_catalog_quality_maps_some_rows() -> None:
    rows = get_catalog_v2_rows("llm")
    payload = load_quality_metrics()
    index = build_quality_index(payload.rows)
    scored = score_catalog_quality(rows, index)
    assert scored
    matched = [item for item in scored if item[1] is not None]
    assert matched, "Expected at least some catalog model keys to map to quality rows"
