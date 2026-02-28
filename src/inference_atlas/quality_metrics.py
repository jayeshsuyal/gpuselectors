"""Quality intelligence layer for model quality metrics and scoring.

v2.0 foundations:
- quality_metrics schema-backed data model
- model-key mapping from catalog keys to quality rows
- normalized 0-100 quality scoring
- confidence-aware quality score handling
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from inference_atlas.data_loader import CatalogV2Row, get_catalog_v2_rows


Confidence = Literal["official", "high", "medium", "low", "estimated"]

CONFIDENCE_WEIGHT: dict[Confidence, float] = {
    "official": 1.00,
    "high": 0.92,
    "medium": 0.80,
    "low": 0.65,
    "estimated": 0.50,
}

QUALITY_METRICS_PATH = Path(__file__).resolve().parents[2] / "data" / "quality" / "quality_metrics.json"


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.strip().lower())


def normalize_quality_score(raw: float, scale_min: float, scale_max: float) -> float:
    """Normalize a raw quality metric into 0-100 (clamped)."""
    if scale_max <= scale_min:
        raise ValueError("scale_max must be greater than scale_min")
    clamped = min(max(raw, scale_min), scale_max)
    normalized = ((clamped - scale_min) / (scale_max - scale_min)) * 100.0
    return max(0.0, min(100.0, normalized))


def confidence_adjusted_quality_score(
    normalized_score: float,
    confidence: Confidence,
    neutral_anchor: float = 50.0,
) -> float:
    """Shrink uncertain scores toward neutral anchor."""
    weight = CONFIDENCE_WEIGHT[confidence]
    adjusted = neutral_anchor + (normalized_score - neutral_anchor) * weight
    return max(0.0, min(100.0, adjusted))


class QualityMetricRow(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_id: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    workload_types: list[str] = Field(default_factory=list)
    provider: Optional[str] = None
    quality_raw: float
    scale_min: float
    scale_max: float
    quality_source: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    confidence: Confidence = "medium"
    last_verified_at: str = Field(min_length=10, max_length=10)

    @model_validator(mode="after")
    def _validate_scale(self) -> "QualityMetricRow":
        if self.scale_max <= self.scale_min:
            raise ValueError("scale_max must be greater than scale_min")
        datetime.strptime(self.last_verified_at, "%Y-%m-%d")
        return self


class QualityMetricsPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    generated_at_utc: str
    rows: list[QualityMetricRow]


@dataclass(frozen=True)
class QualityScore:
    model_id: str
    confidence: Confidence
    confidence_weight: float
    normalized_score: float
    adjusted_score: float
    matched_by: Literal["exact", "alias"]


@dataclass(frozen=True)
class QualityIndex:
    exact: dict[str, QualityMetricRow]
    alias: dict[str, QualityMetricRow]


def load_quality_metrics(path: Path | None = None) -> QualityMetricsPayload:
    source = path or QUALITY_METRICS_PATH
    payload = json.loads(source.read_text(encoding="utf-8"))
    return QualityMetricsPayload.model_validate(payload)


def build_quality_index(metrics: list[QualityMetricRow]) -> QualityIndex:
    exact: dict[str, QualityMetricRow] = {}
    alias: dict[str, QualityMetricRow] = {}
    for row in metrics:
        key = _normalize_key(row.model_id)
        exact[key] = row
        generated_aliases = {row.model_id, *(row.aliases or [])}
        for token in generated_aliases:
            alias[_normalize_key(token)] = row
    return QualityIndex(exact=exact, alias=alias)


def resolve_quality_score(model_key: str, index: QualityIndex) -> QualityScore | None:
    key = _normalize_key(model_key)
    if key in index.exact:
        row = index.exact[key]
        matched_by: Literal["exact", "alias"] = "exact"
    elif key in index.alias:
        row = index.alias[key]
        matched_by = "alias"
    else:
        return None
    normalized = normalize_quality_score(row.quality_raw, row.scale_min, row.scale_max)
    adjusted = confidence_adjusted_quality_score(normalized, row.confidence)
    return QualityScore(
        model_id=row.model_id,
        confidence=row.confidence,
        confidence_weight=CONFIDENCE_WEIGHT[row.confidence],
        normalized_score=round(normalized, 3),
        adjusted_score=round(adjusted, 3),
        matched_by=matched_by,
    )


def score_catalog_quality(
    rows: list[CatalogV2Row],
    index: QualityIndex,
) -> list[tuple[CatalogV2Row, QualityScore | None]]:
    return [(row, resolve_quality_score(row.model_key, index)) for row in rows]


def load_quality_index(path: Path | None = None) -> QualityIndex:
    payload = load_quality_metrics(path)
    return build_quality_index(payload.rows)


def get_quality_scores_for_workload(
    workload_type: str | None = None,
    path: Path | None = None,
) -> list[tuple[CatalogV2Row, QualityScore | None]]:
    rows = get_catalog_v2_rows(workload_type)
    index = load_quality_index(path)
    return score_catalog_quality(rows, index)
