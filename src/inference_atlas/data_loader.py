"""Data loading utilities for pricing catalogs and model specifications.

This module now supports two data layers:
1) Legacy typed `PLATFORMS` in `data/platforms.py` (used by existing cost engine code)
2) CSV-backed unified pricing datasets (routed by workload type)
"""

from __future__ import annotations

import csv
from copy import deepcopy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Add project root to path to allow importing from data/
def _discover_project_root() -> Path:
    """Locate repo root in both editable and installed environments."""
    here = Path(__file__).resolve()
    cwd = Path.cwd().resolve()
    candidates: list[Path] = [
        here.parent.parent.parent,
        cwd,
        *list(cwd.parents),
    ]

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        data_dir = candidate / "data"
        if (data_dir / "catalog_v2" / "pricing_catalog.json").exists():
            return candidate
        if (data_dir / "models.json").exists() and (data_dir / "providers.json").exists():
            return candidate

    # Fallback to previous behavior.
    return here.parent.parent.parent


_project_root = _discover_project_root()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from inference_atlas.workload_types import WorkloadType

if TYPE_CHECKING:
    from data.platforms import Platform
    from typing import TypedDict

    class ModelRequirement(TypedDict):
        """Model memory and parameter specifications."""

        display_name: str
        recommended_memory_gb: int
        parameter_count: int

DATASET_FILES = (
    _project_root / "data" / "master_ai_pricing_dataset_16_providers.csv",
    _project_root / "data" / "ai_pricing_final_4_providers.csv",
)

MVP_CATALOG_DATA_FILES = {
    "providers": _project_root / "data" / "providers.json",
    "models": _project_root / "data" / "models.json",
    "capacity_table": _project_root / "data" / "capacity_table.json",
}

MVP_CATALOG_SCHEMA_FILES = {
    "providers": _project_root / "data" / "providers.schema.json",
    "models": _project_root / "data" / "models.schema.json",
    "capacity_table": _project_root / "data" / "capacity_table.schema.json",
}

HUGGINGFACE_CATALOG_FILE = _project_root / "data" / "huggingface_models.json"
HUGGINGFACE_SCHEMA_FILE = _project_root / "data" / "huggingface_models.schema.json"
CATALOG_V2_FILE = _project_root / "data" / "catalog_v2" / "pricing_catalog.json"

REQUIRED_COLUMNS = {
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
}

WORKLOAD_ALIASES = {
    "llm": WorkloadType.LLM,
    "transcription": WorkloadType.SPEECH_TO_TEXT,
    "speech_to_text": WorkloadType.SPEECH_TO_TEXT,
    "stt": WorkloadType.SPEECH_TO_TEXT,
    "tts": WorkloadType.TEXT_TO_SPEECH,
    "text_to_speech": WorkloadType.TEXT_TO_SPEECH,
    "embeddings": WorkloadType.EMBEDDINGS,
    "embedding": WorkloadType.EMBEDDINGS,
    "rerank": WorkloadType.EMBEDDINGS,
    "image_gen": WorkloadType.IMAGE_GENERATION,
    "image_generation": WorkloadType.IMAGE_GENERATION,
    "vision": WorkloadType.VISION,
}

PROVIDER_KEY_ALIASES = {
    "together_ai": "together",
}


@dataclass(frozen=True)
class PricingRecord:
    """One normalized pricing row from CSV datasets."""

    workload_type: WorkloadType
    provider: str
    billing_type: str
    sku_key: str
    sku_name: str
    model_key: str
    unit_price_usd: float
    unit_name: str
    throughput_value: float | None
    throughput_unit: str | None
    memory_gb: int | None
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    region: str
    notes: str
    source_url: str
    source_date: str
    confidence: str


@dataclass(frozen=True)
class CatalogV2Row:
    """One canonical row from data/catalog_v2/pricing_catalog.json."""

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


_pricing_records_cache: list[PricingRecord] | None = None
_mvp_catalog_validation_summary_cache: dict[str, int] | None = None
_mvp_catalog_data_cache: dict[str, dict[str, Any]] | None = None
_huggingface_models_cache: list[dict[str, Any]] | None = None
_huggingface_catalog_meta_cache: dict[str, Any] | None = None
_catalog_v2_rows_cache: list[CatalogV2Row] | None = None
_catalog_v2_meta_cache: dict[str, Any] | None = None


def _parse_optional_float(value: str, field_name: str, source: Path, row_num: int) -> float | None:
    if not value.strip():
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(
            f"{source}:{row_num} invalid float for '{field_name}': {value!r}"
        ) from exc
    if parsed < 0:
        raise ValueError(f"{source}:{row_num} field '{field_name}' cannot be negative")
    return parsed


def _parse_optional_int(value: str, field_name: str, source: Path, row_num: int) -> int | None:
    if not value.strip():
        return None
    try:
        parsed = int(float(value))
    except ValueError as exc:
        raise ValueError(
            f"{source}:{row_num} invalid int for '{field_name}': {value!r}"
        ) from exc
    if parsed < 0:
        raise ValueError(f"{source}:{row_num} field '{field_name}' cannot be negative")
    return parsed


def _normalize_workload_type(raw: str, source: Path, row_num: int) -> WorkloadType:
    key = raw.strip().lower()
    if key not in WORKLOAD_ALIASES:
        valid = ", ".join(sorted(WORKLOAD_ALIASES))
        raise ValueError(
            f"{source}:{row_num} unsupported workload_type '{raw}'. Valid values/aliases: {valid}"
        )
    return WORKLOAD_ALIASES[key]


def _canonical_workload_value(raw: str) -> str:
    """Normalize workload alias into canonical WorkloadType.value token."""
    return canonicalize_workload_token(raw)


def canonicalize_workload_token(raw: str) -> str:
    """Normalize workload aliases into canonical workload tokens."""
    key = raw.strip().lower()
    workload = WORKLOAD_ALIASES.get(key)
    if workload is None:
        return key
    return workload.value


def _load_pricing_file(path: Path) -> list[PricingRecord]:
    if not path.exists():
        raise FileNotFoundError(f"Pricing dataset not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        rows_iter = iter(reader)
        try:
            fieldnames = next(rows_iter)
        except StopIteration as exc:
            raise ValueError(f"{path} has no header row") from exc
        if not fieldnames:
            raise ValueError(f"{path} has no header row")
        missing = REQUIRED_COLUMNS - set(fieldnames)
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(f"{path} is missing required columns: {missing_str}")

        rows: list[PricingRecord] = []
        header_len = len(fieldnames)
        for row_num, values in enumerate(rows_iter, start=2):
            if len(values) == header_len - 1:
                # Current datasets frequently omit `latency_p95_ms`.
                # Insert an empty value at that position to restore alignment.
                latency_p95_idx = fieldnames.index("latency_p95_ms")
                values = [*values[:latency_p95_idx], "", *values[latency_p95_idx:]]
            if len(values) != header_len:
                raise ValueError(
                    f"{path}:{row_num} expected {header_len} columns, found {len(values)}"
                )
            row = dict(zip(fieldnames, values))
            for field in (
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
            ):
                if not (row.get(field) or "").strip():
                    raise ValueError(f"{path}:{row_num} missing required value in '{field}'")

            workload_type = _normalize_workload_type(row["workload_type"], path, row_num)
            unit_price_usd = _parse_optional_float(row["unit_price_usd"], "unit_price_usd", path, row_num)
            if unit_price_usd is None or unit_price_usd <= 0:
                raise ValueError(f"{path}:{row_num} field 'unit_price_usd' must be > 0")

            rows.append(
                PricingRecord(
                    workload_type=workload_type,
                    provider=row["provider"].strip(),
                    billing_type=row["billing_type"].strip(),
                    sku_key=row["sku_key"].strip(),
                    sku_name=row["sku_name"].strip(),
                    model_key=row["model_key"].strip(),
                    unit_price_usd=unit_price_usd,
                    unit_name=row["unit_name"].strip(),
                    throughput_value=_parse_optional_float(
                        row.get("throughput_value", ""),
                        "throughput_value",
                        path,
                        row_num,
                    ),
                    throughput_unit=(row.get("throughput_unit", "") or "").strip() or None,
                    memory_gb=_parse_optional_int(row.get("memory_gb", ""), "memory_gb", path, row_num),
                    latency_p50_ms=_parse_optional_float(
                        row.get("latency_p50_ms", ""),
                        "latency_p50_ms",
                        path,
                        row_num,
                    ),
                    latency_p95_ms=_parse_optional_float(
                        row.get("latency_p95_ms", ""),
                        "latency_p95_ms",
                        path,
                        row_num,
                    ),
                    region=row["region"].strip(),
                    notes=(row.get("notes", "") or "").strip(),
                    source_url=row["source_url"].strip(),
                    source_date=(row.get("source_date", "") or "").strip(),
                    confidence=(row.get("confidence", "") or "").strip(),
                )
            )
    return rows


def _load_all_pricing_records() -> list[PricingRecord]:
    global _pricing_records_cache
    if _pricing_records_cache is None:
        records: list[PricingRecord] = []
        for path in DATASET_FILES:
            records.extend(_load_pricing_file(path))
        _pricing_records_cache = records
    return list(_pricing_records_cache)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root for {path} must be an object")
    return payload


def _load_catalog_v2_rows() -> list[CatalogV2Row]:
    global _catalog_v2_rows_cache
    global _catalog_v2_meta_cache
    if _catalog_v2_rows_cache is not None:
        return list(_catalog_v2_rows_cache)

    data = _load_json(CATALOG_V2_FILE)
    rows = data.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError(f"{CATALOG_V2_FILE} rows must be a list")

    parsed: list[CatalogV2Row] = []
    for idx, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        try:
            parsed.append(
                CatalogV2Row(
                    provider=str(row["provider"]),
                    workload_type=_canonical_workload_value(str(row["workload_type"])),
                    sku_key=str(row["sku_key"]),
                    sku_name=str(row["sku_name"]),
                    model_key=str(row.get("model_key", "")),
                    billing_mode=str(row["billing_mode"]),
                    unit_price_usd=float(row["unit_price_usd"]),
                    unit_name=str(row["unit_name"]),
                    region=str(row["region"]),
                    source_url=str(row["source_url"]),
                    source_date=str(row.get("source_date", "")),
                    confidence=str(row.get("confidence", "")),
                    source_kind=str(row.get("source_kind", "")),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{CATALOG_V2_FILE}: invalid row at index {idx}: {exc}") from exc

    _catalog_v2_rows_cache = parsed
    _catalog_v2_meta_cache = {
        "generated_at_utc": data.get("generated_at_utc"),
        "row_count": int(data.get("row_count", len(parsed))),
        "providers_synced": list(data.get("providers_synced", [])),
        "connector_counts": dict(data.get("connector_counts", {})),
        "schema_version": data.get("schema_version"),
    }
    return list(_catalog_v2_rows_cache)


def validate_mvp_catalogs(force: bool = False) -> dict[str, int]:
    """Validate MVP JSON catalogs against their schemas and return entry counts."""
    global _mvp_catalog_validation_summary_cache
    global _mvp_catalog_data_cache

    if _mvp_catalog_validation_summary_cache is not None and not force:
        return dict(_mvp_catalog_validation_summary_cache)

    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise RuntimeError(
            "The 'jsonschema' package is required for MVP catalog validation. "
            "Install it with: pip install jsonschema"
        ) from exc

    summary: dict[str, int] = {}
    loaded_data: dict[str, dict[str, Any]] = {}

    for catalog_name, data_path in MVP_CATALOG_DATA_FILES.items():
        schema_path = MVP_CATALOG_SCHEMA_FILES[catalog_name]
        schema = _load_json(schema_path)
        data = _load_json(data_path)

        validator = Draft202012Validator(schema)
        errors = sorted(validator.iter_errors(data), key=lambda e: list(e.path))
        if errors:
            first = errors[0]
            location = ".".join(str(token) for token in first.path) or "<root>"
            raise ValueError(
                f"{data_path} failed schema validation at {location}: {first.message}"
            )

        if catalog_name == "providers":
            summary[data_path.name] = len(data.get("providers", []))
        elif catalog_name == "models":
            summary[data_path.name] = len(data.get("models", []))
        else:
            summary[data_path.name] = len(data.get("entries", []))

        loaded_data[catalog_name] = data

    _mvp_catalog_validation_summary_cache = summary
    _mvp_catalog_data_cache = loaded_data
    return dict(summary)


def get_mvp_catalog(catalog_name: str) -> dict[str, Any]:
    """Return one schema-validated MVP catalog JSON object."""
    validate_mvp_catalogs()
    if _mvp_catalog_data_cache is None:
        raise RuntimeError("MVP catalog cache is empty after validation")
    if catalog_name not in _mvp_catalog_data_cache:
        valid = ", ".join(sorted(_mvp_catalog_data_cache))
        raise KeyError(f"Unknown catalog '{catalog_name}'. Valid options: {valid}")
    return deepcopy(_mvp_catalog_data_cache[catalog_name])


def validate_huggingface_catalog(force: bool = False) -> int:
    """Validate local Hugging Face catalog JSON against schema."""
    global _huggingface_models_cache
    global _huggingface_catalog_meta_cache
    if _huggingface_models_cache is not None and not force:
        return len(_huggingface_models_cache)

    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise RuntimeError(
            "The 'jsonschema' package is required for Hugging Face catalog validation."
        ) from exc

    schema = _load_json(HUGGINGFACE_SCHEMA_FILE)
    data = _load_json(HUGGINGFACE_CATALOG_FILE)
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda e: list(e.path))
    if errors:
        first = errors[0]
        location = ".".join(str(token) for token in first.path) or "<root>"
        raise ValueError(
            f"{HUGGINGFACE_CATALOG_FILE} failed schema validation at {location}: "
            f"{first.message}"
        )

    models = data.get("models", [])
    if not isinstance(models, list):
        raise ValueError(f"{HUGGINGFACE_CATALOG_FILE} models must be a list")
    _huggingface_models_cache = models
    _huggingface_catalog_meta_cache = {
        "generated_at_utc": data.get("generated_at_utc"),
        "model_count": int(data.get("model_count", len(models))),
        "source": data.get("source"),
        "schema_version": data.get("schema_version"),
    }
    return len(_huggingface_models_cache)


def get_huggingface_models(
    min_downloads: int = 0,
    include_gated: bool = False,
) -> list[dict[str, Any]]:
    """Get schema-validated Hugging Face models for open-source lane."""
    if min_downloads < 0:
        raise ValueError("min_downloads must be >= 0")
    validate_huggingface_catalog()
    if _huggingface_models_cache is None:
        raise RuntimeError("Hugging Face model cache is empty after validation")
    models = [
        row
        for row in _huggingface_models_cache
        if int(row.get("downloads", 0)) >= min_downloads
        and (include_gated or not bool(row.get("gated", False)))
    ]
    models.sort(key=lambda row: int(row.get("downloads", 0)), reverse=True)
    return deepcopy(models)


def get_huggingface_catalog_metadata() -> dict[str, Any]:
    """Return metadata for the local Hugging Face catalog snapshot."""
    validate_huggingface_catalog()
    if _huggingface_catalog_meta_cache is None:
        raise RuntimeError("Hugging Face catalog metadata cache is empty after validation")
    return deepcopy(_huggingface_catalog_meta_cache)


def refresh_huggingface_catalog_cache() -> dict[str, Any]:
    """Force-refresh Hugging Face catalog caches after sync."""
    validate_huggingface_catalog(force=True)
    return get_huggingface_catalog_metadata()


def get_pricing_records(workload_type: WorkloadType | str | None = None) -> list[PricingRecord]:
    """Load normalized CSV pricing records, optionally filtered by workload type."""
    records = _load_all_pricing_records()
    if workload_type is None:
        return records

    if isinstance(workload_type, str):
        normalized = _normalize_workload_type(workload_type, DATASET_FILES[0], 0)
    else:
        normalized = workload_type

    return [row for row in records if row.workload_type == normalized]


def get_catalog_v2_rows(workload_type: WorkloadType | str | None = None) -> list[CatalogV2Row]:
    """Load unified catalog_v2 rows, optionally filtered by workload."""
    rows = _load_catalog_v2_rows()
    if workload_type is None:
        return rows
    token = workload_type.value if isinstance(workload_type, WorkloadType) else workload_type.strip().lower()
    return [row for row in rows if row.workload_type == token]


def get_catalog_v2_metadata() -> dict[str, Any]:
    """Return metadata for the unified catalog_v2 snapshot."""
    _load_catalog_v2_rows()
    if _catalog_v2_meta_cache is None:
        raise RuntimeError("catalog_v2 metadata cache is empty after loading rows")
    return deepcopy(_catalog_v2_meta_cache)


def refresh_catalog_v2_cache() -> dict[str, Any]:
    """Force-refresh catalog_v2 rows and metadata after sync jobs."""
    global _catalog_v2_rows_cache
    global _catalog_v2_meta_cache
    _catalog_v2_rows_cache = None
    _catalog_v2_meta_cache = None
    _load_catalog_v2_rows()
    return get_catalog_v2_metadata()


def get_pricing_by_workload() -> dict[WorkloadType, list[PricingRecord]]:
    """Return all pricing rows grouped by workload type."""
    grouped: dict[WorkloadType, list[PricingRecord]] = {}
    for row in _load_all_pricing_records():
        grouped.setdefault(row.workload_type, []).append(row)
    return grouped


def validate_pricing_datasets() -> dict[str, int]:
    """Validate CSV datasets and return loaded row counts by filename."""
    return {path.name: len(_load_pricing_file(path)) for path in DATASET_FILES}


def _platform_type_from_billing(billing: str) -> str:
    if billing == "autoscaling":
        return "serverless"
    if billing in {"per_second", "hourly", "hourly_variable"}:
        return "dedicated"
    if billing == "per_token":
        return "model_based"
    return "marketplace"


def _hourly_rate_from_unit(unit_price_usd: float, unit_name: str) -> float:
    unit = unit_name.strip().lower()
    if unit == "gpu_second":
        return unit_price_usd * 3600.0
    if unit == "gpu_hour":
        return unit_price_usd
    raise ValueError(f"Unsupported GPU pricing unit '{unit_name}' for GPU-backed offering")


def _derive_llm_platforms_from_records(records: list[PricingRecord]) -> dict[str, Any]:
    derived: dict[str, Any] = {}
    per_token_prices: dict[str, dict[str, list[float]]] = {}

    for row in records:
        if row.workload_type != WorkloadType.LLM:
            continue

        platform_key = PROVIDER_KEY_ALIASES.get(row.provider, row.provider)
        platform = derived.setdefault(
            platform_key,
            {
                "type": _platform_type_from_billing(row.billing_type),
                "billing": row.billing_type,
            },
        )

        if row.billing_type == "per_token" and row.unit_name == "1m_tokens":
            model_prices = per_token_prices.setdefault(platform_key, {})
            model_prices.setdefault(row.model_key, []).append(row.unit_price_usd)
            continue

        if row.memory_gb is None:
            continue

        try:
            hourly_rate = _hourly_rate_from_unit(row.unit_price_usd, row.unit_name)
        except ValueError:
            continue

        gpu_entry = {
            "name": row.sku_name,
            "hourly_rate": hourly_rate,
            "memory_gb": row.memory_gb,
            "tokens_per_second": int(row.throughput_value or 8000),
        }
        platform.setdefault("gpus", {})[row.sku_key] = gpu_entry

    for platform_key, model_prices in per_token_prices.items():
        platform = derived.setdefault(
            platform_key,
            {
                "type": "model_based",
                "billing": "per_token",
            },
        )
        platform["models"] = {
            model_key: {"price_per_m_tokens": sum(prices) / len(prices)}
            for model_key, prices in model_prices.items()
        }

    return derived


def get_platforms(workload_type: WorkloadType | str = WorkloadType.LLM) -> dict[str, Platform]:
    """Load GPU platform catalog with pricing and specs.

    Returns:
        Dictionary mapping platform keys to platform configurations.
        Each platform includes GPU specs, billing type, and pricing.
    """
    from data.platforms import PLATFORMS

    # Runtime safety: validate MVP planning catalogs before recommendation paths run.
    validate_mvp_catalogs()

    if isinstance(workload_type, str):
        workload_type = _normalize_workload_type(workload_type, DATASET_FILES[0], 0)

    if workload_type != WorkloadType.LLM:
        return {}

    merged: dict[str, Any] = deepcopy(PLATFORMS)
    derived = _derive_llm_platforms_from_records(get_pricing_records(WorkloadType.LLM))

    for platform_key, derived_platform in derived.items():
        existing = merged.get(platform_key, {})
        combined = deepcopy(existing)

        # Keep existing type/billing defaults for compatibility when present.
        combined.setdefault("type", derived_platform.get("type"))
        combined.setdefault("billing", derived_platform.get("billing"))

        if "gpus" in derived_platform:
            combined.setdefault("gpus", {})
            combined["gpus"].update(derived_platform["gpus"])

        if "models" in derived_platform:
            combined.setdefault("models", {})
            combined["models"].update(derived_platform["models"])

        merged[platform_key] = combined

    return merged


def get_models() -> dict[str, ModelRequirement]:
    """Load LLM model memory requirements and specifications.

    Returns:
        Dictionary mapping model keys to model requirements.
        Includes recommended memory, display name, and parameter count.
    """
    from inference_atlas.performance_data import MODEL_REQUIREMENTS

    return MODEL_REQUIREMENTS


def get_model_display_name(model_key: str) -> str:
    """Get human-readable display name for a model key.

    Args:
        model_key: Internal model identifier (e.g., "llama_70b")

    Returns:
        Display name (e.g., "Llama 3.1 70B")

    Raises:
        KeyError: If model_key is not recognized
    """
    models = get_models()
    if model_key not in models:
        valid_keys = ", ".join(sorted(models.keys()))
        raise KeyError(f"Unknown model '{model_key}'. Valid options: {valid_keys}")
    return models[model_key]["display_name"]
