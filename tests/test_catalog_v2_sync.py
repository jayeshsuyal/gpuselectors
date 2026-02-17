from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from inference_atlas.catalog_v2 import sync_catalog_v2


def test_sync_catalog_v2_openai_and_validate_schema() -> None:
    payload = sync_catalog_v2(providers=["openai"])
    assert payload["row_count"] >= 1
    assert "openai" in payload["providers_synced"]

    catalog_path = Path("data/catalog_v2/pricing_catalog.json")
    schema_path = Path("data/catalog_v2/pricing_catalog.schema.json")
    data = json.loads(catalog_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    validator = Draft202012Validator(schema)
    errors = list(validator.iter_errors(data))
    assert not errors


def test_sync_catalog_v2_all_providers_includes_expected_count() -> None:
    payload = sync_catalog_v2(providers=["all"])
    providers = payload["providers_synced"]
    assert isinstance(providers, list)
    assert len(providers) == 16
    assert "openai" in providers
    assert "anthropic" in providers
