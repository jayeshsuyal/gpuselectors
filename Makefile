PYTHON ?= python3
PYTEST ?= pytest

.PHONY: sync-catalog sync-catalog-verify validate-provider-csv

sync-catalog:
	$(PYTHON) scripts/sync_catalog_v2.py --providers all --fail-on-empty

validate-provider-csv:
	$(PYTHON) scripts/validate_providers_csv.py

sync-catalog-verify: sync-catalog validate-provider-csv
	$(PYTEST) -q tests/test_catalog_v2_sync.py tests/test_catalog_v2_data_loader.py tests/test_catalog_v2_csv_quality.py
