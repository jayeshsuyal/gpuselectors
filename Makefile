PYTHON ?= python3
PYTEST ?= pytest

.PHONY: sync-catalog sync-catalog-verify validate-provider-csv dev-api dev-frontend dev

sync-catalog:
	$(PYTHON) scripts/sync_catalog_v2.py --providers all --fail-on-empty

validate-provider-csv:
	$(PYTHON) scripts/validate_providers_csv.py

sync-catalog-verify: sync-catalog validate-provider-csv
	$(PYTEST) -q tests/test_catalog_v2_sync.py tests/test_catalog_v2_data_loader.py tests/test_catalog_v2_csv_quality.py

dev-api:
	$(PYTHON) scripts/run_api_server.py --host 127.0.0.1 --port 8000

dev-frontend:
	cd frontend && VITE_USE_MOCK_API=false VITE_API_BASE_URL=http://127.0.0.1:8000 npm run dev

dev:
	@echo "Starting API on http://127.0.0.1:8000 and frontend on Vite dev server..."
	@$(PYTHON) scripts/run_api_server.py --host 127.0.0.1 --port 8000 & \
	API_PID=$$!; \
	cd frontend && VITE_USE_MOCK_API=false VITE_API_BASE_URL=http://127.0.0.1:8000 npm run dev; \
	kill $$API_PID 2>/dev/null || true
