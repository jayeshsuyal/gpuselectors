# API Reference

InferenceAtlas exposes an optional FastAPI backend for frontend real-mode and external integration.

## Install

```bash
pip install -e ".[api]"
```

## Run

```bash
python3 scripts/run_api_server.py
```

Default base URL: `http://127.0.0.1:8000`

## Environment

- `INFERENCE_ATLAS_CORS_ORIGINS`
  - comma-separated origins (default: `*`)
  - example: `http://localhost:5173,http://localhost:3000`

## Endpoints

### `GET /healthz`

Health check.

Response:

```json
{"status": "ok"}
```

### `POST /api/v1/plan/llm`

LLM workload ranking via planner engine.

Required fields:

- `tokens_per_day`
- `model_bucket`
- `top_k`

Returns:

- `plans`
- `provider_diagnostics`
- `warnings`

### `POST /api/v1/rank/catalog`

Catalog ranking for non-LLM and generic workload comparisons.

Supports adaptive filter orchestration:

- `strict`
- `relax_unit`
- `relax_budget`
- `relax_provider`

Returns:

- `offers`
- `provider_diagnostics`
- `warnings`
- `relaxation_applied`
- `relaxation_steps`
- `exclusion_breakdown`

### `GET /api/v1/catalog`

Browse catalog rows with optional filters:

- `workload_type`
- `provider`
- `unit_name`

### `POST /api/v1/invoice/analyze`

Multipart file upload endpoint.

Form field:

- `file` (CSV)

Returns normalized line items and provider totals.

### `POST /api/v1/ai/assist`

Grounded AI helper for provider/pricing Q&A.

Returns:

- `reply`
- `suggested_action` (optional)

### `POST /api/v1/ai/copilot`

Multi-turn copilot endpoint for guided configuration extraction.

Supports both payload styles:

- frontend shape: `{message, history, workload_type}`
- internal shape: `{user_text, state}`

## Error Semantics

- `422`: request validation error (schema-level)
- `400`: service-level invalid request (business validation)
- `500`: unhandled backend failure
