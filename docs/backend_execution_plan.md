# Backend Execution Plan (v1 -> v3.1)

## Scope
Strengthen engine reliability, keep ranking math deterministic, and add report-generation + future quality/scaling modules without breaking API consumers.

## Non-Negotiables
- Keep core ranking math deterministic and test-locked.
- Keep API response contracts stable.
- Additive changes only; avoid breaking existing callers.
- Surface warnings/errors explicitly (no silent failures).

## Success Metrics
### Technical Acceptance
- [ ] `pytest` passes with 0 failures, 0 errors.
- [ ] No hidden I/O blending leakage in planner outputs.
- [ ] Pydantic contracts validate request/response boundaries.
- [ ] Non-LLM and LLM ranking endpoints return actionable warnings.
- [ ] Core-module coverage >= 90%.

### Code Quality
- [ ] Service layer remains decoupled from UI.
- [ ] Enums/constants replace magic strings in new code.
- [ ] Error messages include field + reason.
- [ ] API contract snapshots for core endpoints are green.

### Risk Mitigation
- [ ] Existing `rank_configs()` behavior/signature remains stable.
- [ ] Validation bounds match real UI ranges.
- [ ] Integration tests marked; fast unit tests default.

## Phased Plan
## Phase 1 (Days 1-3): Contracts + Guardrails
- Add/extend Pydantic contracts for ranking/report payloads.
- Add contract snapshot tests:
  - `/api/v1/plan/llm`
  - `/api/v1/rank/catalog`
  - `/api/v1/report/generate`
- Lock deterministic ordering and fallback step order.

Exit Criteria:
- [ ] Contract tests pass.
- [ ] Determinism tests pass.

## Phase 2 (Days 4-5): Service Layer Hardening
- Ensure orchestration flow remains:
  - `strict -> relax_unit -> relax_budget -> relax_provider`
- Add exclusion reason aggregation and bounded warning summaries.
- Keep report generation deterministic (no LLM dependency in core export).

Exit Criteria:
- [ ] Orchestration tests green.
- [ ] Exclusion breakdown tests green.
- [ ] Report service tests green.

## Phase 3 (Days 6-7): Endpoint Reliability
- Add endpoint-level tests (happy + failure-path).
- Ensure actionable HTTP error messages.
- Validate no regressions in invoice + copilot + assist endpoints.

Exit Criteria:
- [ ] API integration tests green.
- [ ] Error payloads validated.

## Phase 4 (Days 8-9): Edge Cases
- Stress test unit mismatch, budget-zero, provider-empty, missing throughput.
- Add compatibility tests for provider/workload intersections.
- Validate fallback warnings and diagnostics coverage.

Exit Criteria:
- [ ] Edge-case tests green.
- [ ] No hidden failure paths.

## Phase 5 (Days 10-12): Regression + Freeze
- Full suite pass on backend.
- Generate release readiness checklist.
- Prepare changelog for merge.

Exit Criteria:
- [ ] Full backend regression green.
- [ ] Release checklist complete.

## Commands
```bash
pytest -q tests
pytest -q tests/test_api_service.py tests/test_api_server.py
pytest -q tests/test_catalog_ranking.py tests/test_math_engine.py
```

## v2/v3.1 Extension Hooks (Backend)
- Quality score module (`quality_scoring.py`) with workload-aware weights.
- Quality-vs-price insights endpoint for chart payloads.
- Scaling planner payloads (replicas, capacity rationale).
- Training/fine-tuning lifecycle cost model.
- Invoice Analyzer++ remediation/scenario simulation.

