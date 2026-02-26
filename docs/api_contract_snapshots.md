# API Contract Snapshots Runbook

This project locks core API response shapes with golden snapshots under `tests/golden/api/`.

## What is covered
- Catalog browse contract
- Catalog rank contract
- LLM plan contract
- Report generate contract
- Invoice analyze contract

## CI gate
GitHub Actions workflow: `.github/workflows/contract-gate.yml`

Runs on PRs and on pushes to `main`/`v1`:
- `tests/test_api_contract_snapshots.py`
- `tests/test_api_endpoint_contract_snapshots.py`

## Update snapshots (intentional contract change only)
1. Regenerate goldens:

```bash
python3 scripts/update_api_contract_snapshots.py
```

2. Run tests:

```bash
pytest -q tests/test_api_contract_snapshots.py tests/test_api_endpoint_contract_snapshots.py
```

3. Review diffs carefully:

```bash
git diff -- tests/golden/api/*.json
```

4. Commit only when the contract change is intentional.

## Rules
- Do not update snapshots for unrelated feature work.
- Prefer additive response fields over breaking/removing fields.
- If breaking change is required, update frontend + tests + docs in the same PR.
