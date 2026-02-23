#!/usr/bin/env python3
"""Validate data/providers_csv files and print a quality report."""

from __future__ import annotations

from pathlib import Path

from inference_atlas.catalog_v2.connectors.provider_csv import list_csv_providers
from inference_atlas.catalog_v2.csv_quality import audit_provider_csv


def main() -> int:
    providers = list_csv_providers()
    csv_dir = Path("data/providers_csv")

    if not providers:
        print("No provider CSV files found in data/providers_csv")
        return 1

    failed = 0
    total_rows = 0
    print("Provider CSV validation report")
    print("=" * 80)
    for provider_id in providers:
        path = csv_dir / f"{provider_id}.csv"
        audit = audit_provider_csv(path, provider_id)
        total_rows += audit.row_count
        status = "OK" if audit.ok else "FAIL"
        if not audit.ok:
            failed += 1
        print(
            f"[{status}] {provider_id}: rows={audit.row_count}, "
            f"bad_price={audit.bad_price_rows}, "
            f"provider_mismatch={audit.provider_mismatch_rows}, "
            f"duplicates={audit.duplicate_key_rows}, "
            f"unknown_workload={audit.unknown_workload_rows}, "
            f"missing_cells={audit.missing_required_cells}, "
            f"missing_columns={list(audit.missing_columns)}"
        )

    print("-" * 80)
    print(
        f"providers={len(providers)} total_rows={total_rows} "
        f"failed_providers={failed}"
    )
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
