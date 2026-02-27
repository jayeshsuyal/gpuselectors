#!/usr/bin/env python3
"""Normalize incoming GPU pricing CSVs into gpu/ and managed/ datasets.

This script ingests provider *_gpu.csv files and splits rows into:
1) data/providers_csv/gpu/*.csv      (true GPU-hour offers)
2) data/providers_csv/managed/*.csv  (managed API effective-cost rows)
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


GPU_COLUMNS = [
    "provider",
    "gpu_type",
    "billing_mode",
    "price_per_gpu_hour_usd",
    "region",
    "workload_type",
    "throughput_value",
    "throughput_unit",
    "min_gpus",
    "max_gpus",
    "startup_latency_sec",
    "source_url",
    "confidence",
    "last_verified_at",
]

MANAGED_COLUMNS = [
    "provider",
    "service_type",
    "billing_mode",
    "effective_unit_price_usd",
    "unit_name",
    "region",
    "workload_type",
    "source_url",
    "confidence",
    "last_verified_at",
    "notes",
]

GPU_BILLING_MODES = {"dedicated_hourly", "autoscale_hourly"}


@dataclass(frozen=True)
class Counters:
    gpu_rows: int = 0
    managed_rows: int = 0
    skipped_rows: int = 0


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _write_rows(path: Path, fieldnames: list[str], rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _is_gpu_row(row: dict[str, str]) -> bool:
    mode = (row.get("billing_mode") or "").strip().lower()
    gpu_type = (row.get("gpu_type") or "").strip().lower()
    if mode in GPU_BILLING_MODES:
        return True
    if gpu_type and gpu_type not in {"managed", "api", "n/a"} and "managed" not in mode:
        return True
    return False


def _normalize_gpu_row(row: dict[str, str]) -> dict[str, str]:
    out = {k: (row.get(k, "") or "").strip() for k in GPU_COLUMNS}
    # Map legacy aliases.
    if not out["provider"] and row.get("provider"):
        out["provider"] = (row.get("provider") or "").strip()
    return out


def _normalize_managed_row(row: dict[str, str]) -> dict[str, str]:
    unit = (row.get("throughput_unit") or "").strip() or "unit"
    price = (row.get("price_per_gpu_hour_usd") or "").strip()
    sku = (row.get("model_or_sku") or "").strip()
    return {
        "provider": (row.get("provider") or "").strip(),
        "service_type": sku or "managed_api",
        "billing_mode": (row.get("billing_mode") or "").strip() or "managed_api",
        "effective_unit_price_usd": price,
        "unit_name": unit,
        "region": (row.get("region") or "").strip() or "global",
        "workload_type": (row.get("workload_type") or "").strip(),
        "source_url": (row.get("source_url") or "").strip(),
        "confidence": (row.get("confidence") or "").strip() or "estimated",
        "last_verified_at": (row.get("last_verified_at") or "").strip(),
        "notes": "Managed API pricing; not direct GPU-hour procurement.",
    }


def _out_file_name(provider: str) -> str:
    token = provider.strip().lower().replace(" ", "_")
    return f"{token}.csv"


def normalize(input_dir: Path, gpu_dir: Path, managed_dir: Path) -> Counters:
    totals = Counters()
    files = sorted(input_dir.glob("*_gpu.csv"))
    if not files:
        print(f"No *_gpu.csv files found in {input_dir}")
        return totals

    for file_path in files:
        rows = _read_rows(file_path)
        if not rows:
            continue

        provider = (rows[0].get("provider") or file_path.stem.replace("_gpu", "")).strip()
        gpu_rows: list[dict[str, str]] = []
        managed_rows: list[dict[str, str]] = []
        skipped = 0
        for row in rows:
            if not (row.get("provider") or "").strip():
                skipped += 1
                continue
            if _is_gpu_row(row):
                gpu_rows.append(_normalize_gpu_row(row))
            else:
                managed_rows.append(_normalize_managed_row(row))

        if gpu_rows:
            _write_rows(gpu_dir / _out_file_name(provider), GPU_COLUMNS, gpu_rows)
        if managed_rows:
            _write_rows(managed_dir / _out_file_name(provider), MANAGED_COLUMNS, managed_rows)

        totals = Counters(
            gpu_rows=totals.gpu_rows + len(gpu_rows),
            managed_rows=totals.managed_rows + len(managed_rows),
            skipped_rows=totals.skipped_rows + skipped,
        )
        print(
            f"{file_path.name}: gpu={len(gpu_rows)} managed={len(managed_rows)} skipped={skipped}"
        )
    return totals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/providers_csv/incoming"),
        help="Directory containing incoming *_gpu.csv files",
    )
    parser.add_argument(
        "--gpu-dir",
        type=Path,
        default=Path("data/providers_csv/gpu"),
        help="Output directory for normalized GPU-hour rows",
    )
    parser.add_argument(
        "--managed-dir",
        type=Path,
        default=Path("data/providers_csv/managed"),
        help="Output directory for managed API effective-cost rows",
    )
    args = parser.parse_args()

    counts = normalize(
        input_dir=args.input_dir,
        gpu_dir=args.gpu_dir,
        managed_dir=args.managed_dir,
    )
    print(
        "DONE: "
        f"gpu_rows={counts.gpu_rows}, managed_rows={counts.managed_rows}, skipped_rows={counts.skipped_rows}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
