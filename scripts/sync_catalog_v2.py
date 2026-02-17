#!/usr/bin/env python3
"""Sync catalog_v2 provider pricing rows into canonical JSON."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from inference_atlas.catalog_v2 import sync_catalog_v2  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync catalog_v2 providers")
    parser.add_argument(
        "--providers",
        nargs="*",
        default=["all"],
        help="Provider IDs to sync into catalog_v2 (use 'all' for every available provider)",
    )
    parser.add_argument(
        "--fail-on-empty",
        action="store_true",
        help="Exit non-zero if sync returns zero rows.",
    )
    args = parser.parse_args()
    payload = sync_catalog_v2(providers=args.providers)
    if args.fail_on_empty and int(payload["row_count"]) == 0:
        raise SystemExit("catalog_v2 sync produced zero rows")
    print(
        f"catalog_v2 synced: {payload['row_count']} rows "
        f"from providers={','.join(payload['providers_synced'])}"
    )


if __name__ == "__main__":
    main()
