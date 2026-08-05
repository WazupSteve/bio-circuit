"""Discover causal supernodes from a response-bank JSON file, without prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from circuit_supernodes.discovery import discover_supernodes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prompt-free supernode discovery from a causal response bank"
    )
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--max-groups", type=int, default=8)
    parser.add_argument("--min-group-size", type=int, default=2)
    parser.add_argument(
        "--partition-all",
        action="store_true",
        help="partition every bank feature instead of selecting a high-energy subset",
    )
    args = parser.parse_args()

    result = discover_supernodes(
        json.loads(args.bank.read_text()),
        max_groups=args.max_groups,
        min_group_size=args.min_group_size,
        select_features=not args.partition_all,
    )
    rendered = json.dumps(result, indent=2) + "\n"
    if args.out is None:
        print(rendered, end="")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)
        print(f"wrote prompt-free circuit to {args.out}")


if __name__ == "__main__":
    main()
