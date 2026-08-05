"""Rank causally suppressed features in a provided model profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from circuit_supernodes.suppression import compare_feature_profiles


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Discover suppressed features from aligned model profiles"
    )
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--min-occurrences", type=int, default=2)
    parser.add_argument("--min-reference-support", type=float, default=0.05)
    parser.add_argument("--min-absolute-suppression", type=float, default=0.05)
    parser.add_argument("--min-suppression-fraction", type=float, default=0.25)
    parser.add_argument("--min-consistency", type=float, default=0.6)
    args = parser.parse_args()

    profiles = json.loads(args.profiles.read_text())
    result = compare_feature_profiles(
        profiles["reference"],
        profiles["provided"],
        min_occurrences=args.min_occurrences,
        min_reference_support=args.min_reference_support,
        min_absolute_suppression=args.min_absolute_suppression,
        min_suppression_fraction=args.min_suppression_fraction,
        min_consistency=args.min_consistency,
    )
    rendered = json.dumps(result, indent=2) + "\n"
    if args.out is None:
        print(rendered, end="")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)
        print(f"wrote suppressed-feature report to {args.out}")


if __name__ == "__main__":
    main()
