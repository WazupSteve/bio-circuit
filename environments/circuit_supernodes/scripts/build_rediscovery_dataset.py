"""Build hidden-reference supernode rediscovery rows from a curated graph."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any

from circuit_supernodes.graphs import load_graph_json


def load_source(source: str) -> dict[str, Any]:
    if source.startswith(("https://", "http://")):
        request = urllib.request.Request(
            source, headers={"User-Agent": "biocircuit-rediscovery/1"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    return json.loads(Path(source).read_text())


def curated_groups(data: dict[str, Any]) -> dict[str, list[str]]:
    groups = data.get("qParams", {}).get("supernodes", [])
    parsed = {
        str(group[0]): [str(node_id) for node_id in group[1:]]
        for group in groups
        if isinstance(group, list) and len(group) > 1
    }
    if not parsed:
        raise ValueError("source graph has no curated supernodes")
    return parsed


def target_spec(value: str) -> tuple[str, str]:
    group_name, separator, concept = value.partition("=")
    group_name = group_name.strip()
    concept = concept.strip() if separator else group_name
    if not group_name or not concept:
        raise argparse.ArgumentTypeError("target must be GROUP or GROUP=CONCEPT")
    return group_name, concept


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build zero-shot hidden-supernode rediscovery tasks"
    )
    parser.add_argument(
        "--source", required=True, help="curated graph JSON path or URL"
    )
    parser.add_argument(
        "--target",
        action="append",
        type=target_spec,
        required=True,
        help=(
            "curated GROUP, optionally mapped to the requested CONCEPT as "
            "GROUP=CONCEPT"
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.out.exists() and not args.overwrite:
        raise SystemExit(f"{args.out} exists; pass --overwrite to replace it")

    data = load_source(args.source)
    graph = load_graph_json(data)
    groups = curated_groups(data)
    lookup = {name.casefold(): name for name in groups}
    rows = []
    for requested_group, concept in args.target:
        canonical = lookup.get(requested_group.casefold())
        if canonical is None:
            raise ValueError(
                f"unknown curated group {requested_group!r}; "
                f"available: {sorted(groups)}"
            )
        reference_nodes = groups[canonical]
        missing = sorted(set(reference_nodes) - set(graph.nodes))
        if missing:
            raise ValueError(f"curated group contains missing nodes: {missing[:4]}")
        rows.append(
            {
                "family": "supernode_rediscovery",
                "target_concept": concept,
                "reference_nodes": reference_nodes,
                "graph": data,
                "source": {
                    "location": args.source,
                    "curated_group": canonical,
                    "slug": data.get("metadata", {}).get("slug"),
                },
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".partial")
    temporary.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)
    )
    temporary.replace(args.out)
    print(f"wrote {len(rows)} hidden-reference rediscovery tasks to {args.out}")


if __name__ == "__main__":
    main()
