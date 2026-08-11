"""Build a promptless task library from pre-existing curated supernodes."""

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
    parsed: dict[str, list[str]] = {}
    for group in groups:
        if not isinstance(group, list) or len(group) < 2:
            raise ValueError(f"malformed curated supernode: {group!r}")
        name = str(group[0]).strip()
        nodes = [str(node_id) for node_id in group[1:]]
        if not name or name in parsed:
            raise ValueError(f"curated supernode name is empty or duplicated: {name!r}")
        if len(nodes) != len(set(nodes)):
            raise ValueError(f"curated supernode {name!r} repeats a node")
        parsed[name] = nodes
    if not parsed:
        raise ValueError("source graph has no curated supernodes")
    return parsed


def is_positioned_feature_id(node_id: str) -> bool:
    parts = node_id.split("_")
    return len(parts) == 3 and all(part.isdigit() for part in parts)


def graph_payload(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Accept either a raw circuit graph or a legacy embedded-graph task row."""

    embedded = record.get("graph")
    if isinstance(embedded, dict):
        provenance = record.get("source", {})
        return embedded, provenance if isinstance(provenance, dict) else {}
    return record, {}


def target_spec(value: str) -> tuple[str, str]:
    group_name, separator, concept = value.partition("=")
    group_name = group_name.strip()
    concept = concept.strip() if separator else group_name
    if not group_name or not concept:
        raise argparse.ArgumentTypeError("target must be GROUP or GROUP=CONCEPT")
    return group_name, concept


def task_targets(
    groups: dict[str, list[str]], requested: list[tuple[str, str]] | None
) -> list[tuple[str, str]]:
    """Use exact curator descriptions unless an explicit legacy filter is given."""

    if not requested:
        return [(name, name) for name in groups]
    lookup = {name.casefold(): name for name in groups}
    selected = []
    for requested_group, description in requested:
        canonical = lookup.get(requested_group.casefold())
        if canonical is None:
            raise ValueError(
                f"unknown curated group {requested_group!r}; "
                f"available: {sorted(groups)}"
            )
        selected.append((canonical, description))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build active hidden-supernode discovery tasks"
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="curated graph JSON path or URL; repeat to combine circuits",
    )
    parser.add_argument(
        "--target",
        action="append",
        type=target_spec,
        help=(
            "optional curated GROUP filter, with legacy GROUP=DESCRIPTION mapping; "
            "by default every curated group uses its exact curator-written name"
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.out.exists() and not args.overwrite:
        raise SystemExit(f"{args.out} exists; pass --overwrite to replace it")

    if args.target and len(args.source) != 1:
        raise SystemExit("--target may only be used with one --source")

    rows = []
    for source in args.source:
        data, inherited_source = graph_payload(load_source(source))
        graph = load_graph_json(data)
        groups = curated_groups(data)
        for canonical, description in task_targets(groups, args.target):
            reference_nodes = groups[canonical]
            invalid = sorted(
                node_id
                for node_id in reference_nodes
                if node_id not in graph.nodes
                or graph.nodes[node_id].kind != "feature"
                or not is_positioned_feature_id(node_id)
            )
            if invalid:
                raise ValueError(
                    "curated group contains invalid FEATURE nodes: "
                    f"{invalid[:4]}"
                )
            prompt = str(data.get("metadata", {}).get("prompt", "")).strip()
            target_token = graph.top_outputs[0][0] if graph.top_outputs else ""
            if not prompt or not target_token.strip():
                raise ValueError("source graph needs a prompt and a top output token")
            rows.append(
                {
                    "family": "curated_supernode_library",
                    "target_concept": description,
                    "reference_nodes": reference_nodes,
                    "grading": {"prompt": prompt, "target_token": target_token},
                    "source": {
                        "location": inherited_source.get("location", source),
                        "curated_group": canonical,
                        "slug": data.get("metadata", {}).get("slug"),
                        "subject_model": data.get("metadata", {}).get("scan"),
                    },
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".partial")
    temporary.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)
    )
    temporary.replace(args.out)
    print(f"wrote {len(rows)} active-discovery tasks to {args.out}")


if __name__ == "__main__":
    main()
