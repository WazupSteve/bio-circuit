import argparse
import json
from pathlib import Path

from circuit_supernodes.graphs import (
    Graph,
    embed_supernodes,
    load_graph_json,
    slim_graph,
)
from circuit_supernodes.rubric import (
    MAX_GROUPS,
    completion_text,
    parse_proposal,
    valid_supernodes,
)

DATA_DIR = Path(__file__).parent.parent / "circuit_supernodes" / "data"
DEFAULT_SCAN = "qwen3-4b"
DEFAULT_SOURCE_SET = "transcoder-hp"


def read_proposal(path: Path) -> dict:
    text = path.read_text()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = parse_proposal(text)
    if isinstance(data, dict) and ("completion" in data or "nodes" in data):
        data = parse_proposal(completion_text(data))
    if not isinstance(data, dict) or not isinstance(data.get("supernodes"), list):
        raise SystemExit(f"no proposal found in {path}")
    return data


def prepare_neuronpedia_graph(
    raw: dict,
    supernodes: dict[str, list[str]],
    slug: str,
    scan: str = DEFAULT_SCAN,
    source_set: str = DEFAULT_SOURCE_SET,
) -> dict:
    """Attach agent supernodes and upload-ready Neuronpedia metadata."""
    graph = embed_supernodes(raw, supernodes, slug)
    metadata = dict(graph.get("metadata", {}))
    feature_details = dict(metadata.get("feature_details", {}))
    feature_details["neuronpedia_source_set"] = source_set
    metadata.update(
        {
            "scan": scan,
            "feature_details": feature_details,
        }
    )
    graph["metadata"] = metadata
    return graph


def prune_for_agent_view(raw: dict, max_features: int = 60) -> tuple[dict, Graph]:
    """Return the same feature/embedding/logit subgraph shown to the agent."""
    slim = slim_graph(load_graph_json(raw), max_features)
    keep = set(slim.nodes)
    graph = {
        **raw,
        "nodes": [
            node
            for node in raw.get("nodes", [])
            if str(node.get("node_id", node.get("id"))) in keep
        ],
        "links": [
            edge
            for edge in raw.get("links", [])
            if str(edge.get("source")) in keep and str(edge.get("target")) in keep
        ],
    }
    return graph, slim


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset.jsonl")
    ap.add_argument("--index", type=int, default=0, help="dataset row to export")
    ap.add_argument("--proposal", required=True, type=Path)
    ap.add_argument("--slug", default="", help="graph slug shown in the UI")
    ap.add_argument("--scan", default=DEFAULT_SCAN, help="Neuronpedia model ID")
    ap.add_argument(
        "--source-set",
        default=DEFAULT_SOURCE_SET,
        help="Neuronpedia feature source set",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    lines = (DATA_DIR / args.dataset).read_text().splitlines()
    row = json.loads([ln for ln in lines if ln.strip()][args.index])
    data = read_proposal(args.proposal)
    agent_graph, slim = prune_for_agent_view(row["graph"])
    sns = valid_supernodes(data, slim)
    sns = dict(list(sns.items())[:MAX_GROUPS])
    if not sns:
        raise SystemExit("proposal has no valid supernodes for this graph")

    out = args.out or Path(f"{args.slug or 'circuit'}-{args.index}.json")
    graph = prepare_neuronpedia_graph(
        agent_graph, sns, args.slug, args.scan, args.source_set
    )
    out.write_text(json.dumps(graph, indent=2) + "\n")
    print(
        f"{row['family']}: {len(sns)} supernodes, "
        f"{len(graph['nodes'])} nodes, {len(graph['links'])} links -> {out}"
    )
    print(f"Neuronpedia metadata: {args.scan} / {args.source_set}")


if __name__ == "__main__":
    main()
