import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

import modal
from circuit_supernodes.datagen import generate
from circuit_supernodes.graphs import load_graph_json, slim_graph
from circuit_supernodes.rubric import MODAL_APP, MODAL_CLS

DATA_DIR = Path(__file__).parent.parent / "circuit_supernodes" / "data"
NEURONPEDIA_FEATURES_URL = "https://www.neuronpedia.org/api/features"


def is_faithful(graph_json: dict, target: str) -> tuple[bool, str]:
    g = load_graph_json(graph_json)
    if not g.top_outputs:
        return False, ""
    top_tok = g.top_outputs[0][0]
    t, tt = target.strip(), top_tok.strip()
    return bool(tt) and t.startswith(tt), top_tok


def add_neuronpedia_labels(graph_json: dict, max_features: int = 60) -> int:
    nodes = slim_graph(load_graph_json(graph_json), max_features).feature_nodes
    features = {tuple(map(int, node.id.split("_")[:2])) for node in nodes}
    if not features:
        return 0
    payload = [
        {
            "modelId": "qwen3-4b",
            "layer": f"{layer}-transcoder-hp",
            "index": feature,
            "maxActsToReturn": 1,
        }
        for layer, feature in sorted(features)
    ]
    request = Request(
        NEURONPEDIA_FEATURES_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        records = json.load(response)
    if not isinstance(records, list):
        raise ValueError("unexpected Neuronpedia response")

    labels = {}
    for record in records:
        descriptions = record.get("explanations") or []
        label = next(
            (
                str(item.get("description", "")).strip()
                for item in descriptions
                if str(item.get("description", "")).strip()
            ),
            "",
        )
        if label:
            layer = str(record["layer"]).split("-", 1)[0]
            labels[f"{layer}_{record['index']}"] = label

    count = 0
    for node in graph_json.get("nodes", []):
        node_id = str(node.get("node_id", node.get("id", "")))
        if label := labels.get("_".join(node_id.split("_")[:2])):
            node["label"] = label
            count += 1
    return count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--families", nargs="*", help="restrict to these prompt families")
    ap.add_argument("--out", default=str(DATA_DIR / "dataset.jsonl"))
    ap.add_argument(
        "--neuronpedia-labels",
        action="store_true",
        help="label the 60 displayed features using one Neuronpedia batch request",
    )
    args = ap.parse_args()

    samples = generate(args.n, seed=args.seed, families=args.families)
    print(
        f"Sampled {len(samples)} prompts across families:",
        sorted({s.family for s in samples}),
    )

    attributor = modal.Cls.from_name(MODAL_APP, MODAL_CLS)()
    graphs = list(
        attributor.attribute_prompt.map([s.prompt for s in samples], order_outputs=True)
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    kept, stats = 0, {}
    with tmp.open("w") as f:
        for s, graph in zip(samples, graphs, strict=True):
            if graph is None:
                print(f"  ! attribution failed for {s.prompt!r}, skipping")
                continue
            faithful, top_tok = is_faithful(graph, s.target)
            stats.setdefault(s.family, []).append(faithful)
            if not faithful:
                print(
                    f"  ! unfaithful [{s.family}] {s.prompt!r}: "
                    f"model says {top_tok!r}, wanted {s.target!r} — skipping"
                )
                continue
            if args.neuronpedia_labels:
                try:
                    labeled = add_neuronpedia_labels(graph)
                    print(f"  + added {labeled} Neuronpedia feature labels")
                except (OSError, ValueError) as exc:
                    print(f"  ! Neuronpedia labels unavailable: {exc}")
            row = s.to_row()
            row["graph"] = graph
            f.write(json.dumps(row) + "\n")
            kept += 1
    print(
        "faithfulness by family:",
        {k: f"{sum(v)}/{len(v)}" for k, v in sorted(stats.items())},
    )
    if not kept:
        tmp.unlink()
        raise SystemExit("no faithful graphs generated; existing dataset preserved")
    tmp.replace(out)
    print(f"Wrote {kept} rows -> {out}")


if __name__ == "__main__":
    main()
