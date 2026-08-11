import argparse
import hashlib
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

import modal
from circuit_supernodes.datagen import generate
from circuit_supernodes.graphs import graph_quality, load_graph_json, slim_graph
from circuit_supernodes.interventions import CORRUPTION_VERSION
from circuit_supernodes.restoration import MODAL_APP, MODAL_CLS, REWARD_VERSION

sys.path.insert(0, str(Path(__file__).parent.parent))  # for restoration backend

DATA_DIR = (
    Path(__file__).parent.parent / "circuit_supernodes" / "data" / "output_restoration"
)
NEURONPEDIA_FEATURES_URL = "https://www.neuronpedia.org/api/features"


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a dataset without loading its multi-gigabyte contents into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


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


def is_clean(
    graph_json: dict, max_error: float, max_features: int
) -> tuple[bool, dict]:
    """Screen on graph quality, not just on whether the answer was right.

    Only about a quarter of prompts yield a graph whose circuit is mostly
    explained rather than mostly error, and an unexplained graph teaches nothing
    however well the policy reads it.
    """
    stats = graph_quality(load_graph_json(graph_json), max_features)
    return stats["error"] <= max_error, stats


def write_manifest(out: Path, rows: int, args, quality: list[dict]) -> Path:
    """Pin what a split was built from, next to the split itself.

    Attribution graphs are expensive enough that nobody rebuilds one to check a
    result, so a split has to carry its own provenance: the subject model and
    transcoders, the exporter thresholds, the screening that admitted a graph,
    and a digest of the file itself.
    """
    from modal_app.attribution_restoration import (
        EDGE_THRESHOLD,
        MODEL_NAME,
        NODE_THRESHOLD,
        TRANSCODER_SET,
    )

    manifest = {
        "dataset": out.name,
        "rows": rows,
        "sha256": file_sha256(out),
        "subject_model": MODEL_NAME,
        "transcoders": TRANSCODER_SET,
        "node_threshold": NODE_THRESHOLD,
        "edge_threshold": EDGE_THRESHOLD,
        "reward_version": REWARD_VERSION,
        "corruption_version": CORRUPTION_VERSION,
        "screening": {
            "clean_target_matches_family_sample": True,
            "corruption_changes_top_token": True,
            "all_corrupted_nodes_exported": True,
            "blanket_visible_inverse_restoration_rejected": True,
            "max_error_share": args.max_error_share,
        },
        "generator": {
            "seed": args.seed,
            "requested": args.n,
            "families": sorted(args.families or []),
        },
        "graph_quality": {
            "error_influence": (
                sum(q["error"] for q in quality) / len(quality) if quality else None
            ),
            "displayed_feature_influence": (
                sum(q["displayed"] for q in quality) / len(quality) if quality else None
            ),
        },
    }
    path = out.with_suffix(".manifest.json")
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--families", nargs="*", help="restrict to these prompt families")
    ap.add_argument("--out", default=str(DATA_DIR / "output-restoration.jsonl"))
    ap.add_argument(
        "--max-error-share",
        type=float,
        default=0.35,
        help="reject graphs where error nodes carry more than this influence",
    )
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
    sample_seeds = [
        int.from_bytes(
            hashlib.blake2b(
                f"{args.seed}:{index}:{sample.prompt}".encode(), digest_size=8
            ).digest(),
            "big",
        )
        for index, sample in enumerate(samples)
    ]
    corrupted = list(
        attributor.build_corrupted_sample.map(
            [s.prompt for s in samples],
            [s.target for s in samples],
            sample_seeds,
            order_outputs=True,
        )
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    kept, stats, quality = 0, {}, []
    with tmp.open("w") as f:
        for s, result in zip(samples, corrupted, strict=True):
            if result is None:
                print(f"  ! corrupted attribution failed for {s.prompt!r}, skipping")
                continue
            if rejected := result.get("rejected"):
                stats.setdefault(s.family, []).append(False)
                print(f"  ! rejected [{s.family}] {s.prompt!r}: {rejected}")
                continue
            stats.setdefault(s.family, []).append(True)
            graph = result["graph"]
            clean, graph_stats = is_clean(graph, args.max_error_share, 60)
            if not clean:
                print(
                    f"  ! low-quality graph [{s.family}] {s.prompt!r}: "
                    f"error nodes carry {graph_stats['error']:.0%} of influence "
                    "— skipping"
                )
                continue
            quality.append(graph_stats)
            if args.neuronpedia_labels:
                try:
                    labeled = add_neuronpedia_labels(graph)
                    print(f"  + added {labeled} Neuronpedia feature labels")
                except (OSError, ValueError) as exc:
                    print(f"  ! Neuronpedia labels unavailable: {exc}")
            row = s.to_row()
            row["graph"] = graph
            row["target_token"] = result["target_token"]
            row["corruption"] = result["corruption"]
            row["corruption_id"] = result["corruption_id"]
            row["corruption_version"] = result["corruption_version"]
            row["corruption_attempt"] = result["corruption_attempt"]
            f.write(json.dumps(row) + "\n")
            kept += 1
    print(
        "accepted corruptions by family:",
        {k: f"{sum(v)}/{len(v)}" for k, v in sorted(stats.items())},
    )
    if quality:
        print(
            f"graph quality: error influence "
            f"{sum(q['error'] for q in quality) / len(quality):.1%}, "
            f"top-60 features cover "
            f"{sum(q['displayed'] for q in quality) / len(quality):.1%}"
            f" of feature influence"
        )
    if not kept:
        tmp.unlink()
        raise SystemExit("no faithful graphs generated; existing dataset preserved")
    tmp.replace(out)
    print(f"Wrote {kept} rows -> {out}")
    print(f"Wrote manifest -> {write_manifest(out, kept, args, quality)}")


if __name__ == "__main__":
    main()
