import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

import modal
from circuit_supernodes.datagen import (
    EVAL_FAMILIES,
    TRAIN_FAMILIES,
    CircuitSample,
    generate,
)
from circuit_supernodes.graphs import graph_quality, load_graph_json, slim_graph
from circuit_supernodes.rubric import MODAL_APP, MODAL_CLS, REWARD_VERSION
from circuit_supernodes.sampling import (
    MAX_NEW_TOKENS,
    SamplingResult,
    classify_result,
    sampling_request,
)

sys.path.insert(0, str(Path(__file__).parent.parent))  # for modal_app.attribution

DATA_DIR = Path(__file__).parent.parent / "circuit_supernodes" / "data"
NEURONPEDIA_FEATURES_URL = "https://www.neuronpedia.org/api/features"


@dataclass
class PreparedSample:
    sample: CircuitSample
    target_token: str
    disposition: str
    sampling: dict


def _sample_answers(attributor, samples: list[CircuitSample]) -> list[SamplingResult]:
    requests = [
        sampling_request(sample.prompt, sample.possible_answers) for sample in samples
    ]
    return list(attributor.sample_until_answer.map(requests, order_outputs=True))


def _prepared(
    sample: CircuitSample,
    result: SamplingResult,
    disposition: str,
    sampling: dict,
) -> PreparedSample:
    return PreparedSample(
        sample=CircuitSample(
            family=sample.family,
            prompt=str(result["provided_prompt"]),
            target=sample.target,
            possible_answers=sample.possible_answers,
            source_family=sample.source_family,
        ),
        target_token=str(result["target_token"]),
        disposition=disposition,
        sampling=sampling,
    )


def prepare_samples(
    attributor, samples: list[CircuitSample]
) -> tuple[list[PreparedSample], dict[str, int]]:
    """Keep correct answers and record, but reject, all other outcomes."""

    counts = {"keep_correct": 0, "keep_incorrect": 0, "reject": 0}
    prepared: list[PreparedSample] = []

    initial_results = _sample_answers(attributor, samples)
    for sample, result in zip(samples, initial_results, strict=True):
        disposition = classify_result(sample.target, result)
        counts[disposition] += 1
        if disposition == "keep_correct":
            prepared.append(
                _prepared(
                    sample,
                    result,
                    disposition,
                    {"initial": result, "max_new_tokens": MAX_NEW_TOKENS},
                )
            )
    return prepared, counts


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a dataset without loading its multi-gigabyte contents into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


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
    from modal_app.attribution import (
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
        "screening": {
            "exact_first_token": False,
            "max_error_share": args.max_error_share,
        },
        "generator": {
            "seed": args.seed,
            "requested": args.n,
            "families": sorted(args.families or []),
            "answer_sampling": {
                "max_new_tokens": MAX_NEW_TOKENS,
                "triage": getattr(args, "triage", None),
            },
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
    ap.add_argument(
        "--split",
        choices=("train", "eval"),
        help="use the explicit training or sealed evaluation family set",
    )
    ap.add_argument("--out", default=str(DATA_DIR / "dataset.jsonl"))
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

    if args.split and args.families is not None:
        ap.error("--split and --families are mutually exclusive")
    if args.split == "train":
        args.families = list(TRAIN_FAMILIES)
    elif args.split == "eval":
        args.families = list(EVAL_FAMILIES)
    elif args.families is None:
        args.families = list(TRAIN_FAMILIES)

    samples = generate(args.n, seed=args.seed, families=args.families)
    print(
        f"Sampled {len(samples)} prompts across families:",
        sorted({s.family for s in samples}),
    )

    attributor = modal.Cls.from_name(MODAL_APP, MODAL_CLS)()
    prepared, triage = prepare_samples(attributor, samples)
    args.triage = triage
    print(f"answer sampling triage: {triage}")
    if not prepared:
        raise SystemExit("sampling produced no attributable answers")
    graphs = list(
        attributor.attribute_prompt.map(
            [item.sample.prompt for item in prepared], order_outputs=True
        )
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    kept, stats, quality = 0, {}, []
    with tmp.open("w") as f:
        for item, graph in zip(prepared, graphs, strict=True):
            s = item.sample
            if graph is None:
                print(f"  ! attribution failed for {s.prompt!r}, skipping")
                continue
            faithful, top_tok = is_faithful(graph, item.target_token)
            stats.setdefault(s.family, []).append(faithful)
            if not faithful:
                print(
                    f"  ! unfaithful [{s.family}] {s.prompt!r}: "
                    f"model says {top_tok!r}, sampled {item.target_token!r} — skipping"
                )
                continue
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
            row["sampling"] = item.sampling
            row["sampling"]["disposition"] = item.disposition
            row["sampling"]["attributed_target_token"] = item.target_token
            row["graph"] = graph
            f.write(json.dumps(row) + "\n")
            kept += 1
    print(
        "faithfulness by family:",
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
