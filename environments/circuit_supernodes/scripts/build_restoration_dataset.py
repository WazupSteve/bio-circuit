"""Build common-corruption discovery and held-out prompt cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import modal
from circuit_supernodes.cohorts import stable_feature_id
from circuit_supernodes.datagen import (
    EVAL_FAMILIES,
    FAMILIES,
    TRAIN_FAMILIES,
    generate_pool,
)
from circuit_supernodes.graphs import graph_quality, load_graph_json
from circuit_supernodes.interventions import (
    CORRUPTION_VERSION,
    MAX_CORRUPTION_NODES,
    MIN_CORRUPTION_NODES,
)
from circuit_supernodes.restoration import MODAL_APP, MODAL_CLS, REWARD_VERSION
from circuit_supernodes.sampling import MAX_NEW_TOKENS
from scripts.build_mdl_dataset import is_faithful, prepare_samples

DATA_DIR = (
    Path(__file__).parent.parent / "circuit_supernodes" / "data" / "output_restoration"
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def common_stable_features(graphs: list[dict], limit: int = 512) -> list[str]:
    """Intersect ordinary top-k stable feature identities across clean prompts."""

    feature_sets = []
    ranks: dict[str, int] = defaultdict(int)
    for graph_json in graphs:
        graph = load_graph_json(graph_json)
        ordered = sorted(
            (node for node in graph.feature_nodes if node.activation > 0.0),
            key=lambda node: -node.share,
        )[:limit]
        current = set()
        for rank, node in enumerate(ordered):
            stable = stable_feature_id(node.id)
            if stable is not None:
                current.add(stable)
                ranks[stable] += rank
        feature_sets.append(current)
    shared = set.intersection(*feature_sets) if feature_sets else set()
    return sorted(shared, key=lambda node_id: (ranks[node_id], node_id))


def random_corruption_spec(
    common: list[str], rng: random.Random
) -> list[tuple[int, int, float]]:
    maximum = min(MAX_CORRUPTION_NODES, 12, len(common))
    if maximum < MIN_CORRUPTION_NODES:
        return []
    count = rng.randint(MIN_CORRUPTION_NODES, maximum)
    selected = rng.sample(common, count)
    return [
        (*map(int, node_id.split("_")), round(2 ** rng.uniform(1.0, 3.0), 6))
        for node_id in selected
    ]


def all_changed(results: list[dict]) -> bool:
    return bool(results) and all(
        isinstance(result, dict)
        and not result.get("rejected")
        and result.get("changed_top_token") is True
        for result in results
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=480, help="candidate prompt count")
    parser.add_argument("--tasks", type=int, default=16)
    parser.add_argument("--cohort-size", type=int, default=5)
    parser.add_argument("--heldout-size", type=int, default=5)
    parser.add_argument("--attempts", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=("train", "eval", "all"), default="all")
    parser.add_argument("--families", nargs="*")
    parser.add_argument("--out", default=str(DATA_DIR / "suppressed-features.jsonl"))
    parser.add_argument("--max-error-share", type=float, default=0.35)
    args = parser.parse_args()
    if args.families is not None and args.split != "all":
        parser.error("--families and an explicit --split are mutually exclusive")
    if args.families is not None:
        families = args.families
    elif args.split == "train":
        families = list(TRAIN_FAMILIES)
    elif args.split == "eval":
        families = list(EVAL_FAMILIES)
    else:
        families = list(FAMILIES)

    samples = generate_pool(args.n, seed=args.seed, families=families)
    attributor = modal.Cls.from_name(MODAL_APP, MODAL_CLS)()
    prepared, triage = prepare_samples(attributor, samples)
    clean_graphs = list(
        attributor.attribute_prompt.map(
            [item.sample.prompt for item in prepared], order_outputs=True
        )
    )

    by_family: dict[str, list[dict]] = defaultdict(list)
    quality = []
    for item, graph_json in zip(prepared, clean_graphs, strict=True):
        if not graph_json:
            continue
        faithful, _ = is_faithful(graph_json, item.target_token)
        stats = graph_quality(load_graph_json(graph_json), 60)
        if not faithful or stats["error"] > args.max_error_share:
            continue
        quality.append(stats)
        by_family[item.sample.family].append(
            {
                "sample": item.sample,
                "target_token": item.target_token,
                "sampling": {
                    **item.sampling,
                    "disposition": item.disposition,
                    "attributed_target_token": item.target_token,
                },
                "clean_graph": graph_json,
            }
        )

    rng = random.Random(args.seed)
    group_size = args.cohort_size + args.heldout_size
    candidate_groups = []
    for family, records in sorted(by_family.items()):
        rng.shuffle(records)
        candidate_groups.extend(
            (family, records[offset : offset + group_size])
            for offset in range(0, len(records) - group_size + 1, group_size)
        )
    rng.shuffle(candidate_groups)

    rows = []
    for family, records in candidate_groups:
        common = common_stable_features([record["clean_graph"] for record in records])
        if len(common) < MIN_CORRUPTION_NODES:
            continue
        prompts = [record["sample"].prompt for record in records]
        targets = [record["sample"].target for record in records]
        selected_spec = None
        for _ in range(args.attempts):
            spec = random_corruption_spec(common, rng)
            tested = list(
                attributor.test_corruption_spec.map(
                    prompts, targets, [spec] * len(records), order_outputs=True
                )
            )
            if all_changed(tested):
                selected_spec = spec
                break
        if selected_spec is None:
            continue

        corrupted = list(
            attributor.build_corrupted_sample_from_spec.map(
                prompts,
                targets,
                [selected_spec] * len(records),
                order_outputs=True,
            )
        )
        if not all_changed(corrupted) or any("graph" not in item for item in corrupted):
            continue

        examples = []
        for record, result in zip(records, corrupted, strict=True):
            examples.append(
                {
                    **record["sample"].to_row(),
                    "target_token": result["target_token"],
                    "sampling": record["sampling"],
                    "graph": result["graph"],
                    "corruption": result["corruption"],
                    "corruption_id": result["corruption_id"],
                    "corruption_version": CORRUPTION_VERSION,
                }
            )
        rows.append(
            {
                "family": family,
                "cohort": examples[: args.cohort_size],
                "heldout": examples[args.cohort_size :],
                "corruption_spec_id": hashlib.sha256(
                    json.dumps(selected_spec, separators=(",", ":")).encode()
                ).hexdigest()[:16],
            }
        )
        if len(rows) == args.tasks:
            break

    if len(rows) < args.tasks:
        raise SystemExit(
            f"formed only {len(rows)}/{args.tasks} corruptible cohorts; increase "
            "--n/--attempts or reduce cohort sizes"
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".partial")
    temporary.write_text("".join(json.dumps(row) + "\n" for row in rows))
    temporary.replace(out)
    manifest = {
        "dataset": out.name,
        "rows": len(rows),
        "sha256": file_sha256(out),
        "subject_backend": f"{MODAL_APP}/{MODAL_CLS}",
        "reward_version": REWARD_VERSION,
        "corruption_version": CORRUPTION_VERSION,
        "maximal_graphs": True,
        "generator": {
            "seed": args.seed,
            "candidates": args.n,
            "families": families,
            "cohort_size": args.cohort_size,
            "heldout_size": args.heldout_size,
            "corruption_attempts": args.attempts,
            "corruption_nodes": [MIN_CORRUPTION_NODES, min(12, MAX_CORRUPTION_NODES)],
            "answer_sampling": {
                "max_new_tokens": MAX_NEW_TOKENS,
                "triage": triage,
            },
        },
        "mean_error_share": (
            sum(item["error"] for item in quality) / len(quality) if quality else None
        ),
    }
    manifest_path = out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(rows)} suppressed-feature cohorts to {out}")
    print(f"wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
