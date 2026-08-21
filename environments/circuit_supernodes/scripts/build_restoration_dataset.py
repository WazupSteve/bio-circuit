"""Build common-corruption discovery and held-out prompt cohorts."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Literal

import typer
from circuit_supernodes.grading.restoration import MODAL_APP, MODAL_CLS
from circuit_supernodes.models import (
    CircuitDatasetRow,
    DatasetExample,
    TracerGraphPayload,
)
from circuit_supernodes.runtime.corruption import (
    CORRUPTION_VERSION,
    MAX_CORRUPTION_NODES,
    MIN_CORRUPTION_NODES,
)
from circuit_supernodes.runtime.graphs import (
    MAX_FEATURE_NODES,
    feature_nodes,
    influence_deltas,
    stable_feature_id,
)
from scripts.dataset_pipeline import (
    clean_attributions,
    finish_build,
    start_build,
)

DATA_DIR = (
    Path(__file__).parent.parent / "circuit_supernodes" / "data" / "output_restoration"
)


def common_stable_features(
    graphs: list[TracerGraphPayload], limit: int = MAX_FEATURE_NODES
) -> list[str]:
    """Intersect ordinary top-k stable feature identities across clean prompts."""

    feature_sets = []
    ranks: dict[str, int] = defaultdict(int)
    for graph in graphs:
        deltas = influence_deltas(graph)
        ordered = sorted(
            (node for node in feature_nodes(graph) if (node.activation or 0.0) > 0.0),
            key=lambda node: -deltas[node.node_id],
        )[:limit]
        current = set()
        for rank, node in enumerate(ordered):
            try:
                stable = stable_feature_id(node.node_id)
            except Exception:
                continue
            current.add(stable)
            ranks[stable] += rank
        feature_sets.append(current)
    shared = set.intersection(*feature_sets) if feature_sets else set()
    return sorted(shared, key=lambda node_id: (ranks[node_id], node_id))


def stable_feature_positions(graph: TracerGraphPayload) -> dict[str, int]:
    """Resolve each stable feature to its highest-delta stored graph position."""

    deltas = influence_deltas(graph)
    positions: dict[str, int] = {}
    ordered = sorted(
        (node for node in feature_nodes(graph) if (node.activation or 0.0) > 0.0),
        key=lambda node: -deltas[node.node_id],
    )[:MAX_FEATURE_NODES]
    for node in ordered:
        try:
            stable = stable_feature_id(node.node_id)
        except Exception:
            continue
        positions.setdefault(stable, int(node.node_id.rsplit("_", 1)[1]))
    return positions


def coordinate_specs(
    stable_spec: list[tuple[int, int, float]],
    position_maps: list[dict[str, int]],
) -> list[list[tuple[int, int, int, float]]]:
    """Expand one stable corruption into prompt-specific stored coordinates."""

    return [
        [
            (layer, feature, positions[f"{layer}_{feature}"], factor)
            for layer, feature, factor in stable_spec
        ]
        for positions in position_maps
    ]


def random_corruption_spec(
    common: list[str], rng: random.Random
) -> list[tuple[int, int, float]]:
    maximum = min(MAX_CORRUPTION_NODES, 12, len(common))
    if maximum < MIN_CORRUPTION_NODES:
        return []
    count = rng.randint(MIN_CORRUPTION_NODES, maximum)
    selected = rng.sample(common, count)
    corruption = []
    for node_id in selected:
        layer, feature = map(int, node_id.split("_"))
        corruption.append((layer, feature, round(2 ** rng.uniform(1.0, 3.0), 6)))
    return corruption


def all_changed(results: list[dict]) -> bool:
    return bool(results) and all(
        isinstance(result, dict)
        and not result.get("rejected")
        and result.get("changed_top_token") is True
        for result in results
    )


def main(
    n: Annotated[int, typer.Option(min=1, help="Candidate prompts to sample.")] = 480,
    tasks: Annotated[
        int, typer.Option(min=1, help="Corrupted cohort rows to produce.")
    ] = 16,
    cohort_size: Annotated[
        int,
        typer.Option(
            min=1,
            help="First contribution to the hidden positive cohort in each row.",
        ),
    ] = 5,
    heldout_size: Annotated[
        int,
        typer.Option(
            min=1,
            help="Second contribution to the hidden positive cohort in each row.",
        ),
    ] = 5,
    attempts: Annotated[
        int,
        typer.Option(min=1, help="Shared-corruption attempts per candidate cohort."),
    ] = 32,
    seed: Annotated[int, typer.Option(help="Deterministic sampling seed.")] = 0,
    split: Annotated[
        Literal["train", "eval", "all"],
        typer.Option(help="Family split to build; 'all' includes train and eval."),
    ] = "all",
    family: Annotated[
        list[str] | None,
        typer.Option(
            "--family",
            "--families",
            help="Prompt family to include; repeat for multiple families.",
        ),
    ] = None,
    out: Annotated[
        Path, typer.Option(help="Destination JSONL or JSONL.GZ dataset path.")
    ] = DATA_DIR / "suppressed-features.jsonl",
    max_error_share: Annotated[
        float,
        typer.Option(
            min=0.0,
            max=1.0,
            help="Reject graphs whose error nodes exceed this influence share.",
        ),
    ] = 0.35,
) -> None:
    """Build same-family cohorts with a verified shared multiplicative corruption."""
    build = start_build(
        candidates=n,
        seed=seed,
        requested_families=family,
        split=split,
        default_split="all",
        modal_app=MODAL_APP,
        modal_cls=MODAL_CLS,
    )
    generator = build.subject
    # Maximal graph JSON is large enough that one bulk Modal map can finish its
    # containers yet strand the client while buffering all results. Keep graph
    # transfers bounded to one result at a time.
    by_family_target: dict[tuple[str, str], list[dict]] = defaultdict(list)
    quality = []
    for attributed in clean_attributions(
        generator,
        build.prepared,
        max_error_share=max_error_share,
        batch_size=32,
        concurrent_calls=True,
    ):
        item = attributed.prepared
        quality.append(attributed.quality)
        by_family_target[(item.sample.family, item.target_token)].append(
            {
                "sample": item.sample,
                "target_token": item.target_token,
                "clean_graph": attributed.graph,
            }
        )

    rng = random.Random(seed)
    group_size = cohort_size + heldout_size
    print(
        "answer sampling triage:",
        build.triage,
        "faithful graph records:",
        {
            f"{family}:{target_token!r}": len(records)
            for (family, target_token), records in sorted(by_family_target.items())
        },
    )
    candidate_groups: list[tuple[str, list[dict]]] = []
    for (candidate_family, _target_token), records in sorted(by_family_target.items()):
        rng.shuffle(records)
        candidate_groups.extend(
            (candidate_family, records[offset : offset + group_size])
            for offset in range(0, len(records) - group_size + 1, group_size)
        )
    rng.shuffle(candidate_groups)

    rows = []
    for cohort_family, records in candidate_groups:
        control_candidates = [
            record
            for (other_family, _), family_records in by_family_target.items()
            if other_family != cohort_family
            for record in family_records
        ]
        if len(control_candidates) < group_size:
            continue
        common = common_stable_features([record["clean_graph"] for record in records])
        print(
            f"testing {cohort_family} cohort of {len(records)} prompts with "
            f"{len(common)} shared stable features"
        )
        if len(common) < MIN_CORRUPTION_NODES:
            continue
        prompts = [record["sample"].prompt for record in records]
        target_tokens = [record["target_token"] for record in records]
        position_maps = [
            stable_feature_positions(record["clean_graph"]) for record in records
        ]
        selected_spec = None
        selected_coordinates = None
        for _ in range(attempts):
            spec = random_corruption_spec(common, rng)
            coordinates = coordinate_specs(spec, position_maps)
            tested = list(
                generator.map(
                    "test_corruption_coordinates",
                    prompts,
                    target_tokens,
                    coordinates,
                    order_outputs=True,
                )
            )
            if all_changed(tested):
                selected_spec = spec
                selected_coordinates = coordinates
                break
        if selected_spec is None or selected_coordinates is None:
            print(
                f"  no shared corruption changed all {len(records)} prompts "
                f"after {attempts} attempts"
            )
            continue

        corrupted_calls = [
            generator.spawn(
                "build_corrupted_sample_from_coordinates",
                prompt,
                target_token,
                coordinates,
            )
            for prompt, target_token, coordinates in zip(
                prompts, target_tokens, selected_coordinates, strict=True
            )
        ]
        corrupted = [call.get() for call in corrupted_calls]
        if not all_changed(corrupted) or any("graph" not in item for item in corrupted):
            continue

        examples: list[DatasetExample] = []
        for record, result in zip(records, corrupted, strict=True):
            if result["target_token"] != record["target_token"]:
                raise RuntimeError(
                    "corruption backend changed the sampled target token"
                )
            examples.append(
                DatasetExample(
                    prompt=record["sample"].prompt,
                    target_token=result["target_token"],
                    graph=TracerGraphPayload.model_validate(result["graph"]),
                    corruption=result["corruption"],
                )
            )
        rows.append(
            CircuitDatasetRow(
                family=cohort_family,
                example_prompts=[],
                heldout_prompts=examples,
                control_prompts=[
                    DatasetExample(
                        prompt=record["sample"].prompt,
                        target_token=record["target_token"],
                        graph=record["clean_graph"],
                    )
                    for record in rng.sample(control_candidates, group_size)
                ],
            )
        )
        if len(rows) == tasks:
            break

    if len(rows) < tasks:
        raise SystemExit(
            f"formed only {len(rows)}/{tasks} corruptible cohorts; increase "
            "--n/--attempts or reduce cohort sizes"
        )
    finish_build(
        out,
        rows,
        build=build,
        modal_app=MODAL_APP,
        modal_cls=MODAL_CLS,
        max_error_share=max_error_share,
        quality=quality,
        corruption_version=CORRUPTION_VERSION,
        generator={
            "seed": seed,
            "candidates": n,
            "cohort_size": cohort_size,
            "heldout_size": heldout_size,
            "corruption_attempts": attempts,
            "corruption_nodes": [MIN_CORRUPTION_NODES, min(12, MAX_CORRUPTION_NODES)],
        },
    )


if __name__ == "__main__":
    typer.run(main)
