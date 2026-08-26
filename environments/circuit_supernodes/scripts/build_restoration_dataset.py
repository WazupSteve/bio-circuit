"""Build common-corruption discovery and held-out prompt cohorts."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Literal

import typer
from circuit_supernodes.models import (
    CircuitDatasetRow,
    DatasetExample,
    TracerGraphPayload,
)
from circuit_supernodes.runtime.corruption import (
    MAX_CORRUPTION_NODES,
    MIN_CORRUPTION_NODES,
)
from circuit_supernodes.runtime.graphs import (
    MAX_FEATURE_NODES,
    feature_nodes,
    influence_deltas,
    stable_feature_id,
)
from circuit_supernodes.runtime.modal import SUBJECT, Subject
from scripts.dataset_pipeline import (
    DatasetBuild,
    clean_attributions,
    finish_build,
    resolve_families,
)
from scripts.prompt_generation import iter_pool_batches
from scripts.sample_preparation import PreparedSample, prepare_samples

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
        factor = round(rng.uniform(-8.0, 8.0), 6)
        corruption.append((layer, feature, factor))
    return corruption


def all_changed(results: list[dict]) -> bool:
    return bool(results) and all(
        isinstance(result, dict)
        and not result.get("rejected")
        and result.get("changed_top_token") is True
        for result in results
    )


def _test_candidate_group(
    generator: Subject,
    cohort_family: str,
    records: list[dict],
    control_candidates: list[dict],
    *,
    heldout_size: int,
    attempts: int,
    rng: random.Random,
) -> CircuitDatasetRow | None:
    common = common_stable_features([record["clean_graph"] for record in records])
    print(
        f"testing {cohort_family} cohort of {len(records)} prompts with "
        f"{len(common)} shared stable features"
    )
    if len(common) < MIN_CORRUPTION_NODES:
        return None

    prompts = [record["sample"].prompt for record in records]
    target_tokens = [record["target_token"] for record in records]
    position_maps = [
        stable_feature_positions(record["clean_graph"]) for record in records
    ]
    corrupted: list[dict] | None = None
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
            corrupted = tested
            break
    if corrupted is None:
        print(
            f"  no shared corruption changed all {len(records)} prompts "
            f"after {attempts} attempts"
        )
        return None

    examples: list[DatasetExample] = []
    for record, result in zip(records, corrupted, strict=True):
        if result["target_token"] != record["target_token"]:
            raise RuntimeError("corruption backend changed the sampled target token")
        examples.append(
            DatasetExample(
                prompt=record["sample"].prompt,
                target_token=result["target_token"],
                graph=record["clean_graph"],
                corruption=result["corruption"],
            )
        )
    return CircuitDatasetRow(
        family=cohort_family,
        example_prompts=[],
        heldout_prompts=examples,
        control_prompts=[
            DatasetExample(
                prompt=record["sample"].prompt,
                target_token=record["target_token"],
                graph=record["clean_graph"],
            )
            for record in rng.sample(control_candidates, heldout_size)
        ],
    )


def main(
    n: Annotated[int, typer.Option(min=1, help="Maximum candidate prompts.")] = 480,
    tasks: Annotated[
        int, typer.Option(min=1, help="Corrupted cohort rows to produce.")
    ] = 16,
    heldout_size: Annotated[
        int,
        typer.Option(min=1, help="Hidden corrupted positive prompts in each row."),
    ] = 10,
    attempts: Annotated[
        int,
        typer.Option(min=1, help="Shared-corruption attempts per candidate cohort."),
    ] = 32,
    answer_batch_size: Annotated[
        int, typer.Option(min=1, help="Answers sampled before testing ready cohorts.")
    ] = 128,
    attribution_batch_size: Annotated[
        int, typer.Option(min=1, help="Attribution calls dispatched per batch.")
    ] = 8,
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

    families = resolve_families(family, split, default_split="all")
    if len(set(families)) < 2:
        raise typer.BadParameter(
            "restoration cohorts require at least two families for controls",
            param_hint="--family/--split",
        )
    if n < tasks * heldout_size:
        raise typer.BadParameter(
            f"{tasks} disjoint cohorts require at least "
            f"{tasks * heldout_size} candidates before screening; increase --n",
            param_hint="--n/--tasks/--heldout-size",
        )
    print(
        f"sampling up to {n} candidate prompts across families:",
        sorted(set(families)),
    )

    rng = random.Random(seed)
    by_family_target: dict[tuple[str, str], list[dict]] = defaultdict(list)
    controls_by_family: dict[str, list[dict]] = defaultdict(list)
    quality: list[dict[str, float | int]] = []
    rows: list[CircuitDatasetRow] = []
    triage = {"keep_correct": 0, "keep_incorrect": 0, "reject": 0}
    sampled_candidates = 0
    attributed_candidates = 0
    tested_groups = 0

    for candidates in iter_pool_batches(n, seed, families, answer_batch_size):
        prepared, batch_triage = prepare_samples(SUBJECT, candidates)
        sampled_candidates += len(candidates)
        for disposition, count in batch_triage.items():
            triage[disposition] += count

        prepared_by_family: dict[str, list[PreparedSample]] = defaultdict(list)
        for item in reversed(prepared):
            prepared_by_family[item.sample.family].append(item)
        family_order = list(prepared_by_family)

        while any(prepared_by_family.values()) and len(rows) < tasks:
            batch: list[PreparedSample] = []
            while len(batch) < attribution_batch_size:
                progress = False
                for candidate_family in family_order:
                    if prepared_by_family[candidate_family]:
                        batch.append(prepared_by_family[candidate_family].pop())
                        progress = True
                        if len(batch) == attribution_batch_size:
                            break
                if not progress:
                    break
            if not batch:
                break

            attributed_candidates += len(batch)
            for attributed in clean_attributions(
                SUBJECT,
                batch,
                max_error_share=max_error_share,
                batch_size=attribution_batch_size,
                concurrent_calls=True,
            ):
                item = attributed.prepared
                record = {
                    "sample": item.sample,
                    "target_token": item.target_token,
                    "clean_graph": attributed.graph,
                }
                quality.append(attributed.quality)
                key = (item.sample.family, item.target_token)
                by_family_target[key].append(record)
                if len(controls_by_family[item.sample.family]) < heldout_size:
                    controls_by_family[item.sample.family].append(record)

            test_progress = True
            while test_progress and len(rows) < tasks:
                test_progress = False
                for (cohort_family, _), group_records in sorted(
                    by_family_target.items()
                ):
                    if len(group_records) < heldout_size:
                        continue
                    control_candidates = [
                        record
                        for other_family, family_records in controls_by_family.items()
                        if other_family != cohort_family
                        for record in family_records
                    ]
                    if len(control_candidates) < heldout_size:
                        continue
                    records = group_records[:heldout_size]
                    del group_records[:heldout_size]
                    tested_groups += 1
                    test_progress = True
                    row = _test_candidate_group(
                        SUBJECT,
                        cohort_family,
                        records,
                        control_candidates,
                        heldout_size=heldout_size,
                        attempts=attempts,
                        rng=rng,
                    )
                    if row is not None:
                        rows.append(row)
                        print(f"formed {len(rows)}/{tasks} corruptible cohorts")
                        if len(rows) == tasks:
                            break

            group_counts = {
                f"{family}:{target!r}": len(records)
                for (family, target), records in sorted(by_family_target.items())
                if records
            }
            print(
                f"retained {sum(group_counts.values())} untested graphs; "
                f"tested {tested_groups} groups; formed {len(rows)}/{tasks}; "
                f"groups={group_counts}"
            )

        if len(rows) == tasks:
            break

    if len(rows) < tasks:
        raise SystemExit(
            f"formed only {len(rows)}/{tasks} corruptible cohorts after sampling "
            f"{sampled_candidates}/{n} candidates, attributing "
            f"{attributed_candidates}, and testing {tested_groups} groups; "
            "increase --n/--attempts"
        )

    build = DatasetBuild(families, SUBJECT, [], triage)
    finish_build(
        out,
        rows,
        build=build,
        max_error_share=max_error_share,
        quality=quality,
        generator={
            "seed": seed,
            "candidates": n,
            "candidates_sampled": sampled_candidates,
            "candidates_attributed": attributed_candidates,
            "candidate_groups_tested": tested_groups,
            "tasks": tasks,
            "heldout_size": heldout_size,
            "corruption_attempts": attempts,
            "answer_batch_size": answer_batch_size,
            "attribution_batch_size": attribution_batch_size,
            "corruption_nodes": [MIN_CORRUPTION_NODES, min(12, MAX_CORRUPTION_NODES)],
        },
    )


if __name__ == "__main__":
    typer.run(main)
