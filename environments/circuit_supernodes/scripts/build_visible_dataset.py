import random
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Literal

import typer
from circuit_supernodes.models import (
    CircuitDatasetRow,
    DatasetExample,
)
from circuit_supernodes.runtime.modal import SUBJECT
from scripts.dataset_pipeline import (
    DatasetBuild,
    clean_attributions,
    finish_build,
    resolve_families,
)
from scripts.neuronpedia import add_feature_labels
from scripts.prompt_generation import generate_pool
from scripts.sample_preparation import PreparedSample, prepare_samples

DATA_DIR = (
    Path(__file__).parent.parent / "circuit_supernodes" / "data" / "visible_dataset"
)
EXAMPLE_COUNT = 5
HELDOUT_POSITIVE_COUNT = 25
NEUTRAL_COUNT = 25


def sample_goal(sample: object) -> str:
    """Return the normalized cohort goal for a generated or prepared sample."""

    source = sample.sample if isinstance(sample, PreparedSample) else sample
    return str(source.to_row().get("goal", "")).strip()


def cohort_capacity(
    counts: dict[str, int], *, positive_size: int, null_size: int
) -> int:
    """Count constructible cohorts without allocating graph-backed rows."""

    total = sum(counts.values())
    return sum(
        count // positive_size
        for count in counts.values()
        if total - count >= null_size
    )


def _triage_total(target: dict[str, int], batch: dict[str, int]) -> None:
    for disposition, count in batch.items():
        target[disposition] += count


def _attribution_batch(
    queues: dict[str, list[PreparedSample]],
    retained_per_goal: dict[str, int],
    *,
    max_records_per_goal: int,
    batch_size: int,
) -> list[PreparedSample]:
    """Select useful work fairly, reserving no more than each goal can retain."""

    selected: list[PreparedSample] = []
    reserved: dict[str, int] = defaultdict(int)
    goals = list(queues)
    while len(selected) < batch_size:
        progress = False
        for goal in goals:
            queue = queues[goal]
            available = max_records_per_goal - retained_per_goal[goal] - reserved[goal]
            if not queue or available <= 0:
                continue
            selected.append(queue.pop())
            reserved[goal] += 1
            progress = True
            if len(selected) == batch_size:
                break
        if not progress:
            break
    return selected


def cohort_rows(
    records: list[dict],
    *,
    tasks: int,
    example_size: int,
    holdout_size: int,
    null_size: int,
    seed: int,
) -> list[CircuitDatasetRow]:
    """Form cached same-goal heldout and different-goal control cohorts."""

    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        goal = str(record.get("goal", "")).strip()
        if goal:
            grouped[goal].append(record)
    positive_size = example_size + holdout_size
    eligible = [
        goal for goal, examples in grouped.items() if len(examples) >= positive_size
    ]
    rng = random.Random(seed)
    rng.shuffle(eligible)
    holdout_offsets = {goal: 0 for goal in eligible}
    rows = []
    while len(rows) < tasks:
        progress = False
        for goal in eligible:
            positives = grouped[goal]
            offset = holdout_offsets[goal]
            if offset + positive_size > len(positives):
                continue
            null_candidates = [
                record
                for other_goal, examples in grouped.items()
                if other_goal != goal
                for record in examples
            ]
            if len(null_candidates) < null_size:
                continue
            selected = positives[offset : offset + positive_size]
            examples = selected[:example_size]
            holdout = selected[example_size:]
            holdout_offsets[goal] += positive_size
            nulls = rng.sample(null_candidates, null_size)
            rows.append(
                CircuitDatasetRow(
                    family=holdout[0]["family"],
                    goal=goal,
                    example_prompts=[
                        DatasetExample(
                            prompt=item["prompt"],
                            target_token=item["target_token"],
                            graph=item["graph"],
                        )
                        for item in examples
                    ],
                    heldout_prompts=[
                        DatasetExample(
                            prompt=item["prompt"],
                            target_token=item["target_token"],
                            graph=item["graph"],
                        )
                        for item in holdout
                    ],
                    control_prompts=[
                        DatasetExample(
                            prompt=item["prompt"],
                            target_token=item["target_token"],
                            graph=item["graph"],
                        )
                        for item in nulls
                    ],
                )
            )
            progress = True
            if len(rows) == tasks:
                break
        if not progress:
            break
    return rows


def main(
    n: Annotated[
        int,
        typer.Option(min=1, help="Number of candidate prompts to sample."),
    ] = 2400,
    tasks: Annotated[
        int,
        typer.Option(min=1, help="Number of cohort rows to produce."),
    ] = 16,
    example_size: Annotated[
        int,
        typer.Option(min=1, help="Visible same-goal examples in each row."),
    ] = EXAMPLE_COUNT,
    holdout_size: Annotated[
        int,
        typer.Option(min=1, help="Hidden same-goal positives in each row."),
    ] = HELDOUT_POSITIVE_COUNT,
    null_size: Annotated[
        int,
        typer.Option(min=1, help="Different-goal controls in each row."),
    ] = NEUTRAL_COUNT,
    max_records_per_goal: Annotated[
        int,
        typer.Option(
            min=1,
            help="Maximum accepted graphs retained per semantic goal to bound RAM.",
        ),
    ] = 40,
    answer_batch_size: Annotated[
        int,
        typer.Option(
            min=1,
            help="Candidate answers to sample before checking cohort capacity.",
        ),
    ] = 256,
    attribution_batch_size: Annotated[
        int,
        typer.Option(min=1, help="Useful attribution calls dispatched per batch."),
    ] = 64,
    seed: Annotated[int, typer.Option(help="Deterministic sampling seed.")] = 0,
    family: Annotated[
        list[str] | None,
        typer.Option(
            "--family",
            "--families",
            help="Prompt family to include; repeat for multiple families.",
        ),
    ] = None,
    split: Annotated[
        Literal["train", "eval"] | None,
        typer.Option(help="Use the predefined training or sealed evaluation families."),
    ] = None,
    out: Annotated[
        Path,
        typer.Option(help="Destination JSONL or JSONL.GZ dataset path."),
    ] = DATA_DIR / "circuit-cohorts.jsonl",
    max_error_share: Annotated[
        float,
        typer.Option(
            min=0.0,
            max=1.0,
            help="Reject graphs whose error nodes exceed this influence share.",
        ),
    ] = 0.35,
    neuronpedia_labels: Annotated[
        bool,
        typer.Option(
            help="Label the top-ranked 60 features with one Neuronpedia batch request."
        ),
    ] = False,
) -> None:
    """Build visible-circuit cohorts with hidden same-goal and control caches."""
    requested_sizes = (example_size, holdout_size, null_size)
    required_sizes = (EXAMPLE_COUNT, HELDOUT_POSITIVE_COUNT, NEUTRAL_COUNT)
    if requested_sizes != required_sizes:
        raise typer.BadParameter(
            "the circuit-supernodes contract requires exactly "
            f"{EXAMPLE_COUNT} visible, {HELDOUT_POSITIVE_COUNT} hidden positive, "
            f"and {NEUTRAL_COUNT} control examples per rollout",
            param_hint="--example-size/--holdout-size/--null-size",
        )
    if max_records_per_goal < example_size + holdout_size:
        raise typer.BadParameter(
            "must cover --example-size plus --holdout-size",
            param_hint="--max-records-per-goal",
        )

    families = resolve_families(family, split, default_split="train")
    samples = generate_pool(n, seed=seed, families=families)
    print(
        f"sampled {len(samples)} candidate prompts across families:",
        sorted({sample.family for sample in samples}),
    )
    candidate_counts: dict[str, int] = defaultdict(int)
    for sample in samples:
        candidate_counts[sample_goal(sample)] += 1
    positive_size = example_size + holdout_size
    bounded_candidate_counts = {
        goal: min(count, max_records_per_goal)
        for goal, count in candidate_counts.items()
    }
    maximum_tasks = cohort_capacity(
        bounded_candidate_counts,
        positive_size=positive_size,
        null_size=null_size,
    )
    if len(candidate_counts) < 2 or maximum_tasks < tasks:
        raise typer.BadParameter(
            f"configuration can form at most {maximum_tasks} cohorts from "
            f"{len(candidate_counts)} goals; increase --max-records-per-goal, "
            "select more goals, or reduce --tasks",
            param_hint="--tasks/--max-records-per-goal",
        )

    records_by_goal: dict[str, list[dict]] = defaultdict(list)
    quality = []
    retained_per_goal: dict[str, int] = defaultdict(int)
    triage = {"keep_correct": 0, "keep_incorrect": 0, "reject": 0}
    sampled_candidates = 0
    attributed_candidates = 0

    for start in range(0, len(samples), answer_batch_size):
        candidates = [
            sample
            for sample in samples[start : start + answer_batch_size]
            if retained_per_goal[sample_goal(sample)] < max_records_per_goal
        ]
        if not candidates:
            continue
        prepared, batch_triage = prepare_samples(SUBJECT, candidates)
        sampled_candidates += len(candidates)
        _triage_total(triage, batch_triage)
        queues: dict[str, list[PreparedSample]] = defaultdict(list)
        for item in reversed(prepared):
            queues[sample_goal(item)].append(item)

        while batch := _attribution_batch(
            queues,
            retained_per_goal,
            max_records_per_goal=max_records_per_goal,
            batch_size=attribution_batch_size,
        ):
            attributed_candidates += len(batch)
            for attributed in clean_attributions(
                SUBJECT,
                batch,
                max_error_share=max_error_share,
                batch_size=attribution_batch_size,
                concurrent_calls=False,
            ):
                item = attributed.prepared
                goal = sample_goal(item)
                if retained_per_goal[goal] >= max_records_per_goal:
                    continue
                graph = attributed.graph
                quality.append(attributed.quality)
                if neuronpedia_labels:
                    try:
                        labeled = add_feature_labels(graph)
                        print(f"  + added {labeled} Neuronpedia feature labels")
                    except Exception as exc:
                        print(f"  ! Neuronpedia labels unavailable: {exc}")
                record = item.sample.to_row()
                record["target_token"] = item.target_token
                record["graph"] = graph
                records_by_goal[goal].append(record)
                retained_per_goal[goal] += 1

            capacity = cohort_capacity(
                dict(retained_per_goal),
                positive_size=positive_size,
                null_size=null_size,
            )
            counts = ", ".join(
                f"{goal}={count}" for goal, count in sorted(retained_per_goal.items())
            )
            print(
                f"retained {sum(retained_per_goal.values())} graphs; "
                f"cohort capacity {capacity}/{tasks}; {counts}"
            )
            if capacity >= tasks:
                break
        if (
            cohort_capacity(
                dict(retained_per_goal),
                positive_size=positive_size,
                null_size=null_size,
            )
            >= tasks
        ):
            break

    records = [
        record for goal_records in records_by_goal.values() for record in goal_records
    ]
    rows = cohort_rows(
        records,
        tasks=tasks,
        example_size=example_size,
        holdout_size=holdout_size,
        null_size=null_size,
        seed=seed,
    )
    if quality:
        print(
            f"graph quality: error influence "
            f"{sum(q['error'] for q in quality) / len(quality):.1%}, "
            f"top-60 features cover "
            f"{sum(q['displayed'] for q in quality) / len(quality):.1%}"
            f" of feature influence"
        )
    if len(rows) < tasks:
        raise SystemExit(
            f"formed only {len(rows)}/{tasks} cohorts after sampling "
            f"{sampled_candidates}/{n} candidates and attributing "
            f"{attributed_candidates}; retained by goal: "
            f"{dict(sorted(retained_per_goal.items()))}; increase --n"
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
            "tasks": tasks,
            "example_prompt_count": example_size,
            "holdout_size": holdout_size,
            "null_size": null_size,
            "max_records_per_goal": max_records_per_goal,
            "answer_batch_size": answer_batch_size,
            "attribution_batch_size": attribution_batch_size,
            "positive_grouping": "shared semantic goal",
            "null_exclusion": "different semantic goal",
        },
    )


if __name__ == "__main__":
    typer.run(main)
