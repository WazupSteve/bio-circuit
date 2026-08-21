import random
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Literal

import typer
from circuit_supernodes.models import (
    CircuitDatasetRow,
    DatasetExample,
)
from modal_app.attribution import (
    EDGE_THRESHOLD,
    MODEL_NAME,
    NODE_THRESHOLD,
    TRANSCODER_SET,
)
from scripts.dataset_pipeline import (
    clean_attributions,
    finish_build,
    start_build,
)
from scripts.neuronpedia import add_feature_labels

DATA_DIR = (
    Path(__file__).parent.parent / "circuit_supernodes" / "data" / "visible_dataset"
)
MODAL_APP = "circuit-supernodes-attribution"
MODAL_CLS = "Attributor"
HELDOUT_POSITIVE_COUNT = 25
NEUTRAL_COUNT = 25


def cohort_rows(
    records: list[dict],
    *,
    tasks: int,
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
    eligible = [
        goal for goal, examples in grouped.items() if len(examples) >= holdout_size
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
            if offset + holdout_size > len(positives):
                continue
            null_candidates = [
                record
                for other_goal, examples in grouped.items()
                if other_goal != goal
                for record in examples
            ]
            if len(null_candidates) < null_size:
                continue
            holdout = positives[offset : offset + holdout_size]
            holdout_offsets[goal] += holdout_size
            nulls = rng.sample(null_candidates, null_size)
            rows.append(
                CircuitDatasetRow(
                    family=holdout[0]["family"],
                    goal=goal,
                    example_prompts=[],
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
    requested_sizes = (holdout_size, null_size)
    required_sizes = (HELDOUT_POSITIVE_COUNT, NEUTRAL_COUNT)
    if requested_sizes != required_sizes:
        raise typer.BadParameter(
            "the circuit-supernodes contract requires exactly "
            f"{HELDOUT_POSITIVE_COUNT} hidden positive and {NEUTRAL_COUNT} "
            "control examples per rollout",
            param_hint="--holdout-size/--null-size",
        )
    if max_records_per_goal < holdout_size:
        raise typer.BadParameter(
            "must be at least --holdout-size", param_hint="--max-records-per-goal"
        )

    build = start_build(
        candidates=n,
        seed=seed,
        requested_families=family,
        split=split,
        default_split="train",
        modal_app=MODAL_APP,
        modal_cls=MODAL_CLS,
    )
    records, quality = [], []
    retained_per_goal: dict[str, int] = defaultdict(int)
    rows = []
    for attributed in clean_attributions(
        build.subject,
        build.prepared,
        max_error_share=max_error_share,
        batch_size=64,
        concurrent_calls=False,
    ):
        item = attributed.prepared
        s = item.sample
        graph = attributed.graph
        quality.append(attributed.quality)
        if neuronpedia_labels:
            try:
                labeled = add_feature_labels(graph)
                print(f"  + added {labeled} Neuronpedia feature labels")
            except Exception as exc:
                print(f"  ! Neuronpedia labels unavailable: {exc}")
        record = s.to_row()
        goal = str(record.get("goal", "")).strip()
        if retained_per_goal[goal] >= max_records_per_goal:
            continue
        record["target_token"] = item.target_token
        record["graph"] = graph
        records.append(record)
        retained_per_goal[goal] += 1
        rows = cohort_rows(
            records,
            tasks=tasks,
            holdout_size=holdout_size,
            null_size=null_size,
            seed=seed,
        )
        if len(rows) >= tasks:
            break
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
            f"formed only {len(rows)}/{tasks} cohorts; increase --n or reduce "
            "holdout/null sizes"
        )
    finish_build(
        out,
        rows,
        build=build,
        modal_app=MODAL_APP,
        modal_cls=MODAL_CLS,
        max_error_share=max_error_share,
        quality=quality,
        generator={
            "seed": seed,
            "candidates": n,
            "tasks": tasks,
            "example_prompt_count": 0,
            "holdout_size": holdout_size,
            "null_size": null_size,
            "max_records_per_goal": max_records_per_goal,
            "positive_grouping": "shared semantic goal",
            "null_exclusion": "different semantic goal",
        },
        backend_details={
            "subject_model": MODEL_NAME,
            "transcoders": TRANSCODER_SET,
            "node_threshold": NODE_THRESHOLD,
            "edge_threshold": EDGE_THRESHOLD,
        },
    )


if __name__ == "__main__":
    typer.run(main)
