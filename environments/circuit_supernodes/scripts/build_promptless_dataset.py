"""Build hidden-prompt goal cohorts with answer-aware subject sampling."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Literal

import typer
from circuit_supernodes.models import (
    CircuitDatasetRow,
    DatasetExample,
)
from circuit_supernodes.runtime.answers import answer_key
from scripts.dataset_pipeline import (
    clean_attributions,
    finish_build,
    start_build,
)

DATA_DIR = (
    Path(__file__).parent.parent
    / "circuit_supernodes"
    / "data"
    / "supernode_rediscovery"
)


def cohort_rows(
    records: list[dict],
    *,
    tasks: int,
    cohort_size: int,
    control_size: int,
    seed: int,
) -> list[CircuitDatasetRow]:
    """Group varied-answer examples by semantic goal and sample other-goal controls.

    Positive cohorts deliberately contain at least two distinct full answers, so
    a proposal must describe the goal mechanism rather than one output token.
    Controls come from different semantic goals; their answer may overlap.
    """

    def record_goal(record: dict) -> str:
        return str(
            record.get("goal")
            or record.get("source_family")
            or record.get("family", "")
        ).strip()

    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        goal = record_goal(record)
        if not goal:
            continue
        grouped[goal].append(record)

    rng = random.Random(seed)
    candidates = [
        (goal, values)
        for goal, values in sorted(grouped.items())
        if len(values) >= cohort_size
        and len({answer_key(str(item["target"])) for item in values}) >= 2
    ]
    rng.shuffle(candidates)
    rows = []
    for goal, positives in candidates:
        by_answer: dict[str, list[dict]] = defaultdict(list)
        for record in positives:
            by_answer[answer_key(str(record["target"]))].append(record)
        answer_groups = list(by_answer.values())
        rng.shuffle(answer_groups)
        for group in answer_groups:
            rng.shuffle(group)
        selected_positive = [group.pop() for group in answer_groups]
        remaining = [record for group in answer_groups for record in group]
        rng.shuffle(remaining)
        selected_positive.extend(remaining[: cohort_size - len(selected_positive)])
        selected_positive = selected_positive[:cohort_size]

        controls = [record for record in records if record_goal(record) != goal]
        if len(controls) < control_size:
            continue
        selected_controls = rng.sample(controls, control_size)
        cohort_families = sorted({str(item["family"]) for item in selected_positive})
        family = cohort_families[0] if len(cohort_families) == 1 else "mixed"
        rows.append(
            CircuitDatasetRow(
                family=family,
                goal=goal,
                example_prompts=[],
                heldout_prompts=[
                    DatasetExample(
                        prompt=item["prompt"],
                        target_token=item["target_token"],
                        graph=item["graph"],
                    )
                    for item in selected_positive
                ],
                control_prompts=[
                    DatasetExample(
                        prompt=item["prompt"],
                        target_token=item["target_token"],
                        graph=item["graph"],
                    )
                    for item in selected_controls
                ],
            )
        )
        if len(rows) == tasks:
            break
    return rows


def main(
    n: Annotated[int, typer.Option(min=1, help="Candidate prompts to sample.")] = 480,
    tasks: Annotated[int, typer.Option(min=1, help="Cohort rows to produce.")] = 16,
    cohort_size: Annotated[
        int, typer.Option(min=1, help="Same-goal positive prompts in each row.")
    ] = 4,
    control_size: Annotated[
        int, typer.Option(min=1, help="Different-goal controls in each row.")
    ] = 4,
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
    ] = DATA_DIR / "promptless-supernodes.jsonl",
    max_error_share: Annotated[
        float,
        typer.Option(
            min=0.0,
            max=1.0,
            help="Reject graphs whose error nodes exceed this influence share.",
        ),
    ] = 0.35,
) -> None:
    """Build hidden-prompt goal-supernode cohorts and specificity controls."""
    build = start_build(
        candidates=n,
        seed=seed,
        requested_families=family,
        split=split,
        default_split="all",
    )
    records = []
    quality = []
    rows = []
    for attributed in clean_attributions(
        build.subject,
        build.prepared,
        max_error_share=max_error_share,
        batch_size=32,
        concurrent_calls=True,
    ):
        item = attributed.prepared
        quality.append(attributed.quality)
        records.append(
            {
                **item.sample.to_row(),
                "target_token": item.target_token,
                "graph": attributed.graph,
            }
        )
        rows = cohort_rows(
            records,
            tasks=tasks,
            cohort_size=cohort_size,
            control_size=control_size,
            seed=seed,
        )
        if len(rows) >= tasks:
            break
    print(
        "answer sampling triage:",
        build.triage,
        "faithful graph records:",
        len(records),
    )
    if len(rows) < tasks:
        raise SystemExit(
            f"formed only {len(rows)}/{tasks} cohorts; increase --n or "
            "reduce cohort/control sizes"
        )
    finish_build(
        out,
        rows,
        build=build,
        max_error_share=max_error_share,
        quality=quality,
        generator={
            "seed": seed,
            "candidates": n,
            "cohort_size": cohort_size,
            "control_size": control_size,
            "cohort_grouping": "goal with at least two distinct answers",
            "control_exclusion": "different semantic goal",
        },
    )


if __name__ == "__main__":
    typer.run(main)
