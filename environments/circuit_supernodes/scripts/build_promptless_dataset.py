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
from circuit_supernodes.tasksets.rediscovery import DEFAULT_MODAL_APP, DEFAULT_MODAL_CLS
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
    """Group examples by semantic goal and answer; sample safe controls.

    A shared answer token alone is not evidence of a shared mechanism: ``Yes``
    from comparison and ``Yes`` from another family can have unrelated causal
    features.  The goal key keeps positive cohorts within one generated task
    mechanism. Controls must differ in both full answer and attributed first
    token so the specificity gate is not impossible before any intervention.
    """

    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for record in records:
        goal = str(
            record.get("goal")
            or record.get("source_family")
            or record.get("family", "")
        ).strip()
        if not goal:
            continue
        key = (
            goal,
            answer_key(str(record["target"])),
            str(record["target_token"]),
        )
        grouped[key].append(record)

    rng = random.Random(seed)
    candidates = [
        (key, values)
        for key, values in sorted(grouped.items())
        if len(values) >= cohort_size
    ]
    rng.shuffle(candidates)
    rows = []
    for (goal, _, target_token), positives in candidates:
        desired_answer = str(positives[0]["target"]).strip()
        controls = [
            record
            for record in records
            if answer_key(str(record["target"])) != answer_key(desired_answer)
            and str(record.get("target_token", "")) != target_token
        ]
        if len(controls) < control_size:
            continue
        selected_positive = rng.sample(positives, cohort_size)
        selected_controls = rng.sample(controls, control_size)
        cohort_families = sorted({str(item["family"]) for item in selected_positive})
        family = cohort_families[0] if len(cohort_families) == 1 else "mixed"
        rows.append(
            CircuitDatasetRow(
                family=family,
                goal=goal,
                desired_answer=desired_answer,
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
        int, typer.Option(min=1, help="Different-answer controls in each row.")
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
        modal_app=DEFAULT_MODAL_APP,
        modal_cls=DEFAULT_MODAL_CLS,
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
        modal_app=DEFAULT_MODAL_APP,
        modal_cls=DEFAULT_MODAL_CLS,
        max_error_share=max_error_share,
        quality=quality,
        generator={
            "seed": seed,
            "candidates": n,
            "cohort_size": cohort_size,
            "control_size": control_size,
            "cohort_grouping": "goal+answer+target_token",
            "control_exclusion": "desired_answer+desired_target_token",
        },
    )


if __name__ == "__main__":
    typer.run(main)
