"""Build hidden-prompt goal cohorts with answer-aware subject sampling."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Annotated, Literal

import typer
from circuit_supernodes.models import CircuitDatasetRow, DatasetExample
from circuit_supernodes.runtime.answers import answer_key
from circuit_supernodes.runtime.modal import SUBJECT
from scripts.dataset_pipeline import (
    DatasetBuild,
    clean_attributions,
    finish_build,
    resolve_families,
)
from scripts.prompt_generation import generate_pool
from scripts.sample_preparation import PreparedSample, prepare_samples

DATA_DIR = (
    Path(__file__).parent.parent
    / "circuit_supernodes"
    / "data"
    / "supernode_rediscovery"
)


def record_goal(record: dict) -> str:
    return str(
        record.get("goal") or record.get("source_family") or record.get("family", "")
    ).strip()


def sample_goal(item: object) -> str:
    sample = item.sample if isinstance(item, PreparedSample) else item
    return record_goal(sample.to_row())


def diverse_cohort_count(records: list[dict], cohort_size: int) -> int:
    """Maximum disjoint cohorts containing at least two distinct answers."""

    if cohort_size < 2:
        return 0
    counts = Counter(answer_key(str(record["target"])) for record in records)
    total = sum(counts.values())
    largest = max(counts.values(), default=0)
    return min(total // cohort_size, total - largest)


def cohort_capacity(
    records_by_goal: dict[str, list[dict]], *, cohort_size: int, control_size: int
) -> int:
    total = sum(len(records) for records in records_by_goal.values())
    return sum(
        diverse_cohort_count(records, cohort_size)
        for records in records_by_goal.values()
        if total - len(records) >= control_size
    )


def _select_diverse(
    records: list[dict], cohort_size: int, rng: random.Random
) -> list[dict]:
    by_answer: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_answer[answer_key(str(record["target"]))].append(record)
    groups = [group for group in by_answer.values() if group]
    if len(groups) < 2 or len(records) < cohort_size:
        return []
    for group in groups:
        rng.shuffle(group)
    groups.sort(key=len, reverse=True)
    selected = [groups[0].pop(), groups[1].pop()]
    remaining = [record for group in groups for record in group]
    rng.shuffle(remaining)
    selected.extend(remaining[: cohort_size - 2])
    selected_ids = {id(record) for record in selected}
    records[:] = [record for record in records if id(record) not in selected_ids]
    return selected


def cohort_rows(
    records: list[dict],
    *,
    tasks: int,
    cohort_size: int,
    control_size: int,
    seed: int,
) -> list[CircuitDatasetRow]:
    """Build disjoint varied-answer cohorts with reusable other-goal controls."""

    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        goal = record_goal(record)
        if goal:
            grouped[goal].append(record)

    rng = random.Random(seed)
    goals = sorted(grouped)
    rng.shuffle(goals)
    rows: list[CircuitDatasetRow] = []
    while len(rows) < tasks:
        progress = False
        for goal in goals:
            positives = _select_diverse(grouped[goal], cohort_size, rng)
            if not positives:
                continue
            controls = [record for record in records if record_goal(record) != goal]
            if len(controls) < control_size:
                grouped[goal].extend(positives)
                continue
            selected_controls = rng.sample(controls, control_size)
            cohort_families = sorted({str(item["family"]) for item in positives})
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
                        for item in positives
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
            progress = True
            if len(rows) == tasks:
                break
        if not progress:
            break
    return rows


def main(
    n: Annotated[int, typer.Option(min=1, help="Maximum candidate prompts.")] = 480,
    tasks: Annotated[int, typer.Option(min=1, help="Cohort rows to produce.")] = 16,
    cohort_size: Annotated[
        int, typer.Option(min=2, help="Same-goal positive prompts in each row.")
    ] = 4,
    control_size: Annotated[
        int, typer.Option(min=1, help="Different-goal controls in each row.")
    ] = 4,
    answer_batch_size: Annotated[
        int, typer.Option(min=1, help="Answers sampled before capacity is rechecked.")
    ] = 128,
    attribution_batch_size: Annotated[
        int, typer.Option(min=1, help="Attribution calls dispatched per batch.")
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

    families = resolve_families(family, split, default_split="all")
    samples = generate_pool(n, seed=seed, families=families)
    print(
        f"sampled {len(samples)} candidate prompts across families:",
        sorted({sample.family for sample in samples}),
    )
    raw_by_goal: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        raw_by_goal[sample_goal(sample)].append({"target": sample.target})
    maximum_tasks = cohort_capacity(
        raw_by_goal, cohort_size=cohort_size, control_size=control_size
    )
    if maximum_tasks < tasks:
        raise typer.BadParameter(
            f"candidate pool can form at most {maximum_tasks} varied-answer cohorts; "
            "increase --n, select more goals, or reduce --tasks",
            param_hint="--n/--tasks",
        )

    records_by_goal: dict[str, list[dict]] = defaultdict(list)
    quality: list[dict[str, float | int]] = []
    triage = {"keep_correct": 0, "keep_incorrect": 0, "reject": 0}
    sampled_candidates = 0
    attributed_candidates = 0

    for start in range(0, len(samples), answer_batch_size):
        candidates = samples[start : start + answer_batch_size]
        prepared, batch_triage = prepare_samples(SUBJECT, candidates)
        sampled_candidates += len(candidates)
        for disposition, count in batch_triage.items():
            triage[disposition] += count

        prepared_by_goal: dict[str, list[PreparedSample]] = defaultdict(list)
        for item in reversed(prepared):
            prepared_by_goal[sample_goal(item)].append(item)
        goals = list(prepared_by_goal)
        while any(prepared_by_goal.values()):
            batch: list[PreparedSample] = []
            while len(batch) < attribution_batch_size:
                progress = False
                for goal in goals:
                    if prepared_by_goal[goal]:
                        batch.append(prepared_by_goal[goal].pop())
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
                quality.append(attributed.quality)
                records_by_goal[sample_goal(item)].append(
                    {
                        **item.sample.to_row(),
                        "target_token": item.target_token,
                        "graph": attributed.graph,
                    }
                )

            capacity = cohort_capacity(
                records_by_goal,
                cohort_size=cohort_size,
                control_size=control_size,
            )
            counts = ", ".join(
                f"{goal}={len(records)}"
                for goal, records in sorted(records_by_goal.items())
            )
            print(
                f"retained {sum(len(v) for v in records_by_goal.values())} graphs; "
                f"cohort capacity {capacity}/{tasks}; {counts}"
            )
            if capacity >= tasks:
                break
        if (
            cohort_capacity(
                records_by_goal,
                cohort_size=cohort_size,
                control_size=control_size,
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
        cohort_size=cohort_size,
        control_size=control_size,
        seed=seed,
    )
    if len(rows) < tasks:
        retained_counts = {
            goal: len(values) for goal, values in sorted(records_by_goal.items())
        }
        raise SystemExit(
            f"formed only {len(rows)}/{tasks} cohorts after sampling "
            f"{sampled_candidates}/{n} candidates and attributing "
            f"{attributed_candidates}; retained by goal: {retained_counts}"
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
            "cohort_size": cohort_size,
            "control_size": control_size,
            "answer_batch_size": answer_batch_size,
            "attribution_batch_size": attribution_batch_size,
            "cohort_grouping": "disjoint goal cohorts with at least two answers",
            "control_exclusion": "different semantic goal",
        },
    )


if __name__ == "__main__":
    typer.run(main)
