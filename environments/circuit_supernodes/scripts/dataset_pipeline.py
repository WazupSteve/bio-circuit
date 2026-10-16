"""Shared clean-attribution and provenance pipeline for dataset builders."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import typer
from circuit_supernodes.grading.ablation import (
    ablation_payload,
    validate_ablation_result,
)
from circuit_supernodes.models import (
    CircuitDatasetRow,
    DatasetExample,
    TracerGraphPayload,
)
from circuit_supernodes.runtime.datasets import (
    DATASET_SCHEMA_VERSION,
    GENERATION_POLICY,
)
from circuit_supernodes.runtime.graphs import MAX_FEATURE_NODES, graph_quality
from circuit_supernodes.runtime.modal import SUBJECT, Subject
from circuit_supernodes.runtime.sampling import MAX_NEW_TOKENS
from pydantic import ValidationError
from scripts.build_support import DatasetArtifacts
from scripts.prompt_generation import (
    EVAL_FAMILIES,
    FAMILIES,
    TRAIN_FAMILIES,
    generate_pool,
)
from scripts.sample_preparation import PreparedSample, is_faithful, prepare_samples

FamilySplit = Literal["train", "eval", "all"]


@dataclass(frozen=True)
class CleanAttribution:
    prepared: PreparedSample
    graph: TracerGraphPayload
    quality: dict[str, float | int]


@dataclass(frozen=True)
class DatasetBuild:
    """Shared initialized state for a dataset-builder command."""

    families: list[str]
    subject: Subject
    prepared: list[PreparedSample]
    triage: dict[str, int]


def start_build(
    *,
    candidates: int,
    seed: int,
    requested_families: list[str] | None,
    split: FamilySplit | None,
    default_split: FamilySplit,
) -> DatasetBuild:
    """Resolve families and perform the common sampling/preparation phase."""

    families = resolve_families(requested_families, split, default_split=default_split)
    samples = generate_pool(candidates, seed=seed, families=families)
    print(
        f"sampled {len(samples)} prompts across families:",
        sorted({sample.family for sample in samples}),
    )
    subject = SUBJECT
    prepared, triage = prepare_candidates(subject, samples)
    return DatasetBuild(families, subject, prepared, triage)


def mix_family_rows(
    rows: Sequence[CircuitDatasetRow], *, seed: int
) -> list[CircuitDatasetRow]:
    """Return a reproducibly shuffled mixture of completed family rows."""

    mixed = list(rows)
    random.Random(seed).shuffle(mixed)
    return mixed


def finish_build(
    out: Path,
    rows: Sequence[CircuitDatasetRow],
    *,
    build: DatasetBuild,
    max_error_share: float,
    quality: Sequence[dict[str, float | int]],
    generator: dict[str, Any],
    backend_details: dict[str, Any] | None = None,
) -> Path:
    """Randomly mix cohort families, then write data and provenance."""

    out.parent.mkdir(parents=True, exist_ok=True)
    mixed_rows = mix_family_rows(rows, seed=int(generator["seed"]))
    DatasetArtifacts.write(out, mixed_rows)
    sidecar = write_manifest(
        out,
        rows=len(mixed_rows),
        subject_backend=f"{SUBJECT.app_name}/{SUBJECT.class_name}",
        families=build.families,
        triage=build.triage,
        max_error_share=max_error_share,
        quality=quality,
        generator=generator,
        backend_details=backend_details,
    )
    print(f"wrote {len(mixed_rows)} randomly mixed cohort rows to {out}")
    print(f"wrote manifest to {sidecar}")
    return sidecar


def resolve_families(
    requested: list[str] | None,
    split: FamilySplit | None,
    *,
    default_split: FamilySplit,
) -> list[str]:
    """Resolve the common repeatable-family/split CLI contract."""

    if requested is not None and split not in (None, "all"):
        raise typer.BadParameter(
            "--family and an explicit --split are mutually exclusive"
        )
    if requested is not None:
        return requested
    selected = split or default_split
    if selected == "train":
        return list(TRAIN_FAMILIES)
    if selected == "eval":
        return list(EVAL_FAMILIES)
    return list(FAMILIES)


def prepare_candidates(
    subject: Subject, samples: Sequence[Any]
) -> tuple[list[PreparedSample], dict[str, int]]:
    prepared, triage = prepare_samples(subject, list(samples))
    print("answer sampling triage:", triage)
    if not prepared:
        raise SystemExit("sampling produced no attributable answers")
    return prepared, triage


def grader_baseline_failures(
    subject: Subject, examples: Sequence[DatasetExample]
) -> list[str | None]:
    """Replay each example's unintervened baseline exactly as grading does.

    Returns ``None`` for examples whose replayed top token matches the stored
    graph, ``"drift"`` when the grader would flag baseline drift, and
    ``"error"`` when the replay itself fails (for example, a top token that
    does not round-trip). Grading would score either failure as zero.
    """

    payloads = [ablation_payload(example, []) for example in examples]
    calls = [
        subject.spawn("evaluate_ablation_cohort", [payload], []) for payload in payloads
    ]
    failures: list[str | None] = []
    for payload, call in zip(payloads, calls, strict=True):
        try:
            validate_ablation_result(call.get(), [payload], [])
        except ValidationError:
            failures.append("drift")
        except Exception as exc:
            print(f"  grader replay failed for {payload['prompt']!r}: {exc}")
            failures.append("error")
        else:
            failures.append(None)
    if failures and all(failure == "error" for failure in failures):
        raise RuntimeError("every grader baseline replay in the batch failed")
    return failures


def clean_attributions(
    subject: Subject,
    prepared: Sequence[PreparedSample],
    *,
    max_error_share: float,
    batch_size: int,
    concurrent_calls: bool,
) -> Iterator[CleanAttribution]:
    """Yield faithful, quality-screened maximal graphs in bounded batches.

    Survivors are replayed through the grader's baseline check, so samples
    that would fail as baseline drift are dropped here and the builders'
    batch loops sample replacements.
    """

    for start in range(0, len(prepared), batch_size):
        batch = prepared[start : start + batch_size]
        if concurrent_calls:
            calls = [
                subject.spawn_attribution_graph(item.sample.prompt) for item in batch
            ]
            raw_graphs: Iterable[object] = (call.get() for call in calls)
        else:
            raw_graphs = subject.attribution_graphs(
                [item.sample.prompt for item in batch]
            )
        screened: list[CleanAttribution] = []
        for item, raw_graph in zip(batch, raw_graphs, strict=True):
            if raw_graph is None:
                continue
            graph = TracerGraphPayload.model_validate(raw_graph)
            faithful, _ = is_faithful(graph, item.target_token)
            stats = graph_quality(graph, 60)
            if faithful and stats["error"] <= max_error_share:
                screened.append(CleanAttribution(item, graph, stats))
        failures = grader_baseline_failures(
            subject,
            [
                DatasetExample(
                    prompt=attributed.prepared.sample.prompt,
                    target_token=attributed.prepared.target_token,
                    graph=attributed.graph,
                )
                for attributed in screened
            ],
        )
        for attributed, failure in zip(screened, failures, strict=True):
            if failure is None:
                yield attributed
        dropped = {
            reason: failures.count(reason)
            for reason in ("drift", "error")
            if reason in failures
        }
        print(
            f"processed {min(start + len(batch), len(prepared))}/{len(prepared)} "
            f"clean attributions; grader replay dropped {dropped or 'none'}"
        )


def mean_quality(quality: Sequence[dict[str, float | int]]) -> dict[str, float | None]:
    return {
        "error_influence": (
            sum(float(item["error"]) for item in quality) / len(quality)
            if quality
            else None
        ),
        "displayed_feature_influence": (
            sum(float(item["displayed"]) for item in quality) / len(quality)
            if quality
            else None
        ),
    }


def write_manifest(
    out: Path,
    *,
    rows: int,
    subject_backend: str,
    families: Sequence[str],
    triage: dict[str, int],
    max_error_share: float,
    quality: Sequence[dict[str, float | int]],
    generator: dict[str, Any],
    backend_details: dict[str, Any] | None = None,
) -> Path:
    """Write the provenance schema shared by every generated dataset."""

    manifest = {
        "dataset": out.name,
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "rows": rows,
        "sha256": DatasetArtifacts.sha256(out),
        "subject_backend": subject_backend,
        "subject": {
            "model": SUBJECT.config.model_name,
            "transcoder_set": SUBJECT.config.transcoder_set,
            "node_threshold": SUBJECT.config.node_threshold,
            "edge_threshold": SUBJECT.config.edge_threshold,
            "backend": subject_backend,
        },
        **(backend_details or {}),
        "maximal_graphs": True,
        "max_feature_nodes": MAX_FEATURE_NODES,
        "generation_policy": GENERATION_POLICY,
        "sampler": {"max_new_tokens": MAX_NEW_TOKENS, "triage": triage},
        "screening": {
            "exact_first_token": True,
            "max_error_share": max_error_share,
            "grader_baseline_replay": True,
        },
        "generator": {**generator, "families": list(families)},
        "graph_quality": mean_quality(quality),
    }
    path = DatasetArtifacts.manifest(out)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path
