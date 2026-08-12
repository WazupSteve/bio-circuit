"""Build hidden-prompt goal cohorts with answer-aware subject sampling."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import modal
from circuit_supernodes.datagen import (
    EVAL_FAMILIES,
    FAMILIES,
    TRAIN_FAMILIES,
    generate_pool,
)
from circuit_supernodes.graphs import graph_quality, load_graph_json
from circuit_supernodes.rediscovery import DEFAULT_MODAL_APP, DEFAULT_MODAL_CLS
from circuit_supernodes.sampling import MAX_NEW_TOKENS, answer_key
from scripts.build_mdl_dataset import is_faithful, prepare_samples

DATA_DIR = (
    Path(__file__).parent.parent
    / "circuit_supernodes"
    / "data"
    / "supernode_rediscovery"
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cohort_rows(
    records: list[dict],
    *,
    tasks: int,
    cohort_size: int,
    control_size: int,
    seed: int,
) -> list[dict]:
    """Group same-answer examples and sample different-answer controls."""

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        key = (answer_key(str(record["target"])), str(record["target_token"]))
        grouped[key].append(record)

    rng = random.Random(seed)
    candidates = [
        (key, values)
        for key, values in sorted(grouped.items())
        if len(values) >= cohort_size
    ]
    rng.shuffle(candidates)
    rows = []
    for (_, target_token), positives in candidates:
        desired_answer = str(positives[0]["target"]).strip()
        controls = [
            record
            for record in records
            if answer_key(str(record["target"])) != answer_key(desired_answer)
        ]
        if len(controls) < control_size:
            continue
        selected_positive = rng.sample(positives, cohort_size)
        selected_controls = rng.sample(controls, control_size)
        cohort_families = sorted({str(item["family"]) for item in selected_positive})
        family = cohort_families[0] if len(cohort_families) == 1 else "mixed"
        rows.append(
            {
                "family": family,
                "target_concept": f"{family} goal for answer {desired_answer!r}",
                "desired_answer": desired_answer,
                "desired_target_token": target_token,
                "cohort": selected_positive,
                "controls": selected_controls,
            }
        )
        if len(rows) == tasks:
            break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=480, help="candidate prompt count")
    parser.add_argument("--tasks", type=int, default=16)
    parser.add_argument("--cohort-size", type=int, default=4)
    parser.add_argument("--control-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=("train", "eval", "all"), default="all")
    parser.add_argument("--families", nargs="*")
    parser.add_argument("--out", default=str(DATA_DIR / "promptless-supernodes.jsonl"))
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
    attributor = modal.Cls.from_name(DEFAULT_MODAL_APP, DEFAULT_MODAL_CLS)()
    prepared, triage = prepare_samples(attributor, samples)
    graphs = list(
        attributor.attribute_prompt.map(
            [item.sample.prompt for item in prepared], order_outputs=True
        )
    )

    records = []
    quality = []
    for item, graph_json in zip(prepared, graphs, strict=True):
        if not graph_json:
            continue
        faithful, _ = is_faithful(graph_json, item.target_token)
        stats = graph_quality(load_graph_json(graph_json), 60)
        if not faithful or stats["error"] > args.max_error_share:
            continue
        quality.append(stats)
        records.append(
            {
                **item.sample.to_row(),
                "target_token": item.target_token,
                "sampling": {
                    **item.sampling,
                    "disposition": item.disposition,
                    "attributed_target_token": item.target_token,
                },
                "graph": graph_json,
            }
        )

    rows = cohort_rows(
        records,
        tasks=args.tasks,
        cohort_size=args.cohort_size,
        control_size=args.control_size,
        seed=args.seed,
    )
    if len(rows) < args.tasks:
        raise SystemExit(
            f"formed only {len(rows)}/{args.tasks} cohorts; increase --n or "
            "reduce cohort/control sizes"
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
        "subject_backend": f"{DEFAULT_MODAL_APP}/{DEFAULT_MODAL_CLS}",
        "maximal_graphs": True,
        "generator": {
            "seed": args.seed,
            "candidates": args.n,
            "families": families,
            "cohort_size": args.cohort_size,
            "control_size": args.control_size,
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
    print(f"wrote {len(rows)} promptless cohorts to {out}")
    print(f"wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
