"""Build an exactly balanced dataset from the six training prompt families.

Unlike ``build_mdl_dataset.py --n 30``, this script replenishes prompts rejected by
the subject-faithfulness and graph-quality screens. The destination is replaced
only after every family reaches its requested quota.
"""

import argparse
import json
from pathlib import Path

import modal
from build_mdl_dataset import (
    DATA_DIR,
    add_neuronpedia_labels,
    is_clean,
    is_faithful,
    prepare_samples,
    write_manifest,
)
from circuit_supernodes.datagen import TRAIN_FAMILIES, generate
from circuit_supernodes.rubric import MODAL_APP, MODAL_CLS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an exact per-family quota from the six training families."
        )
    )
    parser.add_argument("--per-family", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(DATA_DIR / "train_5_each.jsonl"))
    parser.add_argument(
        "--oversample",
        type=int,
        default=2,
        help="candidates per missing row in each attribution round",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=10,
        help="stop without replacing the destination if quotas remain unfilled",
    )
    parser.add_argument(
        "--max-error-share",
        type=float,
        default=0.35,
        help="reject graphs where error nodes carry more than this influence",
    )
    parser.add_argument(
        "--neuronpedia-labels",
        action="store_true",
        help="label the top-ranked 60 features using Neuronpedia",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing destination after all quotas are filled",
    )
    args = parser.parse_args()
    if args.per_family < 1:
        parser.error("--per-family must be at least 1")
    if args.oversample < 1:
        parser.error("--oversample must be at least 1")
    if args.max_rounds < 1:
        parser.error("--max-rounds must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    if out.exists() and not args.force:
        raise SystemExit(f"{out} already exists; pass --force to replace it")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    if tmp.exists():
        raise SystemExit(f"{tmp} already exists; remove it after checking its contents")

    accepted = {family: 0 for family in TRAIN_FAMILIES}
    attempted = {family: 0 for family in TRAIN_FAMILIES}
    faithful = {family: 0 for family in TRAIN_FAMILIES}
    triage = {"keep_correct": 0, "keep_incorrect": 0, "reject": 0}
    seen_prompts: set[str] = set()
    quality: list[dict] = []
    attributor = modal.Cls.from_name(MODAL_APP, MODAL_CLS)()

    try:
        with tmp.open("w") as rows:
            for round_index in range(args.max_rounds):
                samples = []
                for family_index, family in enumerate(TRAIN_FAMILIES):
                    missing = args.per_family - accepted[family]
                    if missing <= 0:
                        continue
                    seed = args.seed + 10_000 * round_index + family_index
                    candidates = generate(
                        missing * args.oversample,
                        seed=seed,
                        families=[family],
                    )
                    samples.extend(
                        sample
                        for sample in candidates
                        if sample.prompt not in seen_prompts
                    )
                    seen_prompts.update(sample.prompt for sample in candidates)

                if not samples:
                    break
                print(
                    f"Round {round_index + 1}: sampling {len(samples)} candidates; "
                    f"accepted so far {accepted}"
                )
                for sample in samples:
                    attempted[sample.family] += 1
                prepared, round_triage = prepare_samples(attributor, samples)
                for disposition, count in round_triage.items():
                    triage[disposition] += count
                graphs = attributor.attribute_prompt.map(
                    [item.sample.prompt for item in prepared], order_outputs=True
                )
                for item, graph in zip(prepared, graphs, strict=True):
                    sample = item.sample
                    family = sample.source_family or sample.family
                    if graph is None:
                        print(f"  ! attribution failed [{family}], skipping")
                        continue
                    is_subject_faithful, top_token = is_faithful(
                        graph, item.target_token
                    )
                    if not is_subject_faithful:
                        print(
                            f"  ! unfaithful [{family}]: model says {top_token!r}, "
                            f"sampled {item.target_token!r} — skipping"
                        )
                        continue
                    faithful[family] += 1
                    is_graph_clean, graph_stats = is_clean(
                        graph, args.max_error_share, 60
                    )
                    if not is_graph_clean:
                        print(
                            f"  ! low-quality [{family}]: error nodes carry "
                            f"{graph_stats['error']:.0%} of influence — skipping"
                        )
                        continue
                    if accepted[family] >= args.per_family:
                        continue
                    if args.neuronpedia_labels:
                        try:
                            labeled = add_neuronpedia_labels(graph)
                            print(f"  + added {labeled} Neuronpedia labels")
                        except (OSError, ValueError) as exc:
                            print(f"  ! Neuronpedia labels unavailable: {exc}")
                    row = sample.to_row()
                    row["sampling"] = item.sampling
                    row["sampling"]["disposition"] = item.disposition
                    row["sampling"]["attributed_target_token"] = item.target_token
                    row["graph"] = graph
                    rows.write(json.dumps(row) + "\n")
                    rows.flush()
                    quality.append(graph_stats)
                    accepted[family] += 1

                if all(count == args.per_family for count in accepted.values()):
                    break

        incomplete = {
            family: args.per_family - count
            for family, count in accepted.items()
            if count < args.per_family
        }
        if incomplete:
            tmp.unlink()
            raise SystemExit(
                f"quotas not filled after {args.max_rounds} rounds: {incomplete}; "
                "destination preserved"
            )

        tmp.replace(out)
        args.n = args.per_family * len(TRAIN_FAMILIES)
        args.families = list(TRAIN_FAMILIES)
        args.triage = triage
        manifest_path = write_manifest(out, args.n, args, quality)
        manifest = json.loads(manifest_path.read_text())
        manifest["generator"].update(
            {
                "per_family": args.per_family,
                "accepted": accepted,
                "attempted": attempted,
                "faithful": faithful,
                "triage": triage,
                "oversample": args.oversample,
                "max_rounds": args.max_rounds,
            }
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Wrote {args.n} rows ({args.per_family} per family) -> {out}")
        print(f"Wrote manifest -> {manifest_path}")
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise


if __name__ == "__main__":
    main()
