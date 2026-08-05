"""Profile a reference and provided model on aligned dataset features."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import modal
from circuit_supernodes.graphs import feature_coords, load_graph_json, slim_graph

APP_NAME = "circuit-feature-suppression"
CLASS_NAME = "SuppressionProfiler"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect aligned causal feature profiles for a model pair"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reference-model", required=True)
    parser.add_argument("--provided-model", required=True)
    parser.add_argument("--transcoders", required=True)
    parser.add_argument("--max-features", type=int, default=60)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.max_features < 1:
        parser.error("--max-features must be positive")
    if args.out.exists() and not args.overwrite:
        raise SystemExit(f"{args.out} exists; pass --overwrite to replace it")

    profiler_class = modal.Cls.from_name(APP_NAME, CLASS_NAME)
    reference_profiler = profiler_class(
        model_name=args.reference_model, transcoder_set=args.transcoders
    )
    provided_profiler = profiler_class(
        model_name=args.provided_model, transcoder_set=args.transcoders
    )
    pair = {
        "format": "paired-feature-profiles-v1",
        "reference": {
            "model": args.reference_model,
            "transcoders": args.transcoders,
            "records": [],
        },
        "provided": {
            "model": args.provided_model,
            "transcoders": args.transcoders,
            "records": [],
        },
    }

    with args.dataset.open() as rows:
        for line_number, line in enumerate(rows, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            graph = slim_graph(load_graph_json(row["graph"]), args.max_features)
            feature_ids = sorted(node.id for node in graph.feature_nodes)
            if not feature_ids:
                raise ValueError(f"row {line_number} has no feature nodes")
            if not graph.top_outputs or not graph.top_outputs[0][0].strip():
                raise ValueError(f"row {line_number} has no target token")
            target_token = graph.top_outputs[0][0]
            prompt_id = hashlib.sha256(
                json.dumps(
                    [row["prompt"], target_token, feature_ids], separators=(",", ":")
                ).encode()
            ).hexdigest()[:20]
            coordinates = feature_coords(feature_ids)
            reference = reference_profiler.profile_features.remote(
                row["prompt"], coordinates, target_token
            )
            provided = provided_profiler.profile_features.remote(
                row["prompt"], coordinates, target_token
            )
            if reference["token_fingerprint"] != provided["token_fingerprint"]:
                raise ValueError(
                    f"row {line_number} tokenizes differently across the model pair"
                )
            for destination, profile in (
                (pair["reference"], reference),
                (pair["provided"], provided),
            ):
                destination["records"].append(
                    {
                        "prompt_id": prompt_id,
                        "family": row.get("family"),
                        "target_token": target_token,
                        "token_fingerprint": profile["token_fingerprint"],
                        "measurements": profile["measurements"],
                    }
                )
            print(
                f"profiled row {line_number}: {prompt_id} "
                f"({len(feature_ids)} features)",
                flush=True,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".partial")
    temporary.write_text(json.dumps(pair, separators=(",", ":")) + "\n")
    temporary.replace(args.out)
    print(f"wrote paired profiles to {args.out}")


if __name__ == "__main__":
    main()
