"""Attach frozen causal-response banks to a dataset JSONL.

Run this after deploying ``modal_app/attribution.py``.  The output is a new
dataset file; the input is never edited in place.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import modal
from circuit_supernodes.graphs import feature_coords, load_graph_json
from circuit_supernodes.rubric import MODAL_APP, MODAL_CLS, REWARD_VERSION

DATA_DIR = Path(__file__).parent.parent / "circuit_supernodes" / "data" / "circuit_mdl"


def fractions(value: str) -> list[float]:
    parsed = [float(item) for item in value.split(",") if item.strip()]
    if not parsed or any(not 0.0 < item <= 1.0 for item in parsed):
        raise argparse.ArgumentTypeError(
            "fractions must be comma-separated values in (0,1]"
        )
    return parsed


def dataset_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else DATA_DIR / path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute proposal-independent MDL response banks"
    )
    parser.add_argument("--input", default="circuit-mdl.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--calibration", type=fractions, default=[0.25, 0.5])
    parser.add_argument("--validation", type=fractions, default=[0.75, 1.0])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = dataset_path(args.input).resolve()
    destination = dataset_path(args.out).resolve()
    if source == destination:
        raise SystemExit("--out must differ from --input")
    if destination.exists() and not args.overwrite:
        raise SystemExit(f"{destination} exists; pass --overwrite to replace it")
    if set(args.calibration) & set(args.validation):
        raise SystemExit("calibration and validation fractions must be disjoint")
    if args.workers < 1:
        raise SystemExit("--workers must be positive")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    attributor = modal.Cls.from_name(MODAL_APP, MODAL_CLS)()
    completed = 0

    def augment(line_number: int, row: dict) -> tuple[dict, str]:
        graph = load_graph_json(row["graph"])
        feature_ids = sorted(node.id for node in graph.feature_nodes)
        if not feature_ids:
            raise ValueError(f"row {line_number} has no exported features")
        if not graph.top_outputs or not graph.top_outputs[0][0].strip():
            raise ValueError(f"row {line_number} has no target token")
        bank = attributor.causal_response_bank.remote(
            row["prompt"],
            feature_coords(feature_ids),
            graph.top_outputs[0][0],
            args.calibration,
            args.validation,
        )
        augmented = dict(row)
        augmented.pop("response_bank_max_features", None)
        augmented["response_bank"] = bank
        augmented["response_bank_feature_count"] = len(feature_ids)
        augmented["reward_version"] = REWARD_VERSION
        return augmented, bank["metadata"]["bank_id"]

    try:
        with (
            source.open() as rows,
            temporary.open("w") as output,
            ThreadPoolExecutor(max_workers=args.workers) as executor,
        ):
            pending: deque[Future[tuple[dict, str]]] = deque()

            def write_next() -> None:
                nonlocal completed
                augmented, bank_id = pending.popleft().result()
                output.write(json.dumps(augmented, separators=(",", ":")) + "\n")
                completed += 1
                print(f"banked {completed}: {bank_id}", flush=True)

            for line_number, line in enumerate(rows, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                pending.append(executor.submit(augment, line_number, row))
                if len(pending) >= args.workers:
                    write_next()
            while pending:
                write_next()
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(f"wrote {completed} rows to {destination}")
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    manifest = {
        "dataset": destination.name,
        "rows": completed,
        "sha256": digest,
        "source": str(source),
        "reward_version": REWARD_VERSION,
        "calibration": args.calibration,
        "validation": args.validation,
    }
    manifest_path = destination.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
