"""Validate the released cohorts and stage a public Hugging Face dataset."""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path

from circuit_supernodes.models import CircuitDatasetRow

DATASETS = {
    "visible": "visible_dataset/heldout-cohorts.jsonl.gz",
    "rediscovery": "supernode_rediscovery/promptless-supernodes.jsonl.gz",
    "restoration": "output_restoration/suppressed-features.jsonl.gz",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / "environments/circuit_supernodes/circuit_supernodes/data"
    args.output.mkdir(parents=True, exist_ok=True)
    schema = args.output / "schema.json"
    schema.write_text(
        json.dumps(CircuitDatasetRow.model_json_schema(), indent=2) + "\n"
    )
    inventory = {}
    for name, relative in DATASETS.items():
        path = source / relative
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        manifest_path = path.with_name(path.name.replace(".jsonl.gz", ".manifest.json"))
        manifest = json.loads(manifest_path.read_text())
        if manifest["sha256"] != digest:
            raise ValueError(f"Checksum mismatch: {relative}")
        index_path = args.output / "indexes" / f"{name}.jsonl"
        index_path.parent.mkdir(exist_ok=True)
        count = 0
        with gzip.open(path, "rt") as rows, index_path.open("w") as index:
            for row_index, line in enumerate(rows):
                row = CircuitDatasetRow.model_validate_json(line)
                if name == "rediscovery" and not row.goal:
                    raise ValueError("Rediscovery requires a goal")
                if name == "restoration" and not row.heldout_prompts:
                    raise ValueError("Restoration requires heldout examples")
                summary = {
                    "task_id": f"{name}-{row_index:03d}",
                    "family": row.family,
                    "goal": row.goal,
                    "example_count": len(row.example_prompts),
                    "heldout_count": len(row.heldout_prompts),
                    "control_count": len(row.control_prompts),
                    "data_file": f"data/{relative}",
                    "row_index": row_index,
                    "sha256": digest,
                }
                index.write(json.dumps(summary) + "\n")
                count += 1
                del row, line
                print(f"Validated {name} row {count}", flush=True)
        if count != manifest["rows"]:
            raise ValueError(f"Row count mismatch: {relative}")
        for original in (path, manifest_path):
            target = args.output / "data" / original.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                os.link(original, target)
        inventory[name] = {
            "file": f"data/{relative}",
            "rows": count,
            "sha256": digest,
            "bytes": path.stat().st_size,
        }
    (args.output / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    print(json.dumps(inventory, indent=2), flush=True)


if __name__ == "__main__":
    main()
