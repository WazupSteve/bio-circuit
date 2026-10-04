"""Access to plain or compressed JSONL datasets."""

from __future__ import annotations

import gzip
from collections.abc import Generator, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TextIO, TypeVar

from pydantic import BaseModel

DATASET_SCHEMA_VERSION = "runtime-minimal-v2"
GENERATION_POLICY = "maximal-circuit-tracer-export-v1"
HF_DATASET_REPO = "amitprakash2005/bio-circuit"
# Immutable public release; update only when publishing new dataset bytes.
HF_DATASET_REVISION = "566f6880c28714a0e28cab3128535b6064b17a33"
PUBLIC_DATASETS = {
    "visible_dataset/heldout-cohorts.jsonl.gz",
    "supernode_rediscovery/promptless-supernodes.jsonl.gz",
    "output_restoration/suppressed-features.jsonl.gz",
}


def resolve_dataset(data_dir: Path, filename: str) -> Path:
    """Prefer an explicit local dataset; fetch known release files anonymously."""
    path = data_dir / filename
    if path.is_file():
        return path
    relative = f"{data_dir.name}/{filename}"
    if relative not in PUBLIC_DATASETS:
        raise FileNotFoundError(
            f"{path} missing. Only evaluation cohorts are published; provide a "
            "local dataset_file for custom or training data."
        )
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=f"data/{relative}",
            repo_type="dataset",
            revision=HF_DATASET_REVISION,
            token=False,
        )
    )


class JsonlDatasetFile:
    """Open a plain or gzip-compressed JSONL dataset for streaming."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @contextmanager
    def open(self) -> Generator[TextIO]:
        if self.path.suffix == ".gz":
            with gzip.open(self.path, "rt", encoding="utf-8") as stream:
                yield stream
        else:
            with self.path.open("rt", encoding="utf-8") as stream:
                yield stream


RowT = TypeVar("RowT", bound=BaseModel)


def load_jsonl_rows(  # noqa: UP047 -- package supports Python 3.11
    path: Path,
    row_type: type[RowT],
    *,
    context: Mapping[str, Any] | None = None,
) -> Iterator[RowT]:
    """Stream nonblank JSONL rows through one Pydantic validation boundary."""

    with JsonlDatasetFile(path).open() as rows:
        for line_number, line in enumerate(rows, 1):
            if not line.strip():
                continue
            try:
                yield row_type.model_validate_json(line, context=context)
            except Exception as exc:
                raise ValueError(f"{path} row {line_number}: {exc}") from exc
