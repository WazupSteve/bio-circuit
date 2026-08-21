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
