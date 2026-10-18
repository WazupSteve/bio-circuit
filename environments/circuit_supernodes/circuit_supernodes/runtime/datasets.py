"""Access to plain or compressed JSONL datasets."""

from __future__ import annotations

import gzip
import io
from collections.abc import Collection, Generator, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TextIO, TypeVar

from pydantic import BaseModel

DATASET_SCHEMA_VERSION = "runtime-minimal-v2"
GENERATION_POLICY = "maximal-circuit-tracer-export-v1"
HF_DATASET_REPO = "amitprakash2005/bio-circuit"
# Immutable public release; update only when publishing new dataset bytes.
HF_DATASET_REVISION = "50bcbf595d00be9ca6b8a8969fae2fed278258b5"
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


_CHUNK_BYTES = 16 * 1024 * 1024


def _open_binary(path: Path) -> gzip.GzipFile | io.BufferedReader:
    return gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")


def jsonl_lines(
    path: Path, *, only: Collection[int] | None = None
) -> Iterator[tuple[int, bytes]]:
    """Yield ``(row_index, line)`` for nonblank JSONL rows.

    Rows are numbered like ``load_jsonl_rows``. With ``only``, every other row is
    skipped in fixed-size chunks without being buffered, so reaching a late row
    costs decompression rather than parsing or holding the rows before it.
    """

    stop = max(only, default=-1) + 1 if only is not None else None
    index = 0
    parts: list[bytes] = []
    blank = True
    with _open_binary(path) as stream:
        while chunk := stream.read(_CHUNK_BYTES):
            start = 0
            while True:
                end = chunk.find(b"\n", start)
                keep = only is None or index in only
                if keep or blank:
                    piece = chunk[start:] if end < 0 else chunk[start:end]
                    if blank and piece and not piece.isspace():
                        blank = False
                    if keep:
                        parts.append(piece)
                if end < 0:
                    break
                if not blank:
                    if parts:
                        yield index, b"".join(parts)
                    index += 1
                    if stop is not None and index >= stop:
                        return
                parts.clear()
                blank = True
                start = end + 1
    if not blank and parts:
        yield index, b"".join(parts)


RowT = TypeVar("RowT", bound=BaseModel)


def load_jsonl_row(  # noqa: UP047 -- package supports Python 3.11
    path: Path, row_type: type[RowT], index: int
) -> RowT | None:
    """Validate only the requested nonblank row; ``None`` if the file is shorter."""

    for _, line in jsonl_lines(path, only={index}):
        try:
            return row_type.model_validate_json(line)
        except Exception as exc:
            raise ValueError(f"{path} row {index + 1}: {exc}") from exc
    return None


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
