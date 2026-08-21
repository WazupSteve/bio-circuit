"""Streaming helpers for the very large graph-backed JSONL datasets."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal, TextIO

from pydantic import BaseModel


@contextmanager
def open_text(path: Path, mode: Literal["rt", "wt", "at"] = "rt") -> Iterator[TextIO]:
    """Open ordinary or gzip-compressed UTF-8 text based on the final suffix."""

    if path.suffix == ".gz":
        with gzip.open(path, mode, encoding="utf-8") as stream:
            yield stream
    else:
        with path.open(mode, encoding="utf-8") as stream:
            yield stream


def write_jsonl_atomic(path: Path, rows: Iterable[BaseModel | dict[str, Any]]) -> None:
    """Stream compact model or mapping rows to an atomic destination."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    try:
        with (
            open_text(temporary, "wt")
            if path.suffix != ".gz"
            else gzip.open(temporary, "wt", encoding="utf-8", compresslevel=1) as stream
        ):
            for row in rows:
                payload = (
                    row.model_dump(
                        mode="json", exclude_defaults=True, exclude_none=True
                    )
                    if isinstance(row, BaseModel)
                    else row
                )
                json.dump(payload, stream, separators=(",", ":"))
                stream.write("\n")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    # Keep chunk_size in the compatibility signature; file_digest chooses an
    # efficient implementation-specific buffer size.
    del chunk_size
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def manifest_path(path: Path) -> Path:
    """Return ``name.manifest.json`` for either ``name.jsonl`` or ``name.jsonl.gz``."""

    base = path.with_suffix("") if path.suffix == ".gz" else path
    return base.with_suffix(".manifest.json")


class DatasetArtifacts:
    """Builder-facing persistence for generated datasets and manifests."""

    open = staticmethod(open_text)
    write = staticmethod(write_jsonl_atomic)
    sha256 = staticmethod(file_sha256)
    manifest = staticmethod(manifest_path)
