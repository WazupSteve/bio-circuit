"""One disk-backed attribution graph with bounded, paginated policy queries."""

from __future__ import annotations

import json
import sqlite3
import tempfile
from contextlib import closing, suppress
from pathlib import Path
from typing import Literal
from uuid import uuid4

from ..models import TracerGraphPayload
from .graphs import influence_deltas, node_kind, stable_feature_id, top_outputs

MAX_QUERY_ITEMS = 50
MAX_RESPONSE_BYTES = 12_000
QueryView = Literal["nodes", "edges", "tokens"]


class GraphStore:
    def __init__(self) -> None:
        self.directory = tempfile.TemporaryDirectory(prefix="circuit-graph-")
        self.path = Path(self.directory.name) / "graph.sqlite"
        self.graph_id = uuid4().hex

    def close(self) -> None:
        self.directory.cleanup()

    def write(self, graph: TracerGraphPayload, *, include_outputs: bool) -> dict:
        deltas = influence_deltas(graph)
        hidden = {
            node.node_id
            for node in graph.nodes
            if not include_outputs and node_kind(node) == "logit"
        }
        with closing(sqlite3.connect(self.path)) as db, db:
            db.executescript("""
                CREATE TABLE nodes (
                    id TEXT, stable_id TEXT, kind TEXT, delta REAL, data TEXT
                );
                CREATE TABLE edges (
                    source TEXT, target TEXT, magnitude REAL, data TEXT
                );
                CREATE TABLE tokens (data TEXT);
            """)
            for node in graph.nodes:
                if node.node_id in hidden:
                    continue
                kind = node_kind(node)
                stable_id = None
                if kind == "feature":
                    with suppress(ValueError):
                        stable_id = stable_feature_id(node.node_id)
                data = node.model_dump()
                data.update(
                    kind=kind, stable_id=stable_id, influence_delta=deltas[node.node_id]
                )
                db.execute(
                    "INSERT INTO nodes VALUES (?, ?, ?, ?, ?)",
                    (
                        node.node_id,
                        stable_id,
                        kind,
                        deltas[node.node_id],
                        json.dumps(data),
                    ),
                )
            db.executemany(
                "INSERT INTO edges VALUES (?, ?, ?, ?)",
                (
                    (edge.source, edge.target, abs(edge.weight), edge.model_dump_json())
                    for edge in graph.links
                    if edge.source not in hidden and edge.target not in hidden
                ),
            )
            db.executemany(
                "INSERT INTO tokens VALUES (?)",
                (
                    (json.dumps({"position": i, "token": token}),)
                    for i, token in enumerate(graph.metadata.prompt_tokens)
                ),
            )
            db.executescript("""
                CREATE INDEX node_rank ON nodes(delta DESC);
                CREATE INDEX node_id ON nodes(id);
                CREATE INDEX node_stable ON nodes(stable_id);
                CREATE INDEX edge_rank ON edges(magnitude DESC);
                CREATE INDEX edge_source ON edges(source);
                CREATE INDEX edge_target ON edges(target);
            """)
            summary = {
                "graph_id": self.graph_id,
                "node_count": db.execute("SELECT count(*) FROM nodes").fetchone()[0],
                "feature_count": db.execute(
                    "SELECT count(*) FROM nodes WHERE kind = 'feature'"
                ).fetchone()[0],
                "edge_count": db.execute("SELECT count(*) FROM edges").fetchone()[0],
                "token_count": len(graph.metadata.prompt_tokens),
                "outputs_redacted": not include_outputs,
                "max_query_items": MAX_QUERY_ITEMS,
                "usage": (
                    "Use query_graph with this graph_id. "
                    "Only the latest attributed graph is retained."
                ),
            }
            if include_outputs:
                summary["top_outputs"] = [
                    {"token": token[:120], "probability": probability}
                    for token, probability in top_outputs(graph)[:5]
                ]
            return summary

    def query(
        self,
        *,
        view: QueryView,
        offset: int,
        limit: int,
        node_id: str | None,
        kind: str | None,
    ) -> dict:
        if view not in ("nodes", "edges", "tokens"):
            raise ValueError("view must be nodes, edges, or tokens")
        if type(offset) is not int or offset < 0:
            raise ValueError("offset must be a nonnegative integer")
        if type(limit) is not int or not 1 <= limit <= MAX_QUERY_ITEMS:
            raise ValueError(f"limit must be between 1 and {MAX_QUERY_ITEMS}")
        if kind not in (None, "feature", "embedding", "error", "logit"):
            raise ValueError("kind must be feature, embedding, error, or logit")
        if view != "nodes" and kind is not None:
            raise ValueError("kind is only supported for nodes")
        if view == "tokens" and node_id is not None:
            raise ValueError("node_id is only supported for nodes and edges")
        where, params = [], []
        if view == "nodes":
            order = "delta DESC, rowid"
            if node_id is not None:
                where.append("(id = ? OR stable_id = ?)")
                params.extend([node_id, node_id])
            if kind is not None:
                where.append("kind = ?")
                params.append(kind)
        elif view == "edges":
            order = "magnitude DESC, rowid"
            if node_id is not None:
                where.append("(source = ? OR target = ?)")
                params.extend([node_id, node_id])
        else:
            order = "rowid"
        clause = " WHERE " + " AND ".join(where) if where else ""
        # Table and ordering are selected above; all user filters are parameters.
        with closing(sqlite3.connect(self.path)) as db, db:
            total = db.execute(
                f"SELECT count(*) FROM {view}{clause}", params
            ).fetchone()[0]
            rows = db.execute(
                f"SELECT data FROM {view}{clause} ORDER BY {order} LIMIT ? OFFSET ?",
                [*params, limit, offset],
            )
            result = {
                "graph_id": self.graph_id,
                "view": view,
                "total": total,
                "offset": offset,
                "items": [],
                "next_offset": None,
            }
            for (raw,) in rows:
                item = json.loads(raw)
                for field in ("clerp", "label", "token"):
                    if field in item and len(item[field]) > 300:
                        item[field] = item[field][:300]
                        item["text_truncated"] = True
                result["items"].append(item)
                next_offset = offset + len(result["items"])
                result["next_offset"] = next_offset if next_offset < total else None
                if len(json.dumps(result).encode()) > MAX_RESPONSE_BYTES:
                    result["items"].pop()
                    if not result["items"]:
                        raise ValueError("graph item exceeds response budget")
                    result["next_offset"] = offset + len(result["items"])
                    break
            return result
