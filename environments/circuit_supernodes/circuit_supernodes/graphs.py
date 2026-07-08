"""Normalized attribution-graph representation + prompt rendering.

Attribution graphs are computed on Modal by circuit-tracer (see modal_app/)
and cached as JSON. This module normalizes that JSON into a compact form the
policy model can read in-context, independent of upstream schema drift.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GraphNode:
    id: str
    kind: str  # "feature" | "embedding" | "error" | "logit"
    layer: int
    ctx_idx: int
    token: str  # prompt token at ctx_idx (or logit token)
    influence: float  # circuit-tracer cumulative score: LOWER = more important
    share: float = (
        0.0  # this node's normalized influence share (higher = more important)
    )
    activation: float = 0.0
    label: str = ""  # autointerp / clerp label when available


@dataclass
class GraphEdge:
    source: str
    target: str
    weight: float


@dataclass
class Graph:
    prompt: str
    tokens: list[str]
    top_outputs: list[tuple[str, float]]  # (token, prob)
    nodes: dict[str, GraphNode] = field(default_factory=dict)
    edges: list[GraphEdge] = field(default_factory=list)

    @property
    def feature_nodes(self) -> list[GraphNode]:
        return [n for n in self.nodes.values() if n.kind == "feature"]

    def adjacency(self) -> dict[tuple[str, str], float]:
        adj: dict[tuple[str, str], float] = {}
        for e in self.edges:
            key = (e.source, e.target)
            adj[key] = adj.get(key, 0.0) + e.weight
        return adj


def slim_graph(g: Graph, max_features: int = 60) -> Graph:
    """Reduce to the subgraph shown to the policy model: top features by
    influence share plus all embedding/logit nodes, and edges among them."""
    feats = sorted(g.feature_nodes, key=lambda n: -n.share)[:max_features]
    keep = {n.id for n in feats} | {
        n.id for n in g.nodes.values() if n.kind in ("embedding", "logit")
    }
    out = Graph(prompt=g.prompt, tokens=list(g.tokens), top_outputs=list(g.top_outputs))
    out.nodes = {nid: g.nodes[nid] for nid in keep}
    out.edges = [e for e in g.edges if e.source in keep and e.target in keep]
    return out


def graph_to_json(g: Graph) -> dict:
    """Serialize a (slimmed) Graph to our own compact schema."""
    return {
        "format": SLIM_FORMAT,
        "prompt": g.prompt,
        "tokens": g.tokens,
        "top_outputs": g.top_outputs,
        "nodes": [vars(n) for n in g.nodes.values()],
        "edges": [[e.source, e.target, e.weight] for e in g.edges],
    }


def _load_slim(data: dict) -> Graph:
    g = Graph(
        prompt=data["prompt"],
        tokens=list(data["tokens"]),
        top_outputs=[(t, p) for t, p in data["top_outputs"]],
    )
    g.nodes = {n["id"]: GraphNode(**n) for n in data["nodes"]}
    g.edges = [GraphEdge(s, t, w) for s, t, w in data["edges"]]
    return g


SLIM_FORMAT = "slim"


def _node_kind(raw: dict) -> str:
    ft = str(raw.get("feature_type", raw.get("type", ""))).lower()
    if "embed" in ft:
        return "embedding"
    if "error" in ft:
        return "error"
    if "logit" in ft:
        return "logit"
    return "feature"


def load_graph_json(data: dict) -> Graph:
    """Normalize a circuit-tracer / Neuronpedia graph JSON dict (or our own
    slim format produced by graph_to_json)."""
    if data.get("format") == SLIM_FORMAT:
        return _load_slim(data)
    meta = data.get("metadata", {})
    tokens = meta.get("prompt_tokens", data.get("tokens", []))
    g = Graph(
        prompt=meta.get("prompt", data.get("prompt", "")),
        tokens=list(tokens),
        top_outputs=[],
        nodes={},
        edges=[],
    )
    for raw in data.get("nodes", []):
        nid = str(raw.get("node_id", raw.get("id")))
        kind = _node_kind(raw)
        ctx = int(raw.get("ctx_idx", raw.get("ctx", 0)) or 0)
        layer_raw = raw.get("layer", 0)
        try:
            layer = int(layer_raw)
        except (TypeError, ValueError):
            layer = 0  # embedding layer "E"
        token = ""
        if kind == "logit":
            # clerp looks like: Output " Austin" (p=0.85); token may be multiline
            clerp = str(raw.get("clerp", raw.get("token", "")))
            m = re.search(r'"(.*?)"', clerp, re.DOTALL)
            token = m.group(1) if m else clerp
        elif 0 <= ctx < len(tokens):
            token = tokens[ctx]
        node = GraphNode(
            id=nid,
            kind=kind,
            layer=layer,
            ctx_idx=ctx,
            token=token,
            influence=float(raw.get("influence", 0.0) or 0.0),
            activation=float(raw.get("activation", 0.0) or 0.0),
            label=str(raw.get("clerp", "") or raw.get("label", "") or ""),
        )
        g.nodes[nid] = node
        if kind == "logit":
            prob = float(raw.get("token_prob", 0.0) or 0.0)
            g.top_outputs.append((node.token, prob))
    for raw in data.get("links", data.get("edges", [])):
        g.edges.append(
            GraphEdge(
                source=str(raw.get("source")),
                target=str(raw.get("target")),
                weight=float(raw.get("weight", 0.0) or 0.0),
            )
        )
    # Recover each node's influence share from the cumulative scores:
    # circuit-tracer sorts nodes by influence descending and cumsums, so the
    # share of node i (sorted ascending by cumulative value) is the diff from
    # its predecessor.
    ranked = sorted(g.nodes.values(), key=lambda n: n.influence)
    prev = 0.0
    for n in ranked:
        n.share = max(n.influence - prev, 0.0)
        prev = n.influence
    g.top_outputs.sort(key=lambda tp: -tp[1])
    return g


def load_graph_file(path: str | Path) -> Graph:
    return load_graph_json(json.loads(Path(path).read_text()))


def render_graph(g: Graph, max_nodes: int = 60, max_edges_per_node: int = 5) -> str:
    """Compact textual rendering of the pruned graph for the policy prompt."""
    feats = sorted(g.feature_nodes, key=lambda n: -n.share)[:max_nodes]
    keep = {n.id for n in feats}
    keep |= {n.id for n in g.nodes.values() if n.kind in ("embedding", "logit")}

    lines = [f"PROMPT TOKENS: {' | '.join(g.tokens)}"]
    if g.top_outputs:
        outs = ", ".join(f"{t!r} (p={p:.2f})" for t, p in g.top_outputs[:5])
        lines.append(f"TOP OUTPUTS: {outs}")
    lines.append("")
    lines.append("NODES (id | kind | layer | pos:token | importance | label):")
    ordered = [n for n in g.nodes.values() if n.id in keep]
    ordered.sort(key=lambda n: (n.kind != "embedding", n.layer, n.ctx_idx))
    for n in ordered:
        label = n.label[:70] if n.label else ""
        lines.append(
            f"  {n.id} | {n.kind} | L{n.layer} | {n.ctx_idx}:{n.token!r}"
            f" | {n.share:.4f} | {label}"
        )
    lines.append("")
    lines.append("EDGES (source -> target : weight):")
    by_target: dict[str, list[GraphEdge]] = {}
    for e in g.edges:
        if e.source in keep and e.target in keep:
            by_target.setdefault(e.target, []).append(e)
    for tgt, es in sorted(by_target.items()):
        es.sort(key=lambda e: -abs(e.weight))
        for e in es[:max_edges_per_node]:
            lines.append(f"  {e.source} -> {e.target} : {e.weight:+.3f}")
    return "\n".join(lines)
