import re
from dataclasses import dataclass, field


@dataclass
class GraphNode:
    id: str
    kind: str
    layer: int
    ctx_idx: int
    token: str
    influence: float
    share: float = 0.0
    label: str = ""


@dataclass
class GraphEdge:
    source: str
    target: str
    weight: float


@dataclass
class Graph:
    tokens: list[str]
    top_outputs: list[tuple[str, float]]
    nodes: dict[str, GraphNode] = field(default_factory=dict)
    edges: list[GraphEdge] = field(default_factory=list)

    @property
    def feature_nodes(self) -> list[GraphNode]:
        return [n for n in self.nodes.values() if n.kind == "feature"]


def slim_graph(g: Graph, max_features: int = 60) -> Graph:
    feats = sorted(g.feature_nodes, key=lambda n: -n.share)[:max_features]
    keep = {n.id for n in feats} | {
        n.id for n in g.nodes.values() if n.kind in ("embedding", "logit")
    }
    out = Graph(tokens=list(g.tokens), top_outputs=list(g.top_outputs))
    out.nodes = {nid: g.nodes[nid] for nid in keep}
    out.edges = [e for e in g.edges if e.source in keep and e.target in keep]
    return out


def graph_to_json(g: Graph) -> dict:
    return {
        "format": SLIM_FORMAT,
        "tokens": g.tokens,
        "top_outputs": g.top_outputs,
        "nodes": [vars(n) for n in g.nodes.values()],
        "edges": [[e.source, e.target, e.weight] for e in g.edges],
    }


def _load_slim(data: dict) -> Graph:
    return Graph(
        tokens=list(data["tokens"]),
        top_outputs=[(t, p) for t, p in data["top_outputs"]],
        nodes={n["id"]: GraphNode(**n) for n in data["nodes"]},
        edges=[GraphEdge(s, t, w) for s, t, w in data["edges"]],
    )


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
    if data.get("format") == SLIM_FORMAT:
        return _load_slim(data)
    meta = data.get("metadata", {})
    tokens = meta.get("prompt_tokens", data.get("tokens", []))
    g = Graph(tokens=list(tokens), top_outputs=[])
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
    # Upstream influence is cumulative; adjacent differences recover node shares.
    ranked = sorted(g.nodes.values(), key=lambda n: n.influence)
    prev = 0.0
    for n in ranked:
        n.share = max(n.influence - prev, 0.0)
        prev = n.influence
    g.top_outputs.sort(key=lambda tp: -tp[1])
    return g


def feature_coords(ids: list[str]) -> list[tuple[int, int, int]]:
    return [tuple(map(int, node_id.split("_"))) for node_id in ids]


def embed_supernodes(
    raw: dict, supernodes: dict[str, list[str]], slug: str = ""
) -> dict:
    pinned = [m for ms in supernodes.values() for m in ms]
    logits = [n for n in raw.get("nodes", []) if _node_kind(n) == "logit"]
    if logits:
        top = max(logits, key=lambda n: float(n.get("token_prob", 0.0) or 0.0))
        pinned.append(str(top.get("node_id", top.get("id"))))
    out = dict(raw)
    out["qParams"] = {
        **raw.get("qParams", {}),
        "supernodes": [[name, *ms] for name, ms in supernodes.items()],
        "pinnedIds": pinned,
    }
    if slug:
        out["metadata"] = {**raw.get("metadata", {}), "slug": slug}
    return out


def render_graph(g: Graph, max_edges_per_node: int = 5) -> str:
    lines = [f"PROMPT TOKENS: {' | '.join(g.tokens)}"]
    if g.top_outputs:
        outs = ", ".join(f"{t!r} (p={p:.2f})" for t, p in g.top_outputs[:5])
        lines.append(f"TOP OUTPUTS: {outs}")
    lines.append("")
    lines.append("NODES (id | kind | layer | pos:token | importance | label):")
    ordered = sorted(
        g.nodes.values(), key=lambda n: (n.kind != "embedding", n.layer, n.ctx_idx)
    )
    lines.extend(
        (
            f"  {n.id} | {n.kind} | L{n.layer} | {n.ctx_idx}:{n.token!r}"
            f" | {n.share:.4f} | {n.label[:70]}"
        )
        for n in ordered
    )
    lines.append("")
    lines.append("EDGES (source -> target : weight):")
    by_target: dict[str, list[GraphEdge]] = {}
    for e in g.edges:
        by_target.setdefault(e.target, []).append(e)
    for _, es in sorted(by_target.items()):
        es.sort(key=lambda e: -abs(e.weight))
        lines.extend(
            f"  {e.source} -> {e.target} : {e.weight:+.3f}"
            for e in es[:max_edges_per_node]
        )
    return "\n".join(lines)
