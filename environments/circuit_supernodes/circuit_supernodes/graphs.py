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
    activation: float = 0.0


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


def slim_graph(
    g: Graph,
    max_features: int = 60,
    required_feature_ids: set[str] | None = None,
) -> Graph:
    required_feature_ids = required_feature_ids or set()
    if len(required_feature_ids) > max_features:
        raise ValueError("required features exceed the visible feature budget")
    missing = required_feature_ids - {node.id for node in g.feature_nodes}
    if missing:
        raise ValueError(
            f"required features are absent from the graph: {sorted(missing)}"
        )
    ranked = sorted(g.feature_nodes, key=lambda n: -n.share)
    feats = [node for node in ranked if node.id in required_feature_ids]
    feats.extend(
        node
        for node in ranked
        if node.id not in required_feature_ids
    )
    feats = feats[:max_features]
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
            activation=float(raw.get("activation", 0.0) or 0.0),
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
    # circuit-tracer's `influence` is CUMULATIVE, not per-node: it is the running
    # share of total influence accounted for as nodes are added in importance
    # order, and it stops at the exporter's node_threshold (0.8 for this dataset —
    # every graph's maximum is exactly that, which is how you can tell). So the
    # node with the SMALLEST influence is the most important one, and adjacent
    # differences down the sorted list recover each node's own contribution.
    #
    # Do not "fix" this to rank by raw influence. That inverts the ordering and
    # selects the least important features: measured over the held-out graphs,
    # the top nodes by `share` sit at mean layer 28 with 90% at the final token
    # position, while the top nodes by raw influence sit at mean layer 10 with
    # 28% there. See test_share_recovers_per_node_influence.
    ranked = sorted(g.nodes.values(), key=lambda n: n.influence)
    prev = 0.0
    for n in ranked:
        n.share = max(n.influence - prev, 0.0)
        prev = n.influence
    g.top_outputs.sort(key=lambda tp: -tp[1])
    return g


def graph_quality(g: Graph, max_features: int = 60) -> dict[str, float]:
    """Screening statistics for a candidate graph.

    Error nodes stand for everything the transcoders failed to explain, so a
    graph where they carry most of the influence is one whose circuit is mostly
    missing. ``displayed`` is the share of *feature* influence inside the slice
    the agent actually sees — the influence distribution is flat enough that the
    top 60 features carry only about a third of it, which bounds what any
    proposal can claim regardless of how good it is.
    """
    total = sum(n.share for n in g.nodes.values())
    if total <= 0:
        return {"error": 1.0, "feature": 0.0, "displayed": 0.0, "features": 0}
    kinds: dict[str, float] = {}
    for n in g.nodes.values():
        kinds[n.kind] = kinds.get(n.kind, 0.0) + n.share
    feats = sorted(g.feature_nodes, key=lambda n: -n.share)
    feature_total = sum(n.share for n in feats)
    return {
        "error": kinds.get("error", 0.0) / total,
        "feature": kinds.get("feature", 0.0) / total,
        "displayed": (
            sum(n.share for n in feats[:max_features]) / feature_total
            if feature_total > 0
            else 0.0
        ),
        "features": len(feats),
    }


def feature_coords(ids: list[str]) -> list[tuple[int, int, int]]:
    return [tuple(map(int, node_id.split("_"))) for node_id in ids]


def render_graph(g: Graph, max_edges_per_node: int = 5) -> str:
    lines = [f"PROMPT TOKENS: {' | '.join(g.tokens)}"]
    if g.top_outputs:
        outs = ", ".join(f"{t!r} (p={p:.2f})" for t, p in g.top_outputs[:5])
        lines.append(f"TOP OUTPUTS: {outs}")
    lines.append("")
    lines.append(
        "NODES (id | kind | layer | pos:token | activation | importance | label):"
    )
    ordered = sorted(
        g.nodes.values(), key=lambda n: (n.kind != "embedding", n.layer, n.ctx_idx)
    )
    lines.extend(
        (
            f"  {n.id} | {n.kind} | L{n.layer} | {n.ctx_idx}:{n.token!r}"
            f" | {n.activation:+.5g} | {n.share:.4f} | {n.label[:70]}"
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
