"""Shared fixture graph.

Three groups of seven features, so a proposal can clear ``MIN_GROUP_SIZE`` and
still leave room for the dominance gate to fire (15 of 21 equal shares is over
the 0.7 ceiling).
"""

from circuit_supernodes.graphs import load_graph_json

GROUPS = {
    "A": [f"1_{10 + i}_1" for i in range(7)],
    "B": [f"2_{20 + i}_2" for i in range(7)],
    "C": [f"3_{30 + i}_3" for i in range(7)],
}
FEATURES = [f for ms in GROUPS.values() for f in ms]
SHARE = 1.0 / len(FEATURES)
PROMPT = "Fact: test prompt is"


def make_graph() -> dict:
    nodes = [
        {
            "id": nid,
            "kind": "feature",
            "layer": int(nid.split("_")[0]),
            "ctx_idx": int(nid.split("_")[2]),
            "token": "t",
            "influence": 0.0,
            "share": SHARE,
            "label": f"feature {nid}",
        }
        for nid in FEATURES
    ]
    nodes.append(
        {
            "id": "E_0_0",
            "kind": "embedding",
            "layer": 0,
            "ctx_idx": 0,
            "token": "Fact",
            "influence": 0.0,
            "share": 0.0,
            "label": "",
        }
    )
    return {
        "format": "slim",
        "tokens": ["Fact", ":", "test", "prompt", "is"],
        "top_outputs": [[" Austin", 0.8]],
        "nodes": nodes,
        # A sparse chain, so community detection has something to latch onto.
        "edges": [[GROUPS["A"][i], GROUPS["B"][i], 1.5] for i in range(7)]
        + [[GROUPS["B"][i], GROUPS["C"][i], 1.2] for i in range(7)],
    }


GRAPH = load_graph_json(make_graph())
