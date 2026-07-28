"""Reference scores for the keyword-matched "expected" circuit and a random one.

Adversarial attacks live in ``oracle_audit.py``: a hack that merely trips a gate
proves nothing, so gate-compliant attacks are scored there against both reward
versions.

    uv run python scripts/calibrate.py --dataset heldout.jsonl
"""

import argparse
import asyncio
import json
import random
from itertools import pairwise
from pathlib import Path

from circuit_supernodes.graphs import Graph, load_graph_json, slim_graph
from circuit_supernodes.rubric import MIN_GROUP_SIZE, InterventionScoring

DATA_DIR = Path(__file__).parent.parent / "circuit_supernodes" / "data"


def expected_proposal(row: dict, g: Graph) -> dict:
    supernodes, used = [], set()
    for name, keywords in row["expected_supernodes"]:
        members = [
            n.id
            for n in g.feature_nodes
            if n.id not in used
            and any(k.lower() in f"{n.label} {n.token}".lower() for k in keywords)
        ]
        if members:
            supernodes.append({"name": name, "nodes": members})
            used.update(members)
    edges = [list(e) for e in row["expected_edges"]]
    edges += [
        [sn["name"], "output"] for sn in supernodes if sn["name"].startswith("say-")
    ]
    return {"supernodes": supernodes, "edges": edges}


def random_proposal(g: Graph, seed: int) -> dict:
    """Matched-random control: same size and shape, no understanding."""
    feats = [n.id for n in g.feature_nodes if n.share > 0]
    random.Random(seed).shuffle(feats)
    groups = max(2, min(4, len(feats) // MIN_GROUP_SIZE))
    chunks = [feats[i::groups] for i in range(groups)]
    supernodes = [{"name": f"g{i}", "nodes": c} for i, c in enumerate(chunks) if c]
    names = [sn["name"] for sn in supernodes]
    edges = [list(pair) for pair in pairwise(names)]
    edges.append([names[-1], "output"])
    return {"supernodes": supernodes, "edges": edges}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset.jsonl")
    ap.add_argument("--max-features", type=int, default=60)
    args = ap.parse_args()

    rows = [
        json.loads(line)
        for line in (DATA_DIR / args.dataset).read_text().splitlines()
        if line.strip()
    ]
    scorer = InterventionScoring()

    async def score(row, g, kind, proposal):
        try:
            out = await scorer.score(row["prompt"], g, proposal)
        except Exception as exc:
            out = {"reward": 0.0, "gated": f"error: {exc}"}
        return row["family"], kind, out

    jobs = []
    for i, row in enumerate(rows):
        g = slim_graph(load_graph_json(row["graph"]), max_features=args.max_features)
        for kind, proposal in [
            ("expected", expected_proposal(row, g)),
            ("random", random_proposal(g, seed=i)),
        ]:
            jobs.append(score(row, g, kind, proposal))
    results = await asyncio.gather(*jobs)

    by_kind: dict[str, list[float]] = {}
    header = ("family", "proposal", "reward", "sep", "cov", "comp", "gated")
    print(f"{header[0]:<10} {header[1]:<9} " + " ".join(f"{h:>7}" for h in header[2:]))
    for family, kind, out in results:
        by_kind.setdefault(kind, []).append(out["reward"])
        print(
            f"{family:<10} {kind:<9} {out['reward']:>7.3f}"
            f" {out.get('separation', 0):>6.3f} {out.get('coverage', 0):>6.3f}"
            f" {out.get('completeness', 0):>6.3f}  {out.get('gated', '')}"
        )
    print()
    for kind, rewards in by_kind.items():
        print(f"mean {kind:<9} {sum(rewards) / len(rewards):.3f}  (n={len(rewards)})")


if __name__ == "__main__":
    asyncio.run(main())
