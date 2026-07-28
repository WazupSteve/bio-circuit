"""Red-team the reward: score scripted attacks under v1 and v2 and compare.

The v1 reward can be pre-measured — seven experiments are enough to recover the
whole matrix the scorer will compute — so an attacker who never looks at what the
features *mean* can transcribe a high-scoring circuit. Every attack here does
exactly that: it picks a partition without understanding it, measures the true
effect matrix (unlimited measurement, an upper bound on any budgeted agent), fits
the best reachability claim that measurement allows, and reports the score.

v2 must beat these. If it does not, the reward is not measuring circuit discovery
and the audit itself is the result.

    uv run python scripts/oracle_audit.py --dataset heldout.jsonl --rows 8
"""

import argparse
import asyncio
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from circuit_supernodes.graphs import Graph, load_graph_json, slim_graph
from circuit_supernodes.rubric import (
    MAX_GROUPS,
    MIN_GROUP_SIZE,
    OUTPUT,
    InterventionScoring,
)

DATA_DIR = Path(__file__).parent.parent / "circuit_supernodes" / "data"
THRESHOLD_STEPS = 12  # edge-fit sweep; free after the partition's one intervention


def _ranked_features(g: Graph) -> list[str]:
    return [
        n.id for n in sorted(g.feature_nodes, key=lambda n: -n.share) if n.share > 0
    ]


def _name(chunks: list[list[str]]) -> dict[str, list[str]]:
    return {f"g{i}": c for i, c in enumerate(chunks) if c}


def _rebalance(chunks: list[list[str]], k: int, min_size: int) -> list[list[str]]:
    """Coerce arbitrary clusters into k groups that can pass the membership gate."""
    chunks = sorted((c for c in chunks if c), key=len, reverse=True)
    keep, spill = chunks[:k], [f for c in chunks[k:] for f in c]
    if not keep:
        return []
    for i, feature in enumerate(spill):
        keep[i % len(keep)].append(feature)
    # Top up anything under the gate from the largest group that can spare members.
    for group in keep:
        while len(group) < min_size:
            donor = max(keep, key=len)
            if donor is group or len(donor) <= min_size:
                return [g for g in keep if len(g) >= min_size]
            group.append(donor.pop())
    return keep


def blind_balanced(g: Graph, k: int, min_size: int) -> dict[str, list[str]]:
    """Balanced groups by influence rank. Never inspects labels, tokens or edges."""
    feats = _ranked_features(g)
    return _name(_rebalance([feats[i::k] for i in range(k)], k, min_size))


def community(g: Graph, k: int, min_size: int) -> dict[str, list[str]]:
    """Weighted label propagation over the attribution edges — structure only.

    Stands in for the 'simple attribution methods are embarrassingly strong'
    baseline: if this matches a trained agent, the graph carries the information.
    """
    feats = _ranked_features(g)
    keep = set(feats)
    neighbours: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for e in g.edges:
        if e.source in keep and e.target in keep and e.source != e.target:
            neighbours[e.source].append((e.target, abs(e.weight)))
            neighbours[e.target].append((e.source, abs(e.weight)))
    labels = dict(zip(feats, range(len(feats)), strict=True))
    for _ in range(20):
        changed = False
        for f in feats:
            if not neighbours[f]:
                continue
            tally: Counter = Counter()
            for other, weight in neighbours[f]:
                tally[labels[other]] += weight
            best = tally.most_common(1)[0][0]
            if best != labels[f]:
                labels[f], changed = best, True
        if not changed:
            break
    clusters: dict[int, list[str]] = defaultdict(list)
    for f in feats:
        clusters[labels[f]].append(f)
    return _name(_rebalance(list(clusters.values()), k, min_size))


def random_partition(
    g: Graph, k: int, min_size: int, seed: int
) -> dict[str, list[str]]:
    feats = _ranked_features(g)
    random.Random(seed).shuffle(feats)
    return _name(_rebalance([feats[i::k] for i in range(k)], k, min_size))


def smallest_legal(g: Graph, min_size: int) -> dict[str, list[str]]:
    """As many minimum-size groups as the gate allows, taken off the top of the
    influence ranking. This is the attack the hidden split is most exposed to: a
    group of m admits only C(m, m//2)/2 halvings, so at small m the 'hidden' seed
    hides nothing and the attacker knows exactly what will be ablated.
    """
    feats = _ranked_features(g)
    k = min(MAX_GROUPS, len(feats) // min_size)
    return _name([feats[i * min_size : (i + 1) * min_size] for i in range(k)])


def fit_edges(
    measurement: dict, g: Graph, sns: dict, threshold: float
) -> list[list[str]]:
    """Best reachability claim the measurement supports at this threshold.

    Edges only run forward through the groups' mean layer, so the result is a DAG
    by construction; any group left without a path is wired straight to output so
    the proposal clears the topology gate.
    """
    names = measurement["names"]
    depth = {n: sum(g.nodes[m].layer for m in sns[n]) / len(sns[n]) for n in names}
    order = sorted(range(len(names)), key=lambda i: depth[names[i]])
    rank = {i: r for r, i in enumerate(order)}

    edges = [
        [names[i], names[j]]
        for i in range(len(names))
        for j in range(len(names))
        if rank[i] < rank[j] and measurement["groups"][i][j] >= threshold
    ]
    edges += [
        [names[i], OUTPUT]
        for i in range(len(names))
        if measurement["target"][i] >= threshold
    ]
    adj: dict[str, set[str]] = defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
    for name in names:
        seen, stack = set(), list(adj[name])
        while stack:
            n = stack.pop()
            if n not in seen:
                seen.add(n)
                stack.extend(adj.get(n, ()))
        if OUTPUT not in seen:
            edges.append([name, OUTPUT])
    return edges


async def best_score(
    scorer: InterventionScoring, prompt: str, g: Graph, sns: dict
) -> dict:
    """Measure once, then sweep edge sets. The scorer caches per partition, so the
    whole sweep costs one intervention — which is the point: edge choice is free
    to an attacker who already knows the effects."""
    if len(sns) < 2:
        return {"reward": 0.0, "gated": "attack produced too few groups"}
    measurement = await scorer.measure(prompt, g, sns)
    values = sorted(
        {v for row in measurement["groups"] for v in row} | set(measurement["target"])
    )
    step = max(1, len(values) // THRESHOLD_STEPS)
    supernodes = [{"name": n, "nodes": ms} for n, ms in sns.items()]
    best = {"reward": 0.0, "gated": "no threshold produced a valid proposal"}
    for threshold in values[::step] or [0.0]:
        proposal = {
            "supernodes": supernodes,
            "edges": fit_edges(measurement, g, sns, threshold),
        }
        result = await scorer.score(prompt, g, proposal)
        if result["reward"] >= best["reward"]:
            best = {**result, "threshold": threshold}
    return best


def attacks(g: Graph, k: int, min_size: int, multistart: int) -> dict[str, list[dict]]:
    """Each attack is a list of partitions; the audit reports the best of each."""
    return {
        "blind-balanced": [blind_balanced(g, k, min_size)],
        "community": [community(g, k, min_size)],
        "random-multistart": [
            random_partition(g, k, min_size, seed) for seed in range(multistart)
        ],
        "smallest-legal": [smallest_legal(g, min_size)],
    }


async def audit_row(row: dict, args) -> dict:
    g = slim_graph(load_graph_json(row["graph"]), args.max_features)
    out: dict = {"family": row["family"], "prompt": row["prompt"], "scores": {}}
    for version, split_halves in (("v1", False), ("v2", True)):
        scorer = InterventionScoring(
            split_halves=split_halves, min_group_size=args.min_group_size
        )
        for attack, partitions in attacks(
            g, args.groups, args.min_group_size, args.multistart
        ).items():
            results = [
                await best_score(scorer, row["prompt"], g, sns) for sns in partitions
            ]
            out["scores"][f"{version}/{attack}"] = max(
                results, key=lambda r: r["reward"]
            )
    return out


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="heldout.jsonl")
    ap.add_argument("--rows", type=int, default=8, help="graphs to audit")
    ap.add_argument("--groups", type=int, default=5)
    ap.add_argument("--min-group-size", type=int, default=MIN_GROUP_SIZE)
    ap.add_argument("--multistart", type=int, default=16)
    ap.add_argument("--max-features", type=int, default=60)
    ap.add_argument("--out", type=Path, help="write per-row results as JSON")
    args = ap.parse_args()

    rows = [
        json.loads(line)
        for line in (DATA_DIR / args.dataset).read_text().splitlines()
        if line.strip()
    ][: args.rows]
    if not rows:
        raise SystemExit(f"no rows in {args.dataset}")

    results = await asyncio.gather(*(audit_row(row, args) for row in rows))

    keys = list(results[0]["scores"])
    print(f"{len(rows)} graphs, min group size {args.min_group_size}\n")
    print(f"{'attack':<28}{'reward':>9}{'separation':>12}{'gated':>8}")
    for key in keys:
        scores = [r["scores"][key] for r in results]
        rewards = [s["reward"] for s in scores]
        seps = [s["separation"] for s in scores if "separation" in s]
        gated = sum("gated" in s for s in scores)
        sep = f"{sum(seps) / len(seps):>12.3f}" if seps else f"{'-':>12}"
        print(f"{key:<28}{sum(rewards) / len(rewards):>9.3f}{sep}{gated:>8}")
    print("\nv2 must score below v1 on every attack, or the split is not binding.")

    if args.out:
        args.out.write_text(json.dumps(results, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
