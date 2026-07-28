"""Red-team the reward: score scripted attacks under v1 and v2 and compare.

The v1 reward can be pre-measured — seven experiments are enough to recover the
whole matrix the scorer will compute — so an attacker who never looks at what the
features *mean* can transcribe a high-scoring circuit. Every attack here does
exactly that: it picks a partition without understanding it, measures the true
effect matrix, fits the best reachability claim that measurement allows, and
reports the score.

The ``exps`` column says whether that is hypothetical. Five groups of twelve cost
five experiments against a budget of seven, so the blind attacks are not oracle
fantasies — they are strategies a policy can run with budget to spare. Only the
per-feature signatures behind ``coherence-matched`` are genuinely unaffordable.

v2 must beat these. If it does not, the reward is not measuring circuit discovery
and the audit itself is the result.

    uv run python scripts/oracle_audit.py --dataset heldout.jsonl --rows 8
"""

import argparse
import asyncio
import json
import random
from collections import Counter, defaultdict
from math import comb
from pathlib import Path
from statistics import mean, pstdev

from circuit_supernodes.experiments import (
    DEFAULT_BUDGET,
    MAX_ABLATED_FEATURES,
    MAX_PROBE_FEATURES,
    MAX_PROBE_GROUPS,
)
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
    if len(clusters) < 2:
        # Dense attribution graphs collapse to one label. Fall back to layer
        # bands, which are still structure-only and never degenerate — without
        # this the attack silently reports "gated" and looks harmless.
        ordered = sorted(feats, key=lambda f: (g.nodes[f].layer, g.nodes[f].ctx_idx))
        size = max(min_size, len(ordered) // k)
        clusters = dict(
            enumerate(ordered[i : i + size] for i in range(0, len(ordered), size))
        )
    return _name(_rebalance(list(clusters.values()), k, min_size))


def random_partition(
    g: Graph, k: int, min_size: int, seed: int
) -> dict[str, list[str]]:
    feats = _ranked_features(g)
    random.Random(seed).shuffle(feats)
    return _name(_rebalance([feats[i::k] for i in range(k)], k, min_size))


def smallest_legal(g: Graph, min_size: int) -> dict[str, list[str]]:
    """As many minimum-size groups as the gate allows, taken off the top of the
    influence ranking.

    Intended to test whether small groups make the split guessable; measured, it
    does the opposite — blind small groups are punished *harder*, because two
    features picked off an influence ranking are never interchangeable. The
    attack that actually tests the split is ``coherence_matched``.
    """
    feats = _ranked_features(g)
    k = min(MAX_GROUPS, len(feats) // min_size)
    return _name([feats[i * min_size : (i + 1) * min_size] for i in range(k)])


def budget_feasible(g: Graph, min_size: int) -> dict[str, list[str]]:
    """The best blind partition a *policy* can actually measure.

    Every other attack here partitions all the displayed features, which stops
    being executable once the slice exceeds MAX_GROUPS x MAX_ABLATED_FEATURES. A
    policy would not do that: it would group as many of the top features as its
    budget and the per-call ablation limit allow, leave the rest ungrouped, and
    pay the coverage penalty. This measures what that costs, which is the number
    that decides whether widening the slice is worth its prompt tokens.
    """
    feats = _ranked_features(g)
    k = min(MAX_GROUPS, DEFAULT_BUDGET, max(2, len(feats) // min_size))
    size = min(MAX_ABLATED_FEATURES, max(min_size, -(-len(feats) // k)))
    return _name([feats[i * size : (i + 1) * size] for i in range(k)])


def coherence_matched(
    g: Graph, signatures: dict[str, float], k: int, min_size: int
) -> dict[str, list[str]]:
    """Group features by measured per-feature effect, so members are causally
    interchangeable and *both halves of any split look alike*.

    This is the attack the minimum-size gate was introduced to stop. Knowing the
    split only helps an attacker who can also satisfy it, and matching members by
    effect magnitude is how you satisfy it — which is why the blind
    ``smallest_legal`` attack does not test the gate at all.
    """
    ranked = sorted(signatures, key=lambda f: -signatures[f])
    size = max(min_size, len(ranked) // k)
    chunks = [ranked[i * size : (i + 1) * size] for i in range(k)]
    return _name([c for c in chunks if len(c) >= min_size])


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


def experiment_cost(sns: dict) -> int | None:
    """Experiments a *policy* needs to reconstruct this partition's effect matrix,
    under the real tool limits — or None if the tool cannot express it.

    This is what separates a hypothetical oracle from a reachable strategy. Five
    groups of twelve costs five experiments against a budget of seven, so the
    blind attacks below are not upper bounds an agent could never reach: they are
    strategies a policy can execute and have budget left over.
    """
    sizes = [len(m) for m in sns.values()]
    if len(sizes) < 2 or max(sizes) > MAX_ABLATED_FEATURES:
        return None
    if len(sns) - 1 > MAX_PROBE_GROUPS or sum(sizes) - min(sizes) > MAX_PROBE_FEATURES:
        return None
    return len(sns)


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


def attacks(
    g: Graph, k: int, min_size: int, multistart: int, signatures: dict | None = None
) -> dict[str, list[dict]]:
    """Each attack is a list of partitions; the audit reports the best of each."""
    suite = {
        "blind-balanced": [blind_balanced(g, k, min_size)],
        "community": [community(g, k, min_size)],
        "random-multistart": [
            random_partition(g, k, min_size, seed) for seed in range(multistart)
        ],
        "smallest-legal": [smallest_legal(g, min_size)],
        "budget-feasible": [budget_feasible(g, min_size)],
    }
    if signatures is not None:
        suite["coherence-matched"] = [
            coherence_matched(g, signatures, k, min_size),
            coherence_matched(g, signatures, MAX_GROUPS, min_size),
        ]
    return suite


async def audit_row(row: dict, args) -> dict:
    g = slim_graph(load_graph_json(row["graph"]), args.max_features)
    out: dict = {"family": row["family"], "prompt": row["prompt"], "scores": {}}
    for version, split_halves in (("v1", False), ("v2", True)):
        scorer = InterventionScoring(
            split_halves=split_halves, min_group_size=args.min_group_size
        )
        signatures = (
            None
            if args.no_signatures
            else await scorer.measure_features(row["prompt"], g)
        )
        for attack, partitions in attacks(
            g, args.groups, args.min_group_size, args.multistart, signatures
        ).items():
            results = [
                await best_score(scorer, row["prompt"], g, sns) for sns in partitions
            ]
            best = max(results, key=lambda r: r["reward"])
            costs = [experiment_cost(sns) for sns in partitions]
            affordable = [c for c in costs if c is not None and c <= DEFAULT_BUDGET]
            out["scores"][f"{version}/{attack}"] = {
                **best,
                "experiment_cost": min(affordable) if affordable else None,
                # Every partition, not just the winner: reporting only the max
                # conflates "this attack works" with "trying N times works", and
                # a GRPO group of G rollouts is a search of size G.
                "all_rewards": [r["reward"] for r in results],
            }
    return out


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="heldout.jsonl")
    ap.add_argument("--rows", type=int, default=8, help="graphs to audit")
    ap.add_argument("--groups", type=int, default=5)
    ap.add_argument("--min-group-size", type=int, default=MIN_GROUP_SIZE)
    ap.add_argument("--multistart", type=int, default=16)
    ap.add_argument("--max-features", type=int, default=60)
    ap.add_argument(
        "--no-signatures",
        action="store_true",
        help="skip the coherence-matched attack (saves one intervention per graph)",
    )
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
    print(f"{'attack':<28}{'reward':>9}{'separation':>12}{'gated':>7}{'exps':>7}")
    for key in keys:
        scores = [r["scores"][key] for r in results]
        rewards = [s["reward"] for s in scores]
        seps = [s["separation"] for s in scores if "separation" in s]
        gated = sum("gated" in s for s in scores)
        costs = [s.get("experiment_cost") for s in scores]
        cheapest = [c for c in costs if c is not None]
        sep = f"{sum(seps) / len(seps):>12.3f}" if seps else f"{'-':>12}"
        cost = f"{min(cheapest):>7}" if cheapest else f"{'n/a':>7}"
        print(f"{key:<28}{sum(rewards) / len(rewards):>9.3f}{sep}{gated:>7}{cost}")
    print(
        f"\nv2 must score below v1 on every attack, or the split is not binding."
        f"\n'exps' is what a policy would spend to measure the same matrix"
        f" (budget {DEFAULT_BUDGET}); 'n/a' means the tool cannot express it."
    )
    _report_blind_baseline(results)

    if args.out:
        args.out.write_text(json.dumps(results, indent=1))
        print(f"\nwrote {args.out}")


def _report_blind_baseline(results: list[dict]) -> None:
    """What a random partition scores, and whether a good one can be told apart.

    Quoting the best of N restarts against a single-shot policy compares a search
    to one draw. GRPO scores a rollout against its own group's *mean*, so that is
    the number a training run actually competes with, and the spread around it is
    what decides whether the gradient is signal or luck.
    """
    draws = [
        r["scores"]["v2/random-multistart"].get("all_rewards")
        for r in results
        if r["scores"].get("v2/random-multistart", {}).get("all_rewards")
    ]
    if not draws:
        return
    n = len(draws[0])
    means = [mean(s) for s in draws]
    sds = [pstdev(s) for s in draws]
    print(f"\nblind partition baseline ({len(draws)} graphs x {n} restarts)")
    print(f"{'k':>4}{'E[best of k]':>15}")
    for k in (1, 2, 4, 8, 16):
        if k > n:
            break
        expected = [
            sum(sorted(s)[i] * comb(i, k - 1) for i in range(k - 1, n)) / comb(n, k)
            for s in draws
        ]
        print(f"{k:>4}{mean(expected):>15.3f}")
    print(
        f"  single-draw mean {mean(means):.3f}, within-graph sd {mean(sds):.3f}"
        f"\n  a proposal must beat the mean by ~{2 * mean(sds) / 8**0.5:.3f}"
        f" to clear 2 se at a group of 8"
    )
    saturated = [
        (r["family"], m) for r, m in zip(results, means, strict=True) if m > 0.6
    ]
    if saturated:
        print(
            f"  {len(saturated)} graph(s) where a RANDOM partition already scores"
            f" > 0.6 and so carry little signal: {saturated}"
        )


if __name__ == "__main__":
    asyncio.run(main())
