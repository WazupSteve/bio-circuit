"""Does the reward prefer the TRUE partition over a random one? Currently: no.

The oracle audit shows blind partitions scoring well on real graphs, but it
cannot say what the right answer was. This does: it builds a synthetic subject
whose causal structure we fix by construction, hands the scorer the true feature
grouping and then a shuffled one, and gives both the best edge set an exhaustive
DAG search can find.

Needs no Modal and no dataset — the subject model is the simulator below, so this
is checkable from a clean checkout in seconds.

    uv run python scripts/partition_discrimination.py

The mechanism behind the result: separation rewards a group whose activation
responds to many ablations in a graded way, because that yields a rich, cleanly
ordered set of observations. A functionally *pure* group responds sparsely and
near-binarily; a random group is a mixture and responds to everything a little.
So the metric has an active bias toward mixed groups, which is the opposite of
what a supernode reward is supposed to reward, and no change to how the group is
split can repair it.
"""

import argparse
import asyncio
import itertools
import random

from circuit_supernodes.graphs import load_graph_json
from circuit_supernodes.rubric import OUTPUT, InterventionScoring

BASE_PROB, BASE_ACT = 0.8, 2.0


def build(roles: int, per_role: int):
    """A chain of roles: r0 -> r1 -> ... -> output, plus one inert role."""
    groups = {
        f"r{r}": [f"1_{r * per_role + i}_1" for i in range(per_role)]
        for r in range(roles)
    }
    features = [f for ms in groups.values() for f in ms]
    drives = {f"r{r}": {f"r{r + 1}"} for r in range(roles - 2)}
    drives[f"r{roles - 2}"] = {OUTPUT}
    drives[f"r{roles - 1}"] = set()  # inert distractor
    graph = load_graph_json(
        {
            "format": "slim",
            "tokens": ["a"],
            "top_outputs": [[" x", 0.8]],
            "nodes": [
                {
                    "id": f,
                    "kind": "feature",
                    "layer": 1,
                    "ctx_idx": 1,
                    "token": "t",
                    "influence": 0.0,
                    "share": 1.0 / len(features),
                    "label": "",
                }
                for f in features
            ],
            "edges": [],
        }
    )
    return groups, features, drives, graph


def simulator(roles: dict, drives: dict, proposal: dict):
    """Effects follow the FEATURES' true roles, never the agent's grouping."""
    role_of = {f: r for r, ms in roles.items() for f in ms}
    names = list(proposal)

    async def intervene(prompt, sent, target, complement, indices=None):
        ids = [[f"{a}_{b}_{c}" for a, b, c in grp] for grp in sent]
        ablations = []
        for i in range(len(sent)) if indices is None else indices:
            killed = set(ids[i])
            removed = {r: len(killed & set(ms)) / len(ms) for r, ms in roles.items()}
            downstream: dict[str, float] = {}
            for r, targets in drives.items():
                for t in targets:
                    downstream[t] = downstream.get(t, 0.0) + removed[r]
            acts = []
            for j in range(len(sent)):
                if j >= len(names):
                    acts.append(BASE_ACT)
                    continue
                members = proposal[names[j]]
                lost = sum(
                    min(downstream.get(role_of[m], 0.0), 1.0) for m in members
                ) / len(members)
                direct = len(killed & set(members)) / len(members)
                acts.append(BASE_ACT * (1 - min(1.0, lost + direct)))
            ablations.append(
                {
                    "target_prob": BASE_PROB
                    * (1 - min(1.0, downstream.get(OUTPUT, 0.0))),
                    "group_acts": acts,
                }
            )
        return {
            "base_prob": BASE_PROB,
            "base_group_acts": [BASE_ACT] * len(sent),
            "ablations": ablations,
            "complement_prob": None,
        }

    return intervene


async def score(graph, roles, drives, sns, edges, split_halves) -> float:
    scorer = InterventionScoring(split_halves=split_halves)
    scorer._intervene = simulator(roles, drives, sns)
    result = await scorer.score(
        "p",
        graph,
        {
            "supernodes": [{"name": n, "nodes": ms} for n, ms in sns.items()],
            "edges": edges,
        },
    )
    return result["reward"]


async def best_edges(graph, roles, drives, sns, split_halves) -> float:
    """Exhaustive over DAGs: the most generous edge fit an attacker could buy."""
    top = 0.0
    for order in itertools.permutations(sns):
        pairs = [(a, b) for i, a in enumerate(order) for b in [*order[i + 1 :], OUTPUT]]
        for bits in range(1 << len(pairs)):
            edges = [list(p) for k, p in enumerate(pairs) if bits >> k & 1]
            top = max(top, await score(graph, roles, drives, sns, edges, split_halves))
    return top


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roles", type=int, default=4)
    ap.add_argument("--per-role", type=int, default=6)
    ap.add_argument("--shuffles", type=int, default=4)
    args = ap.parse_args()

    roles, features, drives, graph = build(args.roles, args.per_role)
    true_edges = [[a, b] for a, ts in drives.items() for b in ts] + [
        [f"r{args.roles - 1}", OUTPUT]
    ]

    print(f"{args.roles} roles x {args.per_role} features, exhaustive edge search\n")
    print(f"{'partition':<34}{'v1':>8}{'v2':>8}")
    honest = await score(graph, roles, drives, roles, true_edges, True)
    print(f"{'true partition, TRUE edges':<34}{'-':>8}{honest:>8.3f}")
    for label, sns in [("true partition, best edges", dict(roles))] + [
        (
            f"blind shuffle, seed {seed}",
            _shuffle(features, len(roles), seed),
        )
        for seed in range(args.shuffles)
    ]:
        v1 = await best_edges(graph, roles, drives, sns, False)
        v2 = await best_edges(graph, roles, drives, sns, True)
        print(f"{label:<34}{v1:>8.3f}{v2:>8.3f}")
    print(
        "\nA blind shuffle scoring at or above the true partition means the reward"
        "\nis not measuring circuit discovery, and no change to the split repairs it."
    )


def _shuffle(features: list[str], k: int, seed: int) -> dict[str, list[str]]:
    shuffled = features[:]
    random.Random(seed).shuffle(shuffled)
    size = len(shuffled) // k
    return {f"g{i}": shuffled[i * size : (i + 1) * size] for i in range(k)}


if __name__ == "__main__":
    asyncio.run(main())
