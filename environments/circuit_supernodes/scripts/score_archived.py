"""Re-score archived proposals under the current reward.

The oracle audit says what a blind partition scores; this says what a real one
does. Without both numbers on the same graphs, "the attack scores 0.66" means
nothing — the question is always 0.66 *compared to what*.

Doubles as the frontier baseline under the current reward, since the archived
traces are real model output on the held-out graphs.

    uv run python scripts/score_archived.py --runs archive/eval15/*
"""

import argparse
import asyncio
import json
from pathlib import Path
from statistics import mean

from circuit_supernodes.graphs import load_graph_json, slim_graph
from circuit_supernodes.rubric import (
    InterventionScoring,
    completion_text,
    parse_proposal,
)

DATA_DIR = Path(__file__).parent.parent / "circuit_supernodes" / "data"


async def score_run(scorer, rows, path: Path, max_features: int) -> dict:
    kept, gated, rewards, seps, overs = 0, {}, [], [], []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        trace = json.loads(line)
        row = rows[trace["task"]["data"]["idx"]]
        proposal = parse_proposal(completion_text(trace))
        if proposal is None:
            # A rollout that never produced parseable JSON is a failure, not a
            # rollout that didn't happen — it must count toward the mean like
            # a gated one does, or a model that times out mid-thought would
            # look better than one that tried and got gated.
            gated["no proposal JSON"] = gated.get("no proposal JSON", 0) + 1
            rewards.append(0.0)
            continue
        graph = slim_graph(load_graph_json(row["graph"]), max_features)
        result = await scorer.score(row["prompt"], graph, proposal)
        if "gated" in result:
            reason = result["gated"].split(";")[0].split(":")[0]
            gated[reason] = gated.get(reason, 0) + 1
            rewards.append(0.0)
            continue
        kept += 1
        rewards.append(result["reward"])
        seps.append(result["separation"])
        if result.get("separation_over_null") is not None:
            overs.append(result["separation_over_null"])
    return {
        "run": path.parent.name,
        "scored": kept,
        "total": len(rewards),
        "reward": mean(rewards) if rewards else 0.0,
        "reward_when_valid": mean(r for r in rewards if r > 0) if kept else 0.0,
        "separation": mean(seps) if seps else 0.0,
        "over_null": mean(overs) if overs else 0.0,
        "gated": gated,
    }


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--runs", nargs="+", type=Path, help="directories holding traces.jsonl"
    )
    ap.add_argument("--dataset", default="heldout.jsonl")
    ap.add_argument("--max-features", type=int, default=60)
    ap.add_argument("--v1", action="store_true", help="score without the hidden split")
    ap.add_argument("--out", type=Path, help="write per-run results as JSON")
    args = ap.parse_args()

    rows = [
        json.loads(line)
        for line in (DATA_DIR / args.dataset).read_text().splitlines()
        if line.strip()
    ]
    scorer = InterventionScoring(split_halves=not args.v1)
    results = await asyncio.gather(
        *(
            score_run(scorer, rows, run / "traces.jsonl", args.max_features)
            for run in args.runs
            if (run / "traces.jsonl").exists()
        )
    )

    version = "v1 (whole-group)" if args.v1 else "v2 (hidden split)"
    print(f"{len(results)} runs, reward {version}\n")
    print(
        f"{'run':<12}{'scored':>8}{'reward':>9}{'if valid':>10}"
        f"{'separation':>12}{'over null':>11}"
    )
    for r in sorted(results, key=lambda r: -r["reward"]):
        print(
            f"{r['run']:<12}{r['scored']}/{r['total']:<6}{r['reward']:>9.3f}"
            f"{r['reward_when_valid']:>10.3f}{r['separation']:>12.3f}"
            f"{r['over_null']:>11.3f}"
        )
    for r in results:
        if r["gated"]:
            print(f"\n{r['run']} rejections: {r['gated']}")

    if args.out:
        args.out.write_text(json.dumps(results, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
