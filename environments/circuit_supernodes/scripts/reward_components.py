"""Recover per-component reward means from archived eval traces.

The v1 reward is ``separation^0.6 * coverage^0.2 * completeness^0.2``. Grouping
every displayed feature makes coverage and completeness exactly 1.0 by
construction, so this table shows how much of a model's reward came from the
causal term and how much came from the two degenerate ones.

    uv run python scripts/reward_components.py archive/eval15/*/traces.jsonl
"""

import argparse
import json
from pathlib import Path
from statistics import mean

COMPONENTS = ("separation", "coverage", "completeness")
REWARD = "circuit_faithfulness"


def reward_value(raw) -> float:
    # Verifiers records rewards as {reward_name: value}; older traces stored a
    # bare float.
    if isinstance(raw, (int, float)):
        return float(raw)
    if REWARD in raw:
        return float(raw[REWARD])
    if len(raw) == 1:
        return float(next(iter(raw.values())))
    raise ValueError(f"cannot pick a reward from {sorted(raw)}")


def read_traces(path: Path) -> tuple[list[float], list[dict]]:
    rewards, components = [], []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        trace = json.loads(line)
        rewards.append(reward_value(trace.get("rewards")))
        parts = (trace.get("info") or {}).get("reward_components")
        if parts and all(k in parts for k in COMPONENTS):
            components.append(parts)
    return rewards, components


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("traces", nargs="+", type=Path, help="traces.jsonl files")
    args = ap.parse_args()

    header = ("run", "n", "scored", *COMPONENTS, "reward")
    print(f"{header[0]:<28}" + "".join(f"{h:>14}" for h in header[1:]))
    for path in args.traces:
        rewards, components = read_traces(path)
        if not rewards:
            print(f"{path.parent.name:<28}{'no rollouts':>14}")
            continue
        cells = [
            f"{mean(c[k] for c in components):>14.3f}" if components else f"{'-':>14}"
            for k in COMPONENTS
        ]
        print(
            f"{path.parent.name:<28}{len(rewards):>14}{len(components):>14}"
            + "".join(cells)
            + f"{mean(rewards):>14.3f}"
        )
    print(
        "\nA run with coverage == completeness == 1.000 grouped every displayed "
        "feature;\nits reward is separation^0.6 and nothing else."
    )


if __name__ == "__main__":
    main()
