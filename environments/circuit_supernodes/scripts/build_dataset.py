"""Generate the synthetic dataset: sample prompts, compute attribution graphs on Modal,
    filter faithful outputs, and write circuit_supernodes/data/dataset.jsonl.

Usage:
    uv run python scripts/build_dataset.py --n 16 --seed 0
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ENV_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ENV_ROOT))

from circuit_supernodes.datagen import generate  # noqa: E402
from circuit_supernodes.graphs import load_graph_json  # noqa: E402


def is_faithful(graph_json: dict, target: str) -> tuple[bool, str]:
    """Does the subject model's top predicted token match the intended target?
    Qwen tokenizes digits individually, so the top token is just the first piece of the target
    (example :'9' for target '95')
    """
    g = load_graph_json(graph_json)
    if not g.top_outputs:
        return False, ""
    top_tok = g.top_outputs[0][0]
    t, tt = target.strip(), top_tok.strip()
    return bool(tt) and (t.startswith(tt) or tt.startswith(t)), top_tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out", default=str(ENV_ROOT / "circuit_supernodes" / "data" / "dataset.jsonl")
    )
    args = ap.parse_args()

    samples = generate(args.n, seed=args.seed)
    prompts = [s.prompt for s in samples]
    print(
        f"Sampled {len(samples)} prompts across families:",
        sorted({s.family for s in samples}),
    )

    payload = json.dumps(prompts)
    req = ENV_ROOT / "modal_app" / "_prompts.json"
    req.write_text(payload)
    # Run attribution remotely; modal_app/attribution.py reads _prompts.json and writes _graphs.json next to it.
    subprocess.run(
        [
            "uv",
            "run",
            "modal",
            "run",
            str(ENV_ROOT / "modal_app" / "attribution.py"),
            "--prompts-file",
            str(req),
        ],
        check=True,
        cwd=str(ENV_ROOT.parent.parent),
    )
    graphs = json.loads((ENV_ROOT / "modal_app" / "_graphs.json").read_text())
    assert len(graphs) == len(samples), (
        f"{len(graphs)} graphs != {len(samples)} samples"
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    kept, stats = 0, {}
    with out.open("w") as f:
        for s, graph in zip(samples, graphs):
            if graph is None:
                print(f"  ! attribution failed for {s.prompt!r}, skipping")
                continue
            faithful, top_tok = is_faithful(graph, s.target)
            stats.setdefault(s.family, []).append(faithful)
            if not faithful:
                print(
                    f"  ! unfaithful [{s.family}] {s.prompt!r}: "
                    f"model says {top_tok!r}, wanted {s.target!r} — skipping"
                )
                continue
            row = s.to_row()
            row["graph"] = graph
            f.write(json.dumps(row) + "\n")
            kept += 1
    print(
        "faithfulness by family:",
        {k: f"{sum(v)}/{len(v)}" for k, v in sorted(stats.items())},
    )
    print(f"Wrote {kept} rows -> {out}")


if __name__ == "__main__":
    main()
