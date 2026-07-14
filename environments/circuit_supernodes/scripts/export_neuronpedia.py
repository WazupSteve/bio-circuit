import argparse
import json
from pathlib import Path

from circuit_supernodes.graphs import embed_supernodes, load_graph_json, slim_graph
from circuit_supernodes.rubric import (
    MAX_GROUPS,
    completion_text,
    parse_proposal,
    valid_supernodes,
)

DATA_DIR = Path(__file__).parent.parent / "circuit_supernodes" / "data"


def read_proposal(path: Path) -> dict:
    text = path.read_text()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = parse_proposal(text)
    if isinstance(data, dict) and "completion" in data:
        data = parse_proposal(completion_text(data))
    if not isinstance(data, dict) or not isinstance(data.get("supernodes"), list):
        raise SystemExit(f"no proposal found in {path}")
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset.jsonl")
    ap.add_argument("--index", type=int, default=0, help="dataset row to export")
    ap.add_argument("--proposal", required=True, type=Path)
    ap.add_argument("--slug", default="", help="graph slug shown in the UI")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    lines = (DATA_DIR / args.dataset).read_text().splitlines()
    row = json.loads([ln for ln in lines if ln.strip()][args.index])
    data = read_proposal(args.proposal)
    sns = valid_supernodes(data, slim_graph(load_graph_json(row["graph"])))
    sns = dict(list(sns.items())[:MAX_GROUPS])
    if not sns:
        raise SystemExit("proposal has no valid supernodes for this graph")

    out = args.out or Path(f"{args.slug or 'circuit'}-{args.index}.json")
    out.write_text(json.dumps(embed_supernodes(row["graph"], sns, args.slug)))
    print(f"{row['family']}: {len(sns)} supernodes -> {out}")
    print("upload at https://www.neuronpedia.org/graph/validator")


if __name__ == "__main__":
    main()
