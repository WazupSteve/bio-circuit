"""score_archived.py had no coverage at all, unlike every other audit script.

Its aggregate stats are what FINDINGS.md and the blog charts quote, so the
counting logic and the --out path need the same guarantees oracle_audit.py's
`experiment_cost`/`all_rewards` fields already have.
"""

import asyncio
import json

import pytest
from circuit_supernodes.graphs import graph_to_json
from conftest import GRAPH, PROMPT
from score_archived import score_run


class FakeScorer:
    """Scores by a table keyed on the proposal's supernode names, so a test can
    dictate exactly which rollouts pass, fail the gate, or fail to parse."""

    def __init__(self, table):
        self.table = table

    async def score(self, prompt, graph, proposal):
        key = tuple(sn["name"] for sn in proposal["supernodes"])
        return self.table[key]


def trace(idx: int, name: str, text: str) -> dict:
    return {
        "task": {"data": {"idx": idx}},
        "nodes": [{"sampled": True, "message": {"role": "assistant", "content": text}}],
    }


def proposal_text(*names: str) -> str:
    supernodes = [{"name": n, "nodes": ["1_10_1"]} for n in names]
    return f"```json\n{json.dumps({'supernodes': supernodes, 'edges': []})}\n```"


def test_score_run_counts_parse_failures_gates_and_scored_rollouts(tmp_path):
    rows = [{"prompt": PROMPT, "graph": graph_to_json(GRAPH)}]
    traces = [
        trace(0, "ok", proposal_text("A")),
        trace(0, "ok2", proposal_text("B")),
        trace(0, "gated", proposal_text("C")),
        trace(0, "junk", "no json here at all"),
    ]
    path = tmp_path / "traces.jsonl"
    path.write_text("\n".join(json.dumps(t) for t in traces) + "\n")

    scorer = FakeScorer(
        {
            ("A",): {
                "reward": 0.6,
                "separation": 0.5,
                "separation_over_null": 0.1,
            },
            ("B",): {
                "reward": 0.4,
                "separation": 0.3,
                "separation_over_null": 0.05,
            },
            ("C",): {"reward": 0.0, "gated": "catch-all supernode dominates"},
        }
    )

    result = asyncio.run(score_run(scorer, rows, path, max_features=60))

    assert result["scored"] == 2  # A and B; C gated, junk never parsed
    # A parse failure has to count toward the mean like a gate failure does —
    # every rollout in the file is one of "total", full stop. Silently
    # dropping it from the denominator would let a model that produces
    # garbage look better than one that tried and got gated.
    assert result["total"] == 4
    assert result["reward"] == pytest.approx((0.6 + 0.4 + 0.0 + 0.0) / 4)
    assert result["reward_when_valid"] == pytest.approx((0.6 + 0.4) / 2)
    assert result["separation"] == pytest.approx((0.5 + 0.3) / 2)
    assert result["over_null"] == pytest.approx((0.1 + 0.05) / 2)
    assert result["gated"] == {
        "no proposal JSON": 1,
        "catch-all supernode dominates": 1,
    }


def test_out_flag_writes_the_same_results_main_prints(tmp_path, monkeypatch, capsys):
    import score_archived

    rows = [{"prompt": PROMPT, "graph": graph_to_json(GRAPH)}]
    run_dir = tmp_path / "runs" / "demo"
    run_dir.mkdir(parents=True)
    (run_dir / "traces.jsonl").write_text(
        json.dumps(trace(0, "ok", proposal_text("A"))) + "\n"
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "heldout.jsonl").write_text(json.dumps(rows[0]) + "\n")
    out_path = tmp_path / "results.json"

    monkeypatch.setattr(score_archived, "DATA_DIR", data_dir)
    monkeypatch.setattr(
        score_archived,
        "InterventionScoring",
        lambda split_halves=True: FakeScorer(
            {("A",): {"reward": 0.7, "separation": 0.6, "separation_over_null": 0.2}}
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["score_archived.py", "--runs", str(run_dir), "--out", str(out_path)],
    )

    asyncio.run(score_archived.main())

    assert "wrote" in capsys.readouterr().out
    saved = json.loads(out_path.read_text())
    assert len(saved) == 1
    assert saved[0]["run"] == "demo"
    assert saved[0]["reward"] == pytest.approx(0.7)
