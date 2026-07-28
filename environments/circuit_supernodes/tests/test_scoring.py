import asyncio
import inspect
import json
from types import SimpleNamespace

import pytest
import verifiers.v1 as vf
from circuit_supernodes.circuit_supernodes import CircuitTask, CircuitTaskData
from circuit_supernodes.experiments import (
    MAX_EXPERIMENTS,
    AblationTools,
    CircuitState,
)
from circuit_supernodes.graphs import graph_to_json, load_graph_json, slim_graph
from circuit_supernodes.rubric import (
    InterventionScoring,
    completion_text,
    parse_proposal,
    valid_supernodes,
)

FEATURES = {
    "1_10_1": 0.25,
    "1_11_1": 0.15,
    "2_20_2": 0.20,
    "2_21_2": 0.10,
    "3_30_3": 0.20,
    "3_31_3": 0.10,
}


def make_graph() -> dict:
    nodes = [
        {
            "id": nid,
            "kind": "feature",
            "layer": int(nid.split("_")[0]),
            "ctx_idx": int(nid.split("_")[2]),
            "token": "t",
            "influence": 0.0,
            "share": share,
            "label": f"feature {nid}",
        }
        for nid, share in FEATURES.items()
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
        "edges": [["1_10_1", "2_20_2", 1.5], ["2_20_2", "3_30_3", 1.2]],
    }


GRAPH = load_graph_json(make_graph())

PROPOSAL = {
    "supernodes": [
        {"name": "A", "nodes": ["1_10_1", "1_11_1"]},
        {"name": "B", "nodes": ["2_20_2", "2_21_2"]},
        {"name": "C", "nodes": ["3_30_3", "3_31_3"]},
    ],
    "edges": [["A", "B"], ["B", "C"], ["C", "output"]],
}

CHAIN_RESULT = {
    "base_prob": 0.8,
    "base_group_acts": [2.0, 2.0, 2.0],
    "ablations": [
        {"target_prob": 0.05, "group_acts": [0.0, 0.1, 0.1]},  # ablate A
        {"target_prob": 0.05, "group_acts": [2.0, 0.0, 0.1]},  # ablate B
        {"target_prob": 0.05, "group_acts": [2.0, 2.0, 0.0]},  # ablate C
    ],
    "complement_prob": 0.78,
}


def score(proposal: dict, result: dict, graph=GRAPH) -> dict:
    scorer = InterventionScoring()

    async def fake_intervene(prompt, groups, target, complement):
        return result

    scorer._intervene = fake_intervene
    return asyncio.run(scorer.score("Fact: test prompt is", graph, proposal))


def test_parse_fenced_and_bare():
    data = {"supernodes": [{"name": "A", "nodes": ["1_10_1"]}]}
    fenced = {
        "completion": [
            {"role": "assistant", "content": f"x\n```json\n{json.dumps(data)}\n```"}
        ]
    }
    assert parse_proposal(completion_text(fenced)) == data
    assert parse_proposal(f"analysis {json.dumps(data)} done") == data
    assert parse_proposal("no json") is None


def test_completion_text_reads_verifiers_v1_trace_nodes():
    data = {"supernodes": [{"name": "A", "nodes": ["1_10_1"]}]}
    trace = {
        "nodes": [
            {
                "sampled": False,
                "message": {"role": "user", "content": "find the circuit"},
            },
            {
                "sampled": True,
                "message": {"role": "assistant", "content": "first hypothesis"},
            },
            {
                "sampled": False,
                "message": {"role": "tool", "content": "ablation result"},
            },
            {
                "sampled": True,
                "message": {
                    "role": "assistant",
                    "content": f"final\n```json\n{json.dumps(data)}\n```",
                },
            },
        ]
    }

    assert parse_proposal(completion_text(trace)) == data


def test_supernodes_filters_and_dedups():
    data = {
        "supernodes": [
            {"name": "output", "nodes": ["1_10_1"]},
            {"name": "A", "nodes": ["1_10_1", "E_0_0", "bogus"]},
            {"name": "B", "nodes": ["1_10_1", "2_20_2"]},
        ]
    }
    assert valid_supernodes(data, GRAPH) == {"A": ["1_10_1"], "B": ["2_20_2"]}


def test_reachable_transitive_closure():
    scorer = InterventionScoring()
    closure = scorer._reachable(
        ["A", "B", "C"], [["A", "B"], ["B", "Output"], ["C", "nope"]]
    )
    assert closure["A"] == {"B", "output"}
    assert closure["B"] == {"output"}
    assert closure["C"] == set()


def test_output_cannot_relay_reachability():
    scorer = InterventionScoring()
    closure = scorer._reachable(["A", "B"], [["A", "output"], ["output", "B"]])
    assert closure["A"] == {"output"}
    assert closure["B"] == set()


def test_slim_graph_and_roundtrip():
    g = slim_graph(GRAPH, max_features=2)
    assert {n.id for n in g.feature_nodes} == {"1_10_1", "2_20_2"}
    assert "E_0_0" in g.nodes
    assert all(e.source in g.nodes and e.target in g.nodes for e in g.edges)
    again = load_graph_json(graph_to_json(g))
    assert {n.id for n in again.nodes.values()} == set(g.nodes)
    assert again.top_outputs == g.top_outputs


def test_load_circuit_tracer_format():
    raw = {
        "metadata": {"prompt": "p", "prompt_tokens": ["p"]},
        "nodes": [
            {
                "node_id": "1_10_0",
                "feature_type": "cross layer transcoder",
                "layer": "1",
                "ctx_idx": 0,
                "influence": 0.4,
            },
            {
                "node_id": "L_1",
                "feature_type": "logit",
                "clerp": 'Output " Austin" (p=0.85)',
                "token_prob": 0.85,
                "influence": 0.9,
            },
        ],
        "links": [{"source": "1_10_0", "target": "L_1", "weight": 2.0}],
    }
    g = load_graph_json(raw)
    assert g.nodes["1_10_0"].kind == "feature"
    assert g.top_outputs == [(" Austin", 0.85)]
    assert g.nodes["1_10_0"].share == pytest.approx(0.4)
    assert g.nodes["L_1"].share == pytest.approx(0.5)


def test_faithful_chain_scores_high():
    out = score(PROPOSAL, CHAIN_RESULT)
    assert out["reward"] > 0.8
    assert out["coverage"] == pytest.approx(1.0)
    assert out["completeness"] > 0.85
    assert out["separation"] > 0.8


def test_wrong_edges_score_lower():
    backwards = {**PROPOSAL, "edges": [["C", "B"], ["B", "A"], ["A", "output"]]}
    assert (
        score(backwards, CHAIN_RESULT)["reward"]
        < score(PROPOSAL, CHAIN_RESULT)["reward"] - 0.2
    )


def test_gate_single_supernode():
    blob = {
        "supernodes": [{"name": "all", "nodes": list(FEATURES)}],
        "edges": [["all", "output"]],
    }
    assert score(blob, CHAIN_RESULT)["reward"] == 0.0


def test_gate_catch_all_dominance():
    hack = {
        "supernodes": [
            {
                "name": "big",
                "nodes": ["1_10_1", "2_20_2", "3_30_3", "1_11_1", "2_21_2"],
            },
            {"name": "decoy", "nodes": ["3_31_3"]},
        ],
        "edges": [["big", "output"]],
    }
    out = score(hack, CHAIN_RESULT)
    assert out["reward"] == 0.0 and "dominates" in out["gated"]


def test_gate_fully_connected():
    every = [
        ["A", "B"],
        ["A", "C"],
        ["B", "A"],
        ["B", "C"],
        ["C", "A"],
        ["C", "B"],
        ["A", "output"],
        ["B", "output"],
        ["C", "output"],
    ]
    out = score({**PROPOSAL, "edges": every}, CHAIN_RESULT)
    assert out["reward"] == 0.0 and out["gated"] == "no falsifiable prediction"


def test_partial_coverage_costs():
    partial = {
        "supernodes": [
            {"name": "A", "nodes": ["1_10_1"]},
            {"name": "B", "nodes": ["2_20_2"]},
            {"name": "C", "nodes": ["3_30_3"]},
        ],
        "edges": PROPOSAL["edges"],
    }
    out = score(partial, CHAIN_RESULT)
    assert out["coverage"] == pytest.approx(0.65)
    assert out["reward"] < score(PROPOSAL, CHAIN_RESULT)["reward"]


def test_incomplete_circuit_costs():
    partial = {
        "supernodes": [
            {"name": "A", "nodes": ["1_10_1", "1_11_1"]},
            {"name": "B", "nodes": ["2_20_2", "2_21_2"]},
            {"name": "C", "nodes": ["3_30_3"]},
        ],
        "edges": PROPOSAL["edges"],
    }
    shifted = {**CHAIN_RESULT, "complement_prob": 0.1}
    out = score(partial, shifted)
    assert out["completeness"] < 0.1
    assert out["reward"] < score(PROPOSAL, CHAIN_RESULT)["reward"]


def test_full_proposal_has_no_complement():
    result = {**CHAIN_RESULT, "complement_prob": None}
    assert score(PROPOSAL, result)["completeness"] == 1.0


def test_chance_level_scores_zero():
    flat = {
        "base_prob": 0.8,
        "base_group_acts": [2.0, 2.0, 2.0],
        "ablations": [
            {"target_prob": 0.8, "group_acts": [2.0, 2.0, 2.0]} for _ in range(3)
        ],
        "complement_prob": None,
    }
    assert score(PROPOSAL, flat)["reward"] == 0.0


def test_gate_supernode_without_outgoing_edge():
    dangling = {
        **PROPOSAL,
        "edges": [["A", "B"], ["B", "output"]],
    }  # C predicts nothing
    out = score(dangling, CHAIN_RESULT)
    assert out["reward"] == 0.0 and "outgoing edge" in out["gated"]


def test_balanced_blob_attack_scores_zero():
    attack = {
        "supernodes": [
            {"name": "driver", "nodes": ["1_10_1", "2_20_2", "3_30_3"]},
            {"name": "d1", "nodes": ["1_11_1", "2_21_2"]},
            {"name": "d2", "nodes": ["3_31_3"]},
        ],
        "edges": [["driver", "output"], ["d1", "output"], ["d2", "output"]],
    }
    result = {
        "base_prob": 0.8,
        "base_group_acts": [2.0, 2.0, 2.0],
        "ablations": [
            {"target_prob": 0.05, "group_acts": [0.0, 0.5, 0.5]},
            {"target_prob": 0.8, "group_acts": [2.0, 0.0, 2.0]},
            {"target_prob": 0.8, "group_acts": [2.0, 2.0, 0.0]},
        ],
        "complement_prob": None,
    }
    assert score(attack, result)["reward"] < 0.05


def test_intervene_cache_dedups_identical_proposals():
    scorer = InterventionScoring()
    calls = 0

    async def fake_remote(prompt, groups, target, complement):
        nonlocal calls
        calls += 1
        return CHAIN_RESULT

    async def run():
        scorer._attributor = type(
            "A",
            (),
            {
                "intervene": type(
                    "M",
                    (),
                    {"remote": type("R", (), {"aio": staticmethod(fake_remote)})()},
                )()
            },
        )()
        a = await scorer._intervene("p", [[(1, 10, 1)]], "t", [])
        b = await scorer._intervene("p", [[(1, 10, 1)]], "t", [])
        return a, b

    a, b = asyncio.run(run())
    assert a is b and calls == 1


def test_multiturn_ablation_tool_reports_effects_and_enforces_budget():
    tools = AblationTools(vf.ToolsetConfig())
    task = CircuitTaskData(
        idx=0,
        prompt="identify the circuit",
        subject_prompt="Fact: test prompt is",
        family="test",
        graph=graph_to_json(GRAPH),
    )

    async def fake_remote(prompt, groups, target, complement, indices):
        assert prompt == "Fact: test prompt is"
        assert target == " Austin"
        assert complement is None and indices == [0]
        assert groups == [[(1, 10, 1)], [(2, 20, 2)]]
        return {
            "base_prob": 0.8,
            "base_group_acts": [1.0, 2.0],
            "ablations": [{"target_prob": 0.2, "group_acts": [0.0, 0.5]}],
        }

    async def run():
        await tools.setup_task(task)
        tools._attributor = SimpleNamespace(
            intervene=SimpleNamespace(remote=SimpleNamespace(aio=fake_remote))
        )
        first = await tools.run_ablation(["1_10_1"], {"downstream": ["2_20_2"]})
        for _ in range(MAX_EXPERIMENTS - 1):
            await tools.run_ablation(["1_10_1"], {"downstream": ["2_20_2"]})
        exhausted = await tools.run_ablation(["2_20_2"])
        return first, exhausted

    first, exhausted = asyncio.run(run())
    assert first["target"]["relative_drop"] == pytest.approx(0.75)
    assert first["probe_groups"]["downstream"]["relative_drop"] == pytest.approx(0.75)
    assert first["experiments_remaining"] == MAX_EXPERIMENTS - 1
    assert "budget exhausted" in exhausted["error"]


def test_multiturn_ablation_tool_rejects_non_features_without_spending_budget():
    tools = AblationTools(vf.ToolsetConfig())
    task = CircuitTaskData(
        idx=0,
        prompt="identify the circuit",
        subject_prompt="Fact: test prompt is",
        family="test",
        graph=graph_to_json(GRAPH),
    )

    async def run():
        await tools.setup_task(task)
        return await tools.run_ablation(["E_0_0"])

    result = asyncio.run(run())
    assert "invalid FEATURE" in result["error"]
    assert result["experiments_remaining"] == MAX_EXPERIMENTS


def test_experiments_used_metric_is_awaitable_for_verifiers_v1():
    assert inspect.iscoroutinefunction(CircuitTask.experiments_used)
    trace = SimpleNamespace(state=CircuitState(experiments_used=3))
    result = asyncio.run(CircuitTask.experiments_used(None, trace))
    assert result == 3.0
