import asyncio
import inspect
import json
from types import SimpleNamespace

import pytest
import verifiers.v1 as vf
from circuit_supernodes.circuit_supernodes import (
    CircuitTask,
    CircuitTaskData,
    fails_injected_fault,
    split_nonce,
)
from circuit_supernodes.experiments import (
    DEFAULT_BUDGET,
    AblationTools,
    CircuitState,
)
from circuit_supernodes.graphs import graph_to_json, load_graph_json, slim_graph
from circuit_supernodes.rubric import (
    MIN_GROUP_SIZE,
    SCORING_ABLATION,
    TOOL_ABLATION,
    InterventionScoring,
    completion_text,
    parse_proposal,
    shaped_reward,
    split_members,
    split_seed,
    valid_supernodes,
    validate_members,
    validate_topology,
)
from conftest import FEATURES, GRAPH, GROUPS, PROMPT

TRUTH = {"A": {"B", "C", "output"}, "B": {"C", "output"}, "C": {"output"}}
PROPOSAL = {
    "supernodes": [{"name": n, "nodes": ms} for n, ms in GROUPS.items()],
    "edges": [["A", "B"], ["B", "C"], ["C", "output"]],
}
BASE_PROB, BASE_ACT = 0.8, 2.0


def simulate(truth=TRUTH, weights=None, complement_prob=None, groups=GROUPS):
    """Fake ``_intervene`` backed by an explicit causal model.

    Ablating any subset removes that subset's share of its group's causal weight,
    and the group's true downstream targets drop by exactly that fraction. One
    truth spec therefore drives both reward versions: v1 ablates whole groups, v2
    ablates halves, and ``weights`` decides whether the role is distributed across
    a group's members or concentrated in a few of them.
    """
    weights = weights or dict.fromkeys(FEATURES, 1.0)
    names = list(groups)

    async def fake(prompt, sent, target, complement, indices=None):
        ids = [[f"{a}_{b}_{c}" for a, b, c in group] for group in sent]
        chosen = range(len(sent)) if indices is None else indices
        ablations = []
        for i in chosen:
            ablated = set(ids[i])
            src = next(n for n in names if ablated <= set(groups[n]))
            fraction = sum(weights[f] for f in ablated) / sum(
                weights[f] for f in groups[src]
            )
            affected = truth[src] | {src}
            ablations.append(
                {
                    "target_prob": BASE_PROB * (1 - fraction)
                    if "output" in truth[src]
                    else BASE_PROB,
                    "group_acts": [
                        BASE_ACT * (1 - fraction)
                        if j < len(names) and names[j] in affected
                        else BASE_ACT
                        for j in range(len(sent))
                    ],
                }
            )
        return {
            "base_prob": BASE_PROB,
            "base_group_acts": [BASE_ACT] * len(sent),
            "ablations": ablations,
            "complement_prob": complement_prob,
        }

    return fake


def score(proposal=PROPOSAL, *, split_halves=True, graph=GRAPH, **kwargs) -> dict:
    scorer = InterventionScoring(split_halves=split_halves)
    scorer._intervene = simulate(**kwargs)
    return asyncio.run(scorer.score(PROMPT, graph, proposal))


# --- parsing ---------------------------------------------------------------


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


def test_parse_takes_the_last_valid_block():
    draft = {"supernodes": [{"name": "draft", "nodes": ["1_10_1"]}]}
    final = {"supernodes": [{"name": "final", "nodes": ["2_20_2"]}]}
    text = (
        f"first attempt\n```json\n{json.dumps(draft)}\n```\n"
        f"on reflection\n```json\n{json.dumps(final)}\n```"
    )
    assert parse_proposal(text) == final


def test_parse_skips_trailing_block_that_is_not_a_proposal():
    good = {"supernodes": [{"name": "A", "nodes": ["1_10_1"]}]}
    text = f'```json\n{json.dumps(good)}\n```\n```json\n{{"note": 1}}\n```'
    assert parse_proposal(text) == good


def test_completion_text_reads_verifiers_v1_trace_nodes():
    data = {"supernodes": [{"name": "A", "nodes": ["1_10_1"]}]}
    trace = {
        "nodes": [
            {"sampled": False, "message": {"role": "user", "content": "find it"}},
            {"sampled": True, "message": {"role": "assistant", "content": "draft"}},
            {"sampled": False, "message": {"role": "tool", "content": "ablation"}},
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


# --- strict membership and topology gates ----------------------------------


def members(**overrides) -> dict:
    supernodes = [{"name": n, "nodes": list(ms)} for n, ms in GROUPS.items()]
    return {"supernodes": overrides.get("supernodes", supernodes)}


@pytest.mark.parametrize(
    ("supernodes", "expected"),
    [
        ([{"name": "A", "nodes": GROUPS["A"]}], "2-8 supernodes"),
        (
            [
                {"name": "A", "nodes": GROUPS["A"]},
                {"name": "a", "nodes": GROUPS["B"]},
            ],
            "duplicate supernode name",
        ),
        (
            [
                {"name": "A", "nodes": GROUPS["A"]},
                {"name": "B", "nodes": GROUPS["A"][:3] + GROUPS["B"][:4]},
            ],
            "overlaps another supernode",
        ),
        (
            [
                {"name": "A", "nodes": GROUPS["A"][:6] + ["E_0_0"]},
                {"name": "B", "nodes": GROUPS["B"]},
            ],
            "non-FEATURE ids",
        ),
        (
            [
                {"name": "A", "nodes": GROUPS["A"][:6] + ["nope"]},
                {"name": "B", "nodes": GROUPS["B"]},
            ],
            "non-FEATURE ids",
        ),
        (
            [
                {"name": "A", "nodes": GROUPS["A"][:1]},
                {"name": "B", "nodes": GROUPS["B"]},
            ],
            f"at least {MIN_GROUP_SIZE}",
        ),
        (
            [
                {"name": "A", "nodes": GROUPS["A"][:6] + GROUPS["A"][:1]},
                {"name": "B", "nodes": GROUPS["B"]},
            ],
            "repeats a member",
        ),
        (
            [
                {"name": "output", "nodes": GROUPS["A"]},
                {"name": "B", "nodes": GROUPS["B"]},
            ],
            "reserved terminal",
        ),
        (
            [
                {"name": "big", "nodes": GROUPS["A"] + GROUPS["B"] + GROUPS["C"][:1]},
                {"name": "small", "nodes": GROUPS["C"][1:]},
            ],
            "dominates",
        ),
    ],
)
def test_membership_gate_rejects_rather_than_repairs(supernodes, expected):
    sns, reason = validate_members({"supernodes": supernodes}, GRAPH)
    assert sns is None and expected in reason


def test_membership_gate_accepts_a_clean_proposal():
    sns, reason = validate_members(members(), GRAPH)
    assert reason == "" and sns == GROUPS


@pytest.mark.parametrize(
    ("edges", "expected"),
    [
        ([["A", "B"], ["B", "A"], ["A", "output"]], "cycle"),
        ([["A", "B"]], "no supernode reaches"),  # nothing predicts the output
        (
            [["A", "output"], ["output", "B"], ["B", "output"]],
            "cannot be an edge source",
        ),
        ([["A", "A"], ["A", "output"]], "self-loop"),
        ([["A", "ghost"], ["A", "output"]], "unknown supernode"),
        ([["A"], ["A", "output"]], "[source, target] pair"),
    ],
)
def test_topology_gate_rejects(edges, expected):
    closure, reason = validate_topology(["A", "B"], edges)
    assert closure is None and expected in reason


def test_topology_gate_returns_transitive_closure_and_is_case_insensitive():
    closure, reason = validate_topology(
        ["A", "B", "C"], [["a", "B"], ["B", "c"], ["C", "Output"]]
    )
    assert reason == ""
    assert closure["A"] == {"B", "C", "output"}
    assert closure["C"] == {"output"}


def test_export_path_stays_lenient():
    # The Neuronpedia gallery must render imperfect proposals the reward rejects.
    data = {
        "supernodes": [
            {"name": "output", "nodes": ["1_10_1"]},
            {"name": "A", "nodes": ["1_10_1", "E_0_0", "bogus"]},
            {"name": "B", "nodes": ["1_10_1", "2_20_2"]},
        ]
    }
    assert valid_supernodes(data, GRAPH) == {"A": ["1_10_1"], "B": ["2_20_2"]}


# --- the hidden split ------------------------------------------------------


def test_split_is_deterministic_and_rotates_with_the_nonce():
    a = split_seed("graph-1", "run:3", salt="s")
    assert a == split_seed("graph-1", "run:3", salt="s")  # siblings in a group
    assert a != split_seed("graph-1", "run:4", salt="s")  # next step
    assert a != split_seed("graph-2", "run:3", salt="s")  # different graph
    assert a != split_seed("graph-1", "run:3", salt="other")  # different salt


def test_split_seed_survives_process_hash_randomization():
    # Regression guard: builtin hash() is salted per process, so sibling rollouts
    # in different workers would silently disagree about the split.
    import subprocess
    import sys

    code = (
        "from circuit_supernodes.rubric import split_seed;"
        "print(split_seed('g', 'n', salt='s').hex())"
    )
    runs = {
        subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
        ).stdout.strip()
        for seed in ("0", "1", "random")
    }
    assert len(runs) == 1


def test_split_halves_are_balanced_and_partition_the_group():
    a, b = split_members(GROUPS["A"], split_seed("g", "n", salt="s"))
    assert sorted(a + b) == sorted(GROUPS["A"])
    assert not set(a) & set(b)
    assert abs(len(a) - len(b)) <= 1


def test_split_ignores_member_ordering():
    seed = split_seed("g", "n", salt="s")
    assert split_members(GROUPS["A"], seed) == split_members(
        list(reversed(GROUPS["A"])), seed
    )


# --- scoring ---------------------------------------------------------------


def test_faithful_chain_scores_high_under_both_versions():
    for split_halves in (False, True):
        out = score(split_halves=split_halves)
        assert out["reward"] > 0.6, split_halves
        assert out["coverage"] == pytest.approx(1.0)
        assert out["completeness"] == 1.0


def test_wrong_edges_score_lower():
    backwards = {**PROPOSAL, "edges": [["C", "B"], ["B", "A"], ["A", "output"]]}
    assert score(backwards)["reward"] < score()["reward"] - 0.2


def test_v2_collapses_when_one_half_carries_the_group():
    # A's causal role sits entirely in its first three members. v1 ablates the
    # whole group and cannot tell; v2 ablates each half and the geometric mean of
    # a live half and a dead half goes to zero.
    lopsided = {**dict.fromkeys(FEATURES, 1.0)}
    for f in GROUPS["A"][3:]:
        lopsided[f] = 0.0
    distributed = score()["separation"]
    v1 = score(split_halves=False, weights=lopsided)["separation"]
    v2 = score(split_halves=True, weights=lopsided)["separation"]
    assert v1 > v2
    assert v2 < distributed


def test_v2_sends_both_halves_of_every_group():
    seen: dict = {}

    async def spy(prompt, sent, target, complement, indices=None):
        seen["sent"], seen["indices"] = sent, indices
        return await simulate()(prompt, sent, target, complement, indices)

    scorer = InterventionScoring(split_halves=True)
    scorer._intervene = spy
    asyncio.run(scorer.score(PROMPT, GRAPH, PROPOSAL))
    k = len(GROUPS)
    assert len(seen["sent"]) == 3 * k  # fulls measured, halves ablated
    assert seen["indices"] == list(range(k, 3 * k))


def test_fully_connected_proposal_is_a_cycle():
    # Claiming everything reaches everything used to score as "unfalsifiable";
    # the DAG gate now rejects it before any GPU time is spent.
    every = [[a, b] for a in GROUPS for b in list(GROUPS) + ["output"] if a != b]
    out = score({**PROPOSAL, "edges": every})
    assert out["reward"] == 0.0 and "cycle" in out["gated"]


def test_gate_unfalsifiable_when_no_effect_is_measurable():
    async def dead(prompt, sent, target, complement, indices=None):
        chosen = range(len(sent)) if indices is None else indices
        return {
            "base_prob": BASE_PROB,
            "base_group_acts": [0.0] * len(sent),  # nothing to compare against
            "ablations": [
                {"target_prob": 0.1, "group_acts": [0.0] * len(sent)} for _ in chosen
            ],
            "complement_prob": None,
        }

    scorer = InterventionScoring()
    scorer._intervene = dead
    out = asyncio.run(scorer.score(PROMPT, GRAPH, PROPOSAL))
    assert out["reward"] == 0.0 and out["gated"] == "no falsifiable prediction"


def test_partial_coverage_costs():
    partial = {
        "supernodes": [{"name": n, "nodes": GROUPS[n]} for n in ("A", "B")],
        "edges": [["A", "B"], ["B", "output"]],
    }
    out = score(partial)
    assert out["coverage"] == pytest.approx(14 / 21)
    assert out["reward"] < score()["reward"]


def test_incomplete_circuit_costs():
    out = score(complement_prob=0.1)
    assert out["completeness"] < 0.1
    assert out["reward"] < score()["reward"]


def test_full_proposal_has_no_complement():
    assert score()["completeness"] == 1.0


def test_chance_level_scores_zero():
    # Nothing the agent claims has any causal effect: predicted and unpredicted
    # relations are indistinguishable, so the rank term carries no skill.
    inert = dict.fromkeys(GROUPS, frozenset())
    assert score(truth=inert)["reward"] == 0.0


# --- reward modes and error semantics --------------------------------------


def test_shaped_floor_rewards_partial_validity():
    assert shaped_reward({"parse": False, "reward": 0.0}) == 0.0
    assert shaped_reward({"parse": True, "reward": 0.0}) == pytest.approx(0.01)
    assert shaped_reward(
        {"parse": True, "membership": True, "topology": True, "reward": 0.0}
    ) == pytest.approx(0.05)
    assert shaped_reward(
        {"parse": True, "membership": True, "topology": True, "reward": 1.0}
    ) == pytest.approx(1.0)


def test_gated_results_report_the_stage_they_failed_at():
    tiny = {
        "supernodes": [{"name": "A", "nodes": GROUPS["A"][:2]}],
        "edges": [["A", "output"]],
    }
    out = score(tiny)
    assert out["parse"] and not out["membership"] and not out["topology"]

    dangling = {**PROPOSAL, "edges": [["A", "B"]]}  # nothing reaches output
    out = score(dangling)
    assert out["parse"] and out["membership"] and not out["topology"]


def test_infra_failure_raises_instead_of_scoring_zero():
    # A false 0.0 would corrupt every advantage in the GRPO group.
    scorer = InterventionScoring()

    async def broken(*args, **kwargs):
        raise RuntimeError("modal is down")

    scorer._intervene = broken
    with pytest.raises(RuntimeError, match="modal is down"):
        asyncio.run(scorer.score(PROMPT, GRAPH, PROPOSAL))


def test_remote_retries_then_surfaces_the_failure():
    from circuit_supernodes.rubric import intervene_with_retries

    attempts = 0

    async def flaky(*args):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("transient")
        return {"ok": True}

    attributor = SimpleNamespace(
        intervene=SimpleNamespace(remote=SimpleNamespace(aio=flaky))
    )

    async def run():
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("circuit_supernodes.rubric.RETRY_DELAYS", (0.0, 0.0, 0.0))
            return await intervene_with_retries(attributor, "p", [], "t", None, None)

    assert asyncio.run(run()) == {"ok": True} and attempts == 3


def test_fault_injection_is_inert_by_default_and_deterministic_when_armed():
    ids = [f"rollout-{i}" for i in range(400)]
    assert not any(fails_injected_fault(i, 0.0) for i in ids)
    assert all(fails_injected_fault(i, 1.0) for i in ids)
    armed = [i for i in ids if fails_injected_fault(i, 0.125)]
    assert armed == [i for i in ids if fails_injected_fault(i, 0.125)]  # reproducible
    assert 0.05 < len(armed) / len(ids) < 0.25


def test_split_nonce_is_shared_by_a_step_and_fixed_for_eval():
    train = SimpleNamespace(type="train", id="run-7", step=3)
    assert split_nonce(SimpleNamespace(run=train)) == "run-7:3"
    assert split_nonce(SimpleNamespace(run=train)) != "run-7:4"
    assert split_nonce(SimpleNamespace(run=None)) == "eval"
    assert (
        split_nonce(
            SimpleNamespace(run=SimpleNamespace(type="eval", id="e", step=None))
        )
        == "eval"
    )


def test_scoring_and_the_tool_use_different_ablation_operators():
    # If they matched, an agent could measure exactly what it will be scored on
    # and fitting would be enough. Holding the operator out is what forces the
    # circuit to generalise.
    assert TOOL_ABLATION != SCORING_ABLATION
    assert InterventionScoring().ablation == SCORING_ABLATION


def test_intervene_cache_dedups_identical_proposals():
    scorer = InterventionScoring()
    calls = 0

    async def fake_remote(prompt, groups, target, complement, indices, ablation):
        nonlocal calls
        calls += 1
        assert ablation == SCORING_ABLATION  # scoring never uses the agent's operator
        return {"ok": True}

    async def run():
        scorer._attributor = SimpleNamespace(
            intervene=SimpleNamespace(remote=SimpleNamespace(aio=fake_remote))
        )
        a = await scorer._intervene("p", [[(1, 10, 1)]], "t", [])
        b = await scorer._intervene("p", [[(1, 10, 1)]], "t", [])
        return a, b

    a, b = asyncio.run(run())
    assert a is b and calls == 1


# --- graph plumbing --------------------------------------------------------


def test_slim_graph_and_roundtrip():
    g = slim_graph(GRAPH, max_features=2)
    assert len(g.feature_nodes) == 2
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


# --- the experiment tool ---------------------------------------------------


def task_data(**overrides) -> CircuitTaskData:
    return CircuitTaskData(
        idx=0,
        prompt="identify the circuit",
        subject_prompt=PROMPT,
        family="test",
        graph=graph_to_json(GRAPH),
        **overrides,
    )


def test_multiturn_ablation_tool_reports_effects_and_enforces_budget():
    tools = AblationTools(vf.ToolsetConfig())

    async def fake_remote(prompt, groups, target, complement, indices):
        assert prompt == PROMPT and target == " Austin"
        assert complement is None and indices == [0]
        assert groups == [[(1, 10, 1)], [(2, 20, 2)]]
        return {
            "base_prob": 0.8,
            "base_group_acts": [1.0, 2.0],
            "ablations": [{"target_prob": 0.2, "group_acts": [0.0, 0.5]}],
        }

    async def run():
        await tools.setup_task(task_data())
        tools._attributor = SimpleNamespace(
            intervene=SimpleNamespace(remote=SimpleNamespace(aio=fake_remote))
        )
        first = await tools.run_ablation(["1_10_1"], {"downstream": ["2_20_2"]})
        for _ in range(DEFAULT_BUDGET - 1):
            await tools.run_ablation(["1_10_1"], {"downstream": ["2_20_2"]})
        return first, await tools.run_ablation(["2_20_2"])

    first, exhausted = asyncio.run(run())
    assert first["target"]["relative_drop"] == pytest.approx(0.75)
    assert first["probe_groups"]["downstream"]["relative_drop"] == pytest.approx(0.75)
    assert first["experiments_remaining"] == DEFAULT_BUDGET - 1
    assert "budget exhausted" in exhausted["error"]


def test_experiment_budget_comes_from_the_task_row():
    tools = AblationTools(vf.ToolsetConfig())
    asyncio.run(tools.setup_task(task_data(experiment_budget=0)))
    assert tools.remaining == 0
    result = asyncio.run(tools.run_ablation(["1_10_1"]))
    assert "budget exhausted" in result["error"]


def test_multiturn_ablation_tool_rejects_non_features_without_spending_budget():
    tools = AblationTools(vf.ToolsetConfig())

    async def run():
        await tools.setup_task(task_data())
        return await tools.run_ablation(["E_0_0"])

    result = asyncio.run(run())
    assert "invalid FEATURE" in result["error"]
    assert result["experiments_remaining"] == DEFAULT_BUDGET


def test_experiments_used_metric_is_awaitable_for_verifiers_v1():
    assert inspect.iscoroutinefunction(CircuitTask.experiments_used)
    trace = SimpleNamespace(state=CircuitState(experiments_used=3))
    assert asyncio.run(CircuitTask.experiments_used(None, trace)) == 3.0
