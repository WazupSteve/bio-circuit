import json
import sys
import tomllib
from io import BytesIO
from pathlib import Path

import pytest
import verifiers.v1 as vf
from circuit_supernodes.datagen import generate
from circuit_supernodes.graphs import embed_supernodes

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from build_dataset import add_neuronpedia_labels, is_faithful  # noqa: E402
from export_neuronpedia import (  # noqa: E402
    prepare_neuronpedia_graph,
    prune_for_agent_view,
)
from reward_components import COMPONENTS, read_traces, reward_value  # noqa: E402

CONFIGS = Path(__file__).parents[3] / "configs"

RAW_GRAPH = {
    "metadata": {"slug": "g", "prompt": "p", "prompt_tokens": ["p"]},
    "qParams": {"linkType": "both", "pinnedIds": [], "supernodes": []},
    "nodes": [
        {"node_id": "1_10_0", "feature_type": "cross layer transcoder"},
        {"node_id": "L_1", "feature_type": "logit", "token_prob": 0.85},
        {"node_id": "L_2", "feature_type": "logit", "token_prob": 0.05},
    ],
    "links": [],
}


def test_generate_even_family_quotas():
    samples = generate(16, seed=0, families=["two_hop", "addition"])
    by_family = {
        f: sum(s.family == f for s in samples) for f in ("two_hop", "addition")
    }
    assert by_family == {"two_hop": 8, "addition": 8}
    assert len({s.prompt for s in samples}) == 16


def test_generate_fails_loudly_on_bad_input():
    with pytest.raises(ValueError, match="ran out of unique prompts"):
        # analogy has the smallest pool of the train families; asking for more
        # distinct prompts than exist must fail rather than repeat them.
        generate(100, families=["analogy"])
    with pytest.raises(ValueError, match="unknown"):
        generate(4, families=["twohop"])  # typo'd family name
    with pytest.raises(ValueError, match=">= 1"):
        generate(0)
    with pytest.raises(ValueError, match="empty"):
        generate(4, families=[])  # bare --families must not silently mean "all"


def test_generate_supports_documented_train_recipe():
    fams = ["antonym", "analogy", "acronym", "sequence", "factual", "incontext"]
    samples = generate(48, seed=0, families=fams)
    assert len(samples) == 48
    assert all(sum(s.family == f for s in samples) == 8 for f in fams)


def test_embed_supernodes_builds_qparams():
    out = embed_supernodes(RAW_GRAPH, {"A": ["1_10_0"]}, slug="my-circuit")
    assert out["qParams"]["supernodes"] == [["A", "1_10_0"]]
    assert out["qParams"]["pinnedIds"] == ["1_10_0", "L_1"]
    assert out["qParams"]["linkType"] == "both"
    assert out["metadata"]["slug"] == "my-circuit"
    assert RAW_GRAPH["qParams"]["supernodes"] == []


def test_prepare_neuronpedia_graph_adds_upload_metadata():
    out = prepare_neuronpedia_graph(
        RAW_GRAPH,
        {"A": ["1_10_0"]},
        slug="my-upload",
    )
    assert out["metadata"]["slug"] == "my-upload"
    assert out["metadata"]["scan"] == "qwen3-4b"
    assert out["metadata"]["feature_details"] == {
        "neuronpedia_source_set": "transcoder-hp"
    }


def test_prune_for_agent_view_keeps_displayed_nodes_and_internal_links():
    raw = json.loads(json.dumps(RAW_GRAPH))
    raw["links"] = [
        {"source": "1_10_0", "target": "L_1", "weight": 1.0},
        {"source": "outside", "target": "L_1", "weight": 2.0},
    ]
    out, slim = prune_for_agent_view(raw, max_features=1)
    assert {node["node_id"] for node in out["nodes"]} == set(slim.nodes)
    assert out["links"] == [{"source": "1_10_0", "target": "L_1", "weight": 1.0}]


def test_faithfulness_is_prefix_only():
    def graph(top):
        return {**RAW_GRAPH, "nodes": [{**RAW_GRAPH["nodes"][1], "clerp": top}]}

    assert is_faithful(graph('Output " 9" (p=0.85)'), " 95")[0]
    assert not is_faithful(graph('Output " catch" (p=0.85)'), " cat")[0]
    assert not is_faithful(graph('Output "  " (p=0.85)'), " 95")[0]


def test_neuronpedia_labels_are_batched_and_applied(monkeypatch):
    graph = json.loads(json.dumps(RAW_GRAPH))
    response = [
        {
            "layer": "1-transcoder-hp",
            "index": "10",
            "explanations": [{"description": "city names"}],
        }
    ]

    def fake_open(request, timeout):
        assert timeout == 30
        payload = json.loads(request.data)
        assert payload == [
            {
                "modelId": "qwen3-4b",
                "layer": "1-transcoder-hp",
                "index": 10,
                "maxActsToReturn": 1,
            }
        ]
        return BytesIO(json.dumps(response).encode())

    monkeypatch.setattr("build_dataset.urlopen", fake_open)
    assert add_neuronpedia_labels(graph) == 1
    assert graph["nodes"][0]["label"] == "city names"


def test_verifiers_v1_loader_resolves_taskset_and_lazy_selection(tmp_path, monkeypatch):
    import circuit_supernodes.circuit_supernodes as loader

    row = {"family": "two_hop", "prompt": "p", "target": "x", "graph": RAW_GRAPH}
    # The invalid second row proves select(1) does not eagerly materialize the file.
    (tmp_path / "mini.jsonl").write_text(json.dumps(row) + "\nnot-json\n")
    monkeypatch.setattr(loader, "DATA_DIR", tmp_path)
    env = vf.Environment(
        vf.EnvConfig.model_validate(
            {
                "taskset": {
                    "id": "circuit-supernodes",
                    "dataset_file": "mini.jsonl",
                    "reward_mode": "train",
                    "experiment_budget": 3,
                },
                "harness": {"id": "null"},
                "max_turns": 8,
            }
        )
    )
    assert env.config.max_turns == 8
    assert env.config.harness.id == "null"
    assert env.taskset.config.dataset_file == "mini.jsonl"
    tasks = env.taskset.select(1)
    assert len(tasks) == 1
    assert tasks[0].data.family == "two_hop"
    # Run-level settings must reach the row: the floor and the stated budget.
    assert tasks[0].data.reward_mode == "train"
    assert tasks[0].data.experiment_budget == 3
    assert "Experiment budget: 3." in tasks[0].data.prompt


def test_configs_train_on_different_split_than_eval():
    rl = tomllib.loads((CONFIGS / "rl" / "circuit-supernodes.toml").read_text())
    ev = tomllib.loads((CONFIGS / "eval" / "circuit-supernodes.toml").read_text())
    train = rl["env"][0]["taskset"]["dataset_file"]
    heldout_rl = rl["eval"]["env"][0]["taskset"]["dataset_file"]
    heldout = ev["taskset"]["dataset_file"]
    assert train != heldout
    assert heldout_rl == heldout
    assert rl["env"][0]["taskset"]["id"] == "circuit-supernodes"
    assert ev["taskset"]["id"] == "circuit-supernodes"
    assert rl["env"][0]["harness"]["id"] == "null"
    assert ev["harness"]["id"] == "null"
    assert rl["env"][0]["max_turns"] == ev["max_turns"] == 8


def test_only_training_gets_the_shaped_floor():
    # Checkpoint selection and any published number must see the strict score.
    rl = tomllib.loads((CONFIGS / "rl" / "circuit-supernodes.toml").read_text())
    ev = tomllib.loads((CONFIGS / "eval" / "circuit-supernodes.toml").read_text())
    assert rl["env"][0]["taskset"]["reward_mode"] == "train"
    assert rl["eval"]["env"][0]["taskset"]["reward_mode"] == "eval"
    assert ev["taskset"]["reward_mode"] == "eval"


def test_thinking_matches_between_training_and_evaluation():
    # Comparing a thinking policy against a non-thinking one measures the mode.
    rl = tomllib.loads((CONFIGS / "rl" / "circuit-supernodes.toml").read_text())
    ev = tomllib.loads((CONFIGS / "eval" / "circuit-supernodes.toml").read_text())
    assert rl["sampling"]["enable_thinking"] == ev["sampling"]["enable_thinking"]


def test_training_batches_hold_whole_groups():
    # Prime counts batch_size in rollouts, so a batch that is not a multiple of
    # the group size would split a GRPO group across steps.
    rl = tomllib.loads((CONFIGS / "rl" / "circuit-supernodes.toml").read_text())
    group = rl["rollouts_per_example"]
    assert rl["batch_size"] % group == 0
    assert rl["batch_size"] // group >= 2  # more than one group per step


def test_fault_injection_is_off_outside_the_canary():
    for path in sorted((CONFIGS / "rl").glob("*.toml")):
        taskset = tomllib.loads(path.read_text())["env"][0]["taskset"]
        expected = path.name.startswith("canary-")
        assert bool(taskset.get("fault_rate", 0.0)) is expected, path.name


def test_archived_traces_still_yield_reward_components():
    # The reward audit is derived from these traces; keep them readable so the
    # claim stays checkable from a clean checkout.
    archive = Path(__file__).parents[1] / "archive" / "eval15"
    runs = sorted(archive.glob("*/traces.jsonl"))
    assert [p.parent.name for p in runs] == ["fable", "opus", "sol"]
    for path in runs:
        rewards, components = read_traces(path)
        assert len(rewards) == 15
        assert components
        assert all(set(COMPONENTS) <= set(c) for c in components)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({"circuit_faithfulness": 0.5}, 0.5),  # verifiers v1 records named rewards
        ({"only_one": 0.25}, 0.25),
        (0.75, 0.75),  # older traces stored a bare float
    ],
)
def test_reward_value_reads_both_trace_layouts(raw, expected):
    assert reward_value(raw) == expected


def test_reward_value_refuses_to_guess_between_rewards():
    with pytest.raises(ValueError):
        reward_value({"a": 0.1, "b": 0.2})


def test_sealed_families_are_never_referenced_by_a_config():
    """The "don't touch the sealed families" rule, made mechanical.

    A config that trained or tuned on them would invalidate the final result
    silently, so the check lives here rather than in a document.
    """
    from circuit_supernodes.datagen import SEALED_FAMILIES

    for path in sorted(CONFIGS.rglob("*.toml")):
        text = path.read_text()
        for family in SEALED_FAMILIES:
            assert family not in text, f"{path.name} references sealed {family!r}"


def test_sealed_and_training_families_are_disjoint():
    from circuit_supernodes.datagen import FAMILIES, SEALED_FAMILIES

    train = ["antonym", "analogy", "acronym", "sequence", "factual", "incontext"]
    dev = ["two_hop", "addition"]
    assert not set(SEALED_FAMILIES) & set(train + dev)
    assert set(SEALED_FAMILIES) <= set(FAMILIES)


def test_every_family_can_fill_its_share_of_the_planned_splits():
    # 150 train graphs over 6 families and 36 sealed over 3; a family that runs
    # out repeats prompts, which is memorisation bait at RL visit counts.
    from circuit_supernodes.datagen import SEALED_FAMILIES, generate

    train = ["antonym", "analogy", "acronym", "sequence", "factual", "incontext"]
    for family in train:
        assert len({s.prompt for s in generate(25, seed=0, families=[family])}) == 25
    for family in SEALED_FAMILIES:
        assert len({s.prompt for s in generate(12, seed=0, families=[family])}) == 12


def test_comparison_answer_is_not_readable_from_argument_order():
    """Anti-shortcut check on the sealed comparison family.

    With one relation, "is X greater than Y" is Yes exactly when the larger
    number comes first, so a model can answer without comparing. Varying the
    relation has to break that.
    """
    import re

    from circuit_supernodes.datagen import generate

    samples = generate(400, seed=0, families=["comparison"])
    agree = yes = 0
    for s in samples:
        query = s.prompt.splitlines()[-1]
        left, _, right = re.search(
            r"Is (\d+) (greater|less) than (\d+)", query
        ).groups()
        larger_first = int(left) > int(right)
        answered_yes = s.target.strip() == "Yes"
        agree += larger_first == answered_yes
        yes += answered_yes
    assert 0.4 < yes / len(samples) < 0.6, "answers are unbalanced"
    assert 0.4 < agree / len(samples) < 0.6, "argument order predicts the answer"


def test_agreement_attractor_always_mismatches_the_subject():
    """The attractor exists to be a distractor; if it agreed with the subject
    the prompt would test nothing."""
    from circuit_supernodes.datagen import SVA_SUBJECTS, generate

    plural_nouns = {plur for _, plur, _, _ in SVA_SUBJECTS} | {
        attr_p for _, _, _, attr_p in SVA_SUBJECTS
    }
    for s in generate(60, seed=0, families=["agreement"]):
        words = s.prompt.replace("The ", "").split()
        subject, attractor = words[0], words[-1]
        subject_plural = subject in plural_nouns
        assert s.target.strip() == ("are" if subject_plural else "is")
        if attractor != "aircraft":  # the one noun with no number contrast
            assert (attractor in plural_nouns) != subject_plural


def test_graph_quality_screens_on_unexplained_influence():
    from circuit_supernodes.graphs import graph_quality, load_graph_json

    def graph(error_influence: float):
        # `influence` is cumulative, so build it by accumulating contributions.
        parts = [
            ("1_10_0", "cross layer transcoder", 1.0 - error_influence),
            ("E_0_0", "mlp reconstruction error", error_influence),
        ]
        running, nodes = 0.0, []
        for nid, kind, own in parts:
            running += own
            nodes.append(
                {
                    "node_id": nid,
                    "feature_type": kind,
                    "layer": 1,
                    "ctx_idx": 0,
                    "influence": running,
                }
            )
        return load_graph_json(
            {"metadata": {"prompt_tokens": ["t"]}, "nodes": nodes, "links": []}
        )

    assert graph_quality(graph(0.1))["error"] == pytest.approx(0.1)
    assert graph_quality(graph(0.9))["error"] == pytest.approx(0.9)


def test_is_clean_rejects_a_mostly_error_graph():
    from build_dataset import is_clean

    def raw(error_influence: float):
        parts = [
            ("1_10_0", "cross layer transcoder", 1.0 - error_influence),
            ("E_0_0", "mlp reconstruction error", error_influence),
        ]
        running, nodes = 0.0, []
        for nid, kind, own in parts:
            running += own
            nodes.append(
                {
                    "node_id": nid,
                    "feature_type": kind,
                    "layer": 1,
                    "ctx_idx": 0,
                    "influence": running,
                }
            )
        return {"metadata": {"prompt_tokens": ["t"]}, "nodes": nodes, "links": []}

    assert is_clean(raw(0.2), 0.35, 60)[0]
    assert not is_clean(raw(0.6), 0.35, 60)[0]


def test_budget_variants_expand_at_load_without_duplicating_graphs(
    tmp_path, monkeypatch
):
    """Each graph becomes one task per budget, so the headline budget curve is
    in-distribution and every rollout of a task shares its allowance."""
    import circuit_supernodes.circuit_supernodes as loader

    row = {"family": "two_hop", "prompt": "p", "target": "x", "graph": RAW_GRAPH}
    (tmp_path / "mini.jsonl").write_text(
        json.dumps(row) + "\n" + json.dumps(row) + "\n"
    )
    monkeypatch.setattr(loader, "DATA_DIR", tmp_path)
    env = vf.Environment(
        vf.EnvConfig.model_validate(
            {
                "taskset": {
                    "id": "circuit-supernodes",
                    "dataset_file": "mini.jsonl",
                    "experiment_budgets": [0, 3, 7],
                },
                "harness": {"id": "null"},
                "max_turns": 8,
            }
        )
    )
    tasks = env.taskset.select(6)
    assert len(tasks) == 6  # two graphs x three budgets
    assert [t.data.experiment_budget for t in tasks] == [0, 3, 7, 0, 3, 7]
    assert len({t.data.name for t in tasks}) == 6  # names stay distinct
    assert len({t.data.idx for t in tasks}) == 6
    for t in tasks:
        assert f"Experiment budget: {t.data.experiment_budget}." in t.data.prompt
