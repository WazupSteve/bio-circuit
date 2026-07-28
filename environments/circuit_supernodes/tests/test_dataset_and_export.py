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
    with pytest.raises(ValueError, match="factual"):
        generate(9, families=["factual"])  # only 8 distinct factual prompts exist
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


def test_archived_traces_still_yield_reward_components():
    # The reward audit in STEPS.md sec.2 is derived from these traces; keep them
    # readable so the claim stays checkable from a clean checkout.
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
