import json
import sys
import tomllib
from io import BytesIO
from pathlib import Path

import pytest
from circuit_supernodes.datagen import generate
from circuit_supernodes.graphs import embed_supernodes

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from build_dataset import add_neuronpedia_labels, is_faithful  # noqa: E402

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


def test_load_environment_accepts_dict_config(tmp_path, monkeypatch):
    import circuit_supernodes.circuit_supernodes as loader

    row = {"family": "two_hop", "prompt": "p", "target": "x", "graph": RAW_GRAPH}
    (tmp_path / "mini.jsonl").write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(loader, "DATA_DIR", tmp_path)
    env = loader.load_environment({"taskset": {"dataset_file": "mini.jsonl"}})
    assert env.config.max_turns == 4
    assert env.config.harness.id == "null"
    assert env.taskset.config.dataset_file == "mini.jsonl"
    tasks = env.taskset.load()
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
    assert rl["env"][0]["max_turns"] == ev["max_turns"] == 4
