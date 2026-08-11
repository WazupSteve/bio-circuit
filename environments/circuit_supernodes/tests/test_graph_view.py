import json

import pytest
from circuit_supernodes import circuit_supernodes as environment
from circuit_supernodes.circuit_supernodes import CircuitTaskset, CircuitTasksetConfig
from circuit_supernodes.experiments import graph_view
from circuit_supernodes.graphs import Graph, GraphNode, graph_to_json, load_graph_json


def ranked_graph(count: int = 75) -> Graph:
    graph = Graph(tokens=["x"], top_outputs=[("y", 0.5)])
    for index in range(count):
        node_id = f"0_{index}_0"
        graph.nodes[node_id] = GraphNode(
            id=node_id,
            kind="feature",
            layer=0,
            ctx_idx=0,
            token="x",
            influence=0.0,
            share=float(count - index),
        )
    return graph


def test_graph_view_returns_policy_selected_top_ranked_count() -> None:
    payload = graph_view(ranked_graph(), feature_count=3)

    assert payload["returned_feature_count"] == 3
    assert payload["available_feature_count"] == 75
    rendered = str(payload["graph"])
    assert "0_0_0" in rendered
    assert "0_2_0" in rendered
    assert "0_3_0" not in rendered


def test_graph_view_rejects_counts_outside_exported_graph() -> None:
    for feature_count in (1, 76):
        with pytest.raises(ValueError, match="between 2 and 75"):
            graph_view(ranked_graph(), feature_count=feature_count)


def test_taskset_keeps_complete_exported_graph(tmp_path, monkeypatch) -> None:
    graph = ranked_graph()
    row = {
        "family": "test",
        "prompt": "x",
        "target": "y",
        "graph": graph_to_json(graph),
    }
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(environment, "DATA_DIR", tmp_path)

    task = next(iter(CircuitTaskset(CircuitTasksetConfig()).load()))

    assert len(load_graph_json(task.data.graph).feature_nodes) == 75
    assert "2 and 75 FEATURE nodes" in task.data.prompt


def test_taskset_rejects_legacy_truncated_response_bank(tmp_path, monkeypatch) -> None:
    graph = ranked_graph()
    row = {
        "family": "test",
        "prompt": "x",
        "target": "y",
        "graph": graph_to_json(graph),
        "response_bank": {"feature_ids": ["0_0_0"]},
    }
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(environment, "DATA_DIR", tmp_path)

    taskset = CircuitTaskset(CircuitTasksetConfig())
    with pytest.raises(ValueError, match="rebuild the response-bank dataset"):
        next(iter(taskset.load()))


def test_taskset_rejects_duplicate_response_bank_feature_ids(
    tmp_path, monkeypatch
) -> None:
    graph = ranked_graph(count=2)
    row = {
        "family": "test",
        "prompt": "x",
        "target": "y",
        "graph": graph_to_json(graph),
        "response_bank": {"feature_ids": ["0_0_0", "0_1_0", "0_1_0"]},
    }
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(environment, "DATA_DIR", tmp_path)

    with pytest.raises(ValueError, match="rebuild the response-bank dataset"):
        next(iter(CircuitTaskset(CircuitTasksetConfig()).load()))
