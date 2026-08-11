import json

from circuit_supernodes import output_restoration_task as restoration_environment
from circuit_supernodes.graphs import (
    Graph,
    GraphNode,
    graph_to_json,
    load_graph_json,
    render_graph,
    slim_graph,
)
from circuit_supernodes.interventions import CORRUPTION_VERSION
from circuit_supernodes.output_restoration_task import (
    CircuitTaskset as RestorationTaskset,
)
from circuit_supernodes.output_restoration_task import (
    CircuitTasksetConfig as RestorationTasksetConfig,
)
from circuit_supernodes.restoration import validate_intervention_graph


def ranked_graph(count: int = 100) -> Graph:
    graph = Graph(tokens=["x"], top_outputs=[(" clean", 0.9)])
    for index in range(count):
        node_id = f"0_{index}_0"
        graph.nodes[node_id] = GraphNode(
            id=node_id,
            kind="feature",
            layer=0,
            ctx_idx=0,
            token="x",
            influence=float(index + 1),
            share=float(count - index),
            label="" if index % 2 else f"feature {index}",
            activation=index / 10,
        )
    return graph


def test_combined_graph_schema_round_trips_old_and_new_views() -> None:
    legacy = {
        "format": "slim",
        "tokens": ["legacy"],
        "top_outputs": [[" token", 0.5]],
        "nodes": [
            {
                "id": "1_2_0",
                "kind": "feature",
                "layer": 1,
                "ctx_idx": 0,
                "token": "legacy",
                "influence": 0.2,
                "share": 0.2,
            }
        ],
        "edges": [],
    }
    old_graph = load_graph_json(legacy)
    assert old_graph.nodes["1_2_0"].activation == 0.0
    assert old_graph.nodes["1_2_0"].label == ""

    complete = ranked_graph()
    round_tripped = load_graph_json(graph_to_json(complete))
    assert len(round_tripped.feature_nodes) == 100
    assert round_tripped.nodes["0_7_0"].activation == 0.7
    rendered = render_graph(slim_graph(round_tripped, 3), include_labels=False)
    assert "label" not in rendered

    ordinary = slim_graph(round_tripped, 3)
    retained = slim_graph(round_tripped, 3, {"0_99_0"})
    assert "0_99_0" not in ordinary.nodes
    assert "0_99_0" in retained.nodes


def test_restoration_keeps_full_graph_without_forcing_corruption_into_top_k(
    tmp_path, monkeypatch
) -> None:
    graph = ranked_graph()
    hidden_ids = {f"0_{index}_0" for index in range(60, 100)}
    corruption = [[0, index, 0, 2.0] for index in range(60, 100)]
    row = {
        "family": "integration",
        "prompt": "x",
        "target_token": " clean",
        "graph": graph_to_json(graph),
        "corruption": corruption,
        "corruption_version": CORRUPTION_VERSION,
    }
    dataset = tmp_path / "output-restoration.jsonl"
    dataset.write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(restoration_environment, "DATA_DIR", tmp_path)

    config = RestorationTasksetConfig(max_features=40)
    task = next(iter(RestorationTaskset(config).load()))
    loaded = load_graph_json(task.data.graph)
    initial_view = slim_graph(loaded, 40)

    assert len(loaded.feature_nodes) == 100
    assert hidden_ids.isdisjoint(initial_view.nodes)
    assert all(f"  {node_id} |" not in task.data.prompt for node_id in hidden_ids)

    proposal = {"nodes": [{"id": "0_99_0", "intervention_weight": 0.5}]}
    validated, error = validate_intervention_graph(proposal, loaded)
    assert error == ""
    assert validated == [("0_99_0", 0.5)]


def test_all_plugin_entrypoints_export_independent_tasksets() -> None:
    from circuit_supernodes import CircuitTaskset as MdlTaskset
    from output_restoration import CircuitTaskset as OutputRestorationTaskset
    from supernode_rediscovery import DiscoveryTaskset as RediscoveryTaskset

    assert len({MdlTaskset, RediscoveryTaskset, OutputRestorationTaskset}) == 3
