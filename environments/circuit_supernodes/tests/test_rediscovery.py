import pytest
from circuit_supernodes import CircuitTaskset
from circuit_supernodes.graphs import Graph, GraphEdge, GraphNode
from circuit_supernodes.rediscovery import (
    SYSTEM_PROMPT,
    RediscoveryTask,
    RediscoveryTaskset,
    membership_score,
    parse_nodes,
    rediscovery_graph,
    rediscovery_prompt,
)


def test_package_exports_the_zero_shot_taskset() -> None:
    assert CircuitTaskset is RediscoveryTaskset


def sample_graph() -> Graph:
    graph = Graph(tokens=["Dallas", "is", "in"], top_outputs=[(" Texas", 0.8)])
    for index, (node_id, label) in enumerate(
        (
            ("1_10_0", "city: Dallas"),
            ("2_11_0", "city or municipality"),
            ("3_12_1", "copula"),
            ("4_13_2", "location preposition"),
        )
    ):
        graph.nodes[node_id] = GraphNode(
            node_id,
            "feature",
            index + 1,
            min(index, 2),
            graph.tokens[min(index, 2)],
            0.0,
            1.0 - index * 0.1,
            label,
        )
    graph.edges = [GraphEdge("1_10_0", "2_11_0", 0.7)]
    return graph


def test_zero_shot_prompt_hides_labels_and_reference_membership() -> None:
    graph = rediscovery_graph(sample_graph(), ["1_10_0", "2_11_0"], 4)
    prompt = rediscovery_prompt(graph, "city")
    assert "Requested concept: 'city'" in prompt
    assert "city: Dallas" not in prompt
    assert "city or municipality" not in prompt
    assert "reference_nodes" not in prompt
    assert "demonstration" not in prompt.lower()
    assert "no demonstrations" in SYSTEM_PROMPT.lower()
    assert "correctness" in SYSTEM_PROMPT.lower()
    assert "feedback" in SYSTEM_PROMPT.lower()
    assert not getattr(RediscoveryTask, "tools", ())


def test_reference_members_are_retained_among_ranked_distractors() -> None:
    graph = rediscovery_graph(sample_graph(), ["2_11_0"], 2)
    assert set(graph.nodes) == {"1_10_0", "2_11_0"}
    assert all(node.label == "" for node in graph.nodes.values())


def test_hidden_membership_grade_rewards_exact_rediscovery() -> None:
    result = membership_score(["1_10_0", "2_11_0"], ["1_10_0", "2_11_0"])
    assert result["reward"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["exact_match"] is True


def test_hidden_membership_grade_penalizes_false_nodes_and_omissions() -> None:
    result = membership_score(["1_10_0", "wrong"], ["1_10_0", "2_11_0", "3"])
    assert result["precision"] == 0.5
    assert result["recall"] == pytest.approx(1 / 3)
    assert result["f1"] == pytest.approx(0.4)
    assert result["exact_match"] is False


def test_last_valid_json_node_list_is_parsed() -> None:
    reply = 'draft ```json\n{"nodes": ["wrong"]}\n``` final {"nodes": ["a", "b"]}'
    assert parse_nodes(reply) == ["a", "b"]
    assert parse_nodes("no json") is None
