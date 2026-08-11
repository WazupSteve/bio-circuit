import asyncio
from types import SimpleNamespace

import pytest
import verifiers.v1 as vf
from circuit_supernodes import CircuitTaskset as OriginalCircuitTaskset
from circuit_supernodes.circuit_supernodes import CircuitTaskset
from circuit_supernodes.graphs import Graph, GraphEdge, GraphNode, graph_to_json
from circuit_supernodes.rediscovery import (
    SYSTEM_PROMPT,
    CircuitTracerTools,
    DiscoveryTask,
    DiscoveryTaskset,
    call_with_retries,
    continuous_replacement_reward,
    discovery_graph,
    discovery_prompt,
    feature_key,
    membership_score,
    parse_nodes,
    replacement_coordinates,
    replacement_pass,
)
from scripts.build_rediscovery_dataset import (
    curated_groups,
    graph_payload,
    is_positioned_feature_id,
    task_targets,
)
from supernode_rediscovery import DiscoveryTaskset as ExportedDiscoveryTaskset


def sample_graph() -> Graph:
    graph = Graph(tokens=["Dallas", "is", "a"], top_outputs=[(" city", 0.8)])
    for index, (node_id, label) in enumerate(
        (
            ("1_10_0", "city: Dallas"),
            ("2_11_1", "city or municipality"),
            ("3_12_2", "copula"),
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
    graph.edges = [GraphEdge("1_10_0", "2_11_1", 0.7)]
    return graph


def test_rediscovery_is_a_separate_taskset_plugin() -> None:
    assert OriginalCircuitTaskset is CircuitTaskset
    assert ExportedDiscoveryTaskset is DiscoveryTaskset
    assert CircuitTracerTools in DiscoveryTask.tools


def test_task_library_uses_exact_curator_descriptions_by_default() -> None:
    groups = {"capital": ["1_2_3"], "capital cities / say one": ["4_5_6"]}
    assert task_targets(groups, None) == [
        ("capital", "capital"),
        ("capital cities / say one", "capital cities / say one"),
    ]
    graph = {"qParams": {"supernodes": []}}
    assert graph_payload({"graph": graph, "source": {"location": "curated"}}) == (
        graph,
        {"location": "curated"},
    )
    assert is_positioned_feature_id("21_5943_10")
    assert not is_positioned_feature_id("21_5943")


def test_task_library_rejects_ambiguous_curated_groups() -> None:
    with pytest.raises(ValueError, match="repeats a node"):
        curated_groups({"qParams": {"supernodes": [["capital", "1_2_3", "1_2_3"]]}})
    with pytest.raises(ValueError, match="duplicated"):
        curated_groups(
            {"qParams": {"supernodes": [["capital", "1_2_3"], ["capital", "4_5_6"]]}}
        )


def test_initial_prompt_has_description_but_no_subject_prompt_or_graph() -> None:
    prompt = discovery_prompt("capital cities / say a capital city", 5)
    assert (
        "Requested supernode description: 'capital cities / say a capital city'"
        in prompt
    )
    assert "Subject-prompt budget: 5" in prompt
    assert "Dallas" not in prompt
    assert "ATTRIBUTION GRAPH" not in prompt
    assert "circuit_run_prompt" in SYSTEM_PROMPT


def test_tool_graph_strips_labels_and_bounds_distractors() -> None:
    graph = discovery_graph(sample_graph(), 2)
    assert set(graph.nodes) == {"1_10_0", "2_11_1"}
    assert all(node.label == "" for node in graph.nodes.values())


def test_feature_identity_ignores_prompt_specific_position() -> None:
    assert feature_key("21_5943_10") == "21_5943"
    assert feature_key("21_5943") == "21_5943"
    assert feature_key("bad") is None
    result = membership_score(
        ["21_5943_2", "17_7178"],
        ["21_5943_10", "17_7178_10"],
    )
    assert result["reward"] == 1.0
    assert result["exact_match"] is True


def test_hidden_membership_grade_penalizes_false_nodes_and_omissions() -> None:
    result = membership_score(
        ["1_10_9", "99_99", "invalid"],
        ["1_10_0", "2_11_0", "3_12_0"],
    )
    assert result["precision"] == pytest.approx(1 / 3)
    assert result["recall"] == pytest.approx(1 / 3)
    assert result["f1"] == pytest.approx(1 / 3)
    assert result["exact_match"] is False

    duplicate = membership_score(
        ["1_10", "1_10", "2_11"],
        ["1_10_0", "2_11_0"],
    )
    assert duplicate["precision"] == pytest.approx(2 / 3)
    assert duplicate["exact_match"] is False


def test_replacement_coordinates_reuse_hidden_semantic_positions() -> None:
    coordinates, error = replacement_coordinates(
        ["21_5943", "3_99"],
        ["21_5943_10", "17_7178_10", "8_22_12"],
    )
    assert error == ""
    assert coordinates == [
        (3, 99, 10),
        (3, 99, 12),
        (21, 5943, 10),
    ]
    duplicate, error = replacement_coordinates(["3_99_123", "3_99"], ["8_22_12"])
    assert duplicate is None
    assert error == "duplicate feature identity '3_99'"


def test_replacement_grade_requires_same_top_token_and_close_distribution() -> None:
    assert replacement_pass(
        {"valid": True, "top_token_match": True, "output_total_variation": 0.02},
        0.05,
    )
    assert not replacement_pass(
        {"valid": True, "top_token_match": False, "output_total_variation": 0.0},
        0.05,
    )
    assert not replacement_pass(
        {"valid": True, "top_token_match": True, "output_total_variation": 0.06},
        0.05,
    )


def test_continuous_reward_uses_distance_half_life() -> None:
    assert (
        continuous_replacement_reward(
            {"valid": True, "output_total_variation": 0.0}, 0.05
        )
        == 1.0
    )
    assert continuous_replacement_reward(
        {"valid": True, "output_total_variation": 0.05}, 0.05
    ) == pytest.approx(0.5)
    assert continuous_replacement_reward(
        {"valid": True, "output_total_variation": 0.1}, 0.05
    ) == pytest.approx(0.25)
    assert (
        continuous_replacement_reward(
            {"valid": False, "output_total_variation": 0.0}, 0.05
        )
        == 0.0
    )
    assert (
        continuous_replacement_reward(
            {"valid": True, "output_total_variation": float("nan")}, 0.05
        )
        == 0.0
    )


def test_last_valid_json_node_list_is_parsed() -> None:
    reply = 'draft {"nodes": ["1_2"]} final {"nodes": ["3_4", "5_6"]}'
    assert parse_nodes(reply) == ["3_4", "5_6"]
    assert parse_nodes("no json") is None
    assert parse_nodes(None) is None


def test_retry_reports_attempt_count_and_preserves_final_cause(monkeypatch) -> None:
    async def exercise() -> None:
        import circuit_supernodes.rediscovery as rediscovery

        monkeypatch.setattr(rediscovery, "RETRY_DELAYS", (0.0, 0.0))
        attempts = 0

        async def fail() -> None:
            nonlocal attempts
            attempts += 1
            raise OSError(f"failure {attempts}")

        with pytest.raises(RuntimeError, match="failed after 3 attempts") as caught:
            await call_with_retries("test operation", fail)
        assert attempts == 3
        assert isinstance(caught.value.__cause__, OSError)
        assert str(caught.value.__cause__) == "failure 3"

    asyncio.run(exercise())


def test_agent_prompt_runs_subject_tracer_and_spends_budget(monkeypatch) -> None:
    async def exercise() -> None:
        tools = CircuitTracerTools(vf.ToolsetConfig())
        await tools.setup_task(
            SimpleNamespace(
                prompt_budget=1,
                max_features=2,
                modal_app="test-app",
                modal_cls="TestAttributor",
            )
        )

        prompts = []

        async def fake_attribute(prompt: str) -> dict:
            prompts.append(prompt)
            return graph_to_json(sample_graph())

        monkeypatch.setattr(tools, "_attribute", fake_attribute)
        result = await tools.run_prompt("Dallas and Austin are both")
        assert prompts == ["Dallas and Austin are both"]
        assert result["feature_nodes"] == 2
        assert "city: Dallas" not in result["graph"]
        assert " | label" not in result["graph"]
        assert result["prompts_remaining"] == 0

        exhausted = await tools.run_prompt("Paris and London are both")
        assert exhausted["error"].startswith("prompt budget exhausted")
        assert prompts == ["Dallas and Austin are both"]

    asyncio.run(exercise())
