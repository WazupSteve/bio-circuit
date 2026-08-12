import asyncio
import json
from types import SimpleNamespace

import pytest
import verifiers.v1 as vf
from circuit_supernodes import CircuitTaskset as OriginalCircuitTaskset
from circuit_supernodes.circuit_supernodes import CircuitTaskset
from circuit_supernodes.cohorts import parse_weighted_nodes
from circuit_supernodes.graphs import Graph, GraphEdge, GraphNode, graph_to_json
from circuit_supernodes.rediscovery import (
    SYSTEM_PROMPT,
    CohortGraphTools,
    DiscoveryTask,
    DiscoveryTaskset,
    DiscoveryTasksetConfig,
    call_with_retries,
    cohort_prompt,
    promptless_graph,
    promptless_score,
)
from scripts.build_promptless_dataset import cohort_rows
from supernode_rediscovery import DiscoveryTaskset as ExportedDiscoveryTaskset


def sample_graph(count: int = 6) -> Graph:
    graph = Graph(tokens=["secret", "prompt"], top_outputs=[(" answer", 0.8)])
    for index in range(count):
        node_id = f"1_{10 + index}_{index % 2}"
        graph.nodes[node_id] = GraphNode(
            node_id,
            "feature",
            1,
            index % 2,
            graph.tokens[index % 2],
            0.0,
            float(count - index),
            f"human label {index}",
        )
    graph.edges = [GraphEdge("1_10_0", "1_11_1", 0.7)]
    return graph


def example(prompt: str, target: str = " answer") -> dict:
    return {
        "family": "test",
        "prompt": prompt,
        "target": target,
        "target_token": target,
        "possible_answers": ("answer", "other"),
        "sampling": {"disposition": "keep_correct"},
        "graph": graph_to_json(sample_graph()),
    }


def test_promptless_is_a_separate_cohort_taskset() -> None:
    assert OriginalCircuitTaskset is CircuitTaskset
    assert ExportedDiscoveryTaskset is DiscoveryTaskset
    assert CohortGraphTools in DiscoveryTask.tools


def test_prompt_contains_goal_but_no_subject_prompts() -> None:
    prompt = cohort_prompt("capital-city goal", "Austin", 4, 4)
    assert "capital-city goal" in prompt
    assert "Austin" in prompt
    assert "Hidden goal-cohort prompts: 4" in prompt
    assert "Dallas" not in prompt
    assert "circuit_view_graph" in SYSTEM_PROMPT
    assert "circuit_run_prompt" not in SYSTEM_PROMPT


def test_promptless_graph_hides_prompt_tokens_labels_and_output() -> None:
    view = promptless_graph(sample_graph(), 2)
    assert len(view.feature_nodes) == 2
    assert view.tokens == ["<hidden>"]
    assert view.top_outputs == []
    assert all(node.label == "" and node.token == "" for node in view.nodes.values())


def test_weighted_supernode_parser_requires_names_weights_and_stable_ids() -> None:
    parsed, error = parse_weighted_nodes(
        json.dumps(
            {
                "supernodes": [
                    {
                        "name": "goal",
                        "nodes": ["1_10_7", "2_20"],
                        "intervention_weight": 1.25,
                    }
                ]
            }
        ),
        require_supernodes=True,
    )
    assert error == ""
    assert parsed == [("1_10", 1.25), ("2_20", 1.25)]
    invalid, error = parse_weighted_nodes(
        '{"nodes": [{"id": "1_2", "intervention_weight": 1}]}',
        require_supernodes=True,
    )
    assert invalid is None
    assert "supernodes" in error


def test_promptless_verifier_requires_whole_cohort_and_clean_controls() -> None:
    passed = promptless_score(
        {
            "cohort": [{"shifted_to_target": True}] * 4,
            "controls": [{"intervened_target_is_top": False}] * 4,
        },
        1.0,
        0.0,
    )
    assert passed["reward"] == 1.0
    assert passed["cohort_success_rate"] == 1.0
    assert passed["control_activation_rate"] == 0.0

    one_missed = promptless_score(
        {
            "cohort": [
                {"shifted_to_target": True},
                {"shifted_to_target": False},
            ],
            "controls": [{"intervened_target_is_top": False}],
        },
        1.0,
        0.0,
    )
    assert one_missed["reward"] == 0.0
    target_hack = promptless_score(
        {
            "cohort": [{"shifted_to_target": True}],
            "controls": [{"intervened_target_is_top": True}],
        },
        1.0,
        0.0,
    )
    assert target_hack["reward"] == 0.0


def test_builder_groups_same_answers_and_uses_different_answer_controls() -> None:
    records = [example(f"positive {index}") for index in range(4)]
    records += [example(f"control {index}", " other") for index in range(4)]
    rows = cohort_rows(records, tasks=1, cohort_size=3, control_size=2, seed=7)
    assert len(rows) == 1
    row = rows[0]
    assert {item["target"].strip() for item in row["cohort"]} == {"answer"}
    assert all(item["target"].strip() != "answer" for item in row["controls"])


def test_taskset_loads_stored_maximal_graph_cohorts(tmp_path, monkeypatch) -> None:
    import circuit_supernodes.rediscovery as environment

    row = {
        "family": "test",
        "target_concept": "answer goal",
        "desired_answer": "answer",
        "desired_target_token": " answer",
        "cohort": [example("secret positive")],
        "controls": [example("secret control", " other")],
    }
    path = tmp_path / "promptless-supernodes.jsonl"
    path.write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(environment, "DATA_DIR", tmp_path)

    task = next(iter(DiscoveryTaskset(DiscoveryTasksetConfig()).load()))
    assert task.data.cohort[0]["prompt"] == "secret positive"
    assert "secret positive" not in task.data.prompt
    assert len(task.data.cohort[0]["graph"]["nodes"]) == 6


def test_graph_tool_lets_policy_choose_subset_without_prompt_leak() -> None:
    async def exercise() -> None:
        tools = CohortGraphTools(vf.ToolsetConfig())
        await tools.setup_task(
            SimpleNamespace(cohort=[example("secret positive")], max_view_features=4096)
        )
        result = await tools.view_graph(0, 2)
        assert result["returned_feature_count"] == 2
        assert result["available_feature_count"] == 6
        assert "secret" not in result["graph"]
        assert "human label" not in result["graph"]

    asyncio.run(exercise())


def test_retry_reports_attempt_count_and_preserves_cause(monkeypatch) -> None:
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
        assert str(caught.value.__cause__) == "failure 3"

    asyncio.run(exercise())
