import asyncio
import json

import pytest
import verifiers.v1 as vf
from circuit_supernodes.models import TracerGraphPayload, TracerNode
from circuit_supernodes.runtime import responses as response_runtime
from circuit_supernodes.runtime.circuit_tracer import CircuitExperimentTools
from circuit_supernodes.runtime.graphs import (
    feature_nodes,
    influence_deltas,
    node_by_id,
    parse_coordinate,
    resolve_interventions,
    select_features,
    stable_feature_id,
    top_outputs,
    weighted_nodes,
)
from circuit_supernodes.runtime.responses import (
    circuit_from_response,
    render_feature_view,
    render_graph,
)


def ranked_graph(count: int = 75) -> TracerGraphPayload:
    nodes = []
    cumulative = 0.0
    for index in range(count):
        cumulative += count - index
        nodes.append(
            TracerNode(
                node_id=f"0_{index}_0",
                feature_type="feature",
                layer=0,
                ctx_idx=0,
                influence=cumulative,
                activation=index / 10,
            )
        )
    nodes.append(
        TracerNode(
            node_id="logit",
            feature_type="logit",
            clerp='"y"',
            token_prob=0.5,
            influence=cumulative,
        )
    )
    return TracerGraphPayload(metadata={"prompt_tokens": ["x"]}, nodes=nodes, links=[])


def test_moved_graph_helpers_preserve_ranking_rendering_and_resolution() -> None:
    graph = ranked_graph()
    deltas = influence_deltas(graph)
    assert deltas["0_0_0"] == 75
    assert deltas["0_1_0"] == 74
    assert top_outputs(graph) == [("y", 0.5)]
    assert parse_coordinate("01_002_3") == (1, 2, 3)
    assert stable_feature_id("01_002_3") == "1_2"

    view = select_features(graph, 3)
    assert [node.node_id for node in feature_nodes(view)] == [
        "0_0_0",
        "0_1_0",
        "0_2_0",
    ]
    retained = select_features(graph, 3, {"0_74_0"})
    assert "0_74_0" in node_by_id(retained)
    assert resolve_interventions(graph, [("0_2", 0.5), ("9_9", 1.0)]) == [
        (0, 2, 0, 0.5)
    ]
    rendered = render_graph(view)
    assert "PROMPT TOKENS: x" in rendered
    assert "TOP OUTPUTS: 'y'" in rendered
    assert "0_2_0" in rendered


def test_graph_view_enforces_exported_feature_budget() -> None:
    payload = render_feature_view(ranked_graph(), feature_count=3)
    assert payload["returned_feature_count"] == 3
    assert payload["available_feature_count"] == 75
    for count in (0, 76):
        with pytest.raises(ValueError, match="between 1 and 75"):
            render_feature_view(ranked_graph(), feature_count=count)


@pytest.mark.parametrize(
    ("payload", "kwargs", "expected"),
    [
        (
            {"nodes": [{"id": "1_2_9", "intervention_weight": 0.5}]},
            {},
            {"submitted": {"1_2": 0.5}},
        ),
        (
            {
                "supernodes": [
                    {
                        "name": "role",
                        "nodes": ["1_2", "3_4"],
                        "intervention_weight": 1.25,
                    }
                ]
            },
            {"require_supernodes": True},
            {"role": {"1_2": 1.25, "3_4": 1.25}},
        ),
        (
            {
                "supernodes": [
                    {
                        "name": "role",
                        "nodes": [
                            {"id": "1_2", "intervention_weight": -0.5},
                            "3_4",
                        ],
                        "intervention_weight": 2.0,
                    }
                ]
            },
            {},
            {"role": {"1_2": -0.5, "3_4": 2.0}},
        ),
        (
            {"supernodes": [{"name": "role", "nodes": ["1_2"]}]},
            {"default_weight": 1.0},
            {"role": {"1_2": 1.0}},
        ),
    ],
)
def test_response_parser_accepts_every_supported_wire_shape(
    payload: dict, kwargs: dict, expected: dict
) -> None:
    circuit, error = circuit_from_response(f"analysis\n{json.dumps(payload)}", **kwargs)
    assert error == ""
    assert circuit is not None
    assert circuit.supernodes == expected
    assert weighted_nodes(circuit) == [
        item for group in expected.values() for item in group.items()
    ]


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"nodes": [{"id": "bad", "intervention_weight": 1.0}]}, "invalid FEATURE"),
        ({"nodes": [{"id": "1_2", "intervention_weight": 0.0}]}, "nonzero"),
        (
            {"nodes": [{"id": "1_2", "intervention_weight": 9.0}]},
            "less than or equal to 8",
        ),
        (
            {
                "supernodes": [
                    {"name": "Role", "nodes": ["1_2"], "intervention_weight": 1.0},
                    {"name": "role", "nodes": ["3_4"], "intervention_weight": 1.0},
                ]
            },
            "names must be unique",
        ),
        (
            {
                "supernodes": [
                    {"name": "a", "nodes": ["1_2"], "intervention_weight": 1.0},
                    {"name": "b", "nodes": ["1_2_9"], "intervention_weight": 1.0},
                ]
            },
            "duplicate stable FEATURE",
        ),
    ],
)
def test_response_parser_preserves_rejection_rules(payload: dict, message: str) -> None:
    circuit, error = circuit_from_response(json.dumps(payload))
    assert circuit is None
    assert message in error


def test_visible_exact_group_and_rediscovery_named_group_rules() -> None:
    circuit, error = circuit_from_response(
        '{"supernodes":[{"name":"a","nodes":["1_2"]},{"name":"b","nodes":["3_4"]}]}',
        default_weight=1.0,
        exact_groups=1,
    )
    assert circuit is None
    assert error == "submit exactly 1 supernode"
    circuit, error = circuit_from_response(
        '{"nodes":[{"id":"1_2","intervention_weight":1}]}',
        require_supernodes=True,
    )
    assert circuit is None
    assert "supernodes" in error


def test_response_parser_rejects_nonfinite_and_excessive_submissions() -> None:
    circuit, error = circuit_from_response(
        '{"nodes":[{"id":"1_2","intervention_weight":NaN}]}'
    )
    assert circuit is None
    assert "finite" in error

    too_many_groups = {
        "supernodes": [
            {
                "name": f"group-{index}",
                "nodes": [f"0_{index}"],
                "intervention_weight": 1,
            }
            for index in range(9)
        ]
    }
    circuit, error = circuit_from_response(json.dumps(too_many_groups))
    assert circuit is None
    assert "at most 8" in error

    too_many_nodes = {
        "nodes": [{"id": f"0_{index}", "intervention_weight": 1} for index in range(61)]
    }
    circuit, error = circuit_from_response(json.dumps(too_many_nodes))
    assert circuit is None
    assert "at most 60" in error


def test_response_parser_forgives_any_processing_failure(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise RuntimeError("unexpected parser failure")

    monkeypatch.setattr(response_runtime, "_normalize_circuit", fail)
    circuit, error = circuit_from_response('{"nodes":["1_2"]}')
    assert circuit is None
    assert error == "unexpected parser failure"


def test_shared_tools_use_graph_helpers_for_arbitrary_prompts() -> None:
    class Subject:
        async def call(self, method: str, *args):
            if method == "attribute_prompt":
                return ranked_graph().model_dump()
            positives, controls = args
            assert controls == []
            assert positives[0]["interventions"] == [(0, 0, 0, -1.0)]
            return {
                "positives": [{"changed_top_token": True}],
                "controls": [],
            }

    async def exercise() -> None:
        tools = CircuitExperimentTools(vf.ToolsetConfig())
        tools._subject = Subject()
        attributed = await tools.attribute("source", 3)
        assert attributed["returned_feature_count"] == 3
        result = await tools.intervene("source", "0_0_9", "target", -1.0)
        assert result["changed_top_token"] is True

    asyncio.run(exercise())


def test_shared_tools_return_errors_for_unexpected_failures() -> None:
    async def exercise() -> None:
        tools = CircuitExperimentTools(vf.ToolsetConfig())

        async def fail(prompt: str):
            raise RuntimeError(f"backend failed for {prompt}")

        tools._graph = fail
        assert await tools.attribute("source", 3) == {
            "error": "backend failed for source"
        }
        assert await tools.intervene("source", "0_0", "target", -1.0) == {
            "error": "backend failed for source"
        }

    asyncio.run(exercise())
