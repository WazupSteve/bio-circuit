import asyncio
import json
from types import SimpleNamespace

import pytest
import verifiers.v1 as vf
from circuit_supernodes.models import TracerGraphPayload
from circuit_supernodes.runtime import circuit_tracer
from circuit_supernodes.runtime.circuit_tracer import CircuitExperimentTools
from circuit_supernodes.runtime.graph_store import MAX_RESPONSE_BYTES, GraphStore
from mcp.server.fastmcp import FastMCP


def graph():
    return TracerGraphPayload.model_validate(
        {
            "metadata": {"prompt_tokens": ["prompt", " token"]},
            "nodes": [
                {
                    "node_id": f"1_{i}_0",
                    "feature_type": "feature",
                    "layer": 1,
                    "influence": i + 1.0,
                    "clerp": "long label " * 1000,
                }
                for i in range(100)
            ]
            + [{"node_id": "logit", "feature_type": "logit", "clerp": "secret-output"}],
            "links": [
                {"source": "1_0_0", "target": "1_1_0", "weight": -0.5},
                {"source": "1_1_0", "target": "1_0_0", "weight": 0.25},
                {"source": "1_0_0", "target": "logit", "weight": 1.0},
            ],
        }
    )


def test_toolset_registers_fastmcp_schemas():
    tools = CircuitExperimentTools(vf.ToolsetConfig())
    tools._register(FastMCP("schema-test"))


def test_store_paginates_all_nodes_with_byte_budget_and_filters():
    store = GraphStore()
    try:
        summary = store.write(graph(), include_outputs=True)
        assert summary["feature_count"] == 100
        offset, ids = 0, []
        while offset is not None:
            page = store.query(
                view="nodes", offset=offset, limit=50, node_id=None, kind="feature"
            )
            assert len(json.dumps(page).encode()) <= MAX_RESPONSE_BYTES
            assert page["items"]
            ids.extend(item["node_id"] for item in page["items"])
            assert all(item["text_truncated"] for item in page["items"])
            offset = page["next_offset"]
        assert ids == [f"1_{i}_0" for i in range(100)]
        page = store.query(view="nodes", offset=0, limit=20, node_id="1_90", kind=None)
        assert [item["node_id"] for item in page["items"]] == ["1_90_0"]
        page = store.query(view="edges", offset=0, limit=20, node_id="1_0_0", kind=None)
        assert [item["weight"] for item in page["items"]] == [1.0, -0.5, 0.25]
        page = store.query(view="tokens", offset=1, limit=1, node_id=None, kind=None)
        assert page["items"] == [{"position": 1, "token": " token"}]
        assert page["next_offset"] is None
    finally:
        store.close()
    assert not store.path.exists()


def test_redacted_store_omits_logits_and_all_incident_edges():
    store = GraphStore()
    try:
        summary = store.write(graph(), include_outputs=False)
        assert summary["outputs_redacted"]
        assert "top_outputs" not in summary
        assert summary["edge_count"] == 2
        assert b"secret-output" not in store.path.read_bytes()
        for view, kwargs in [
            ("nodes", {"kind": "logit"}),
            ("edges", {"node_id": "logit"}),
        ]:
            options = {
                "view": view,
                "offset": 0,
                "limit": 20,
                "node_id": None,
                "kind": None,
            }
            options.update(kwargs)
            assert store.query(**options)["items"] == []
    finally:
        store.close()


@pytest.mark.parametrize(
    "query",
    [
        {"limit": 0},
        {"limit": 10000},
        {"offset": -1},
        {"view": "invalid"},
        {"kind": "invalid"},
    ],
)
def test_invalid_queries_return_small_errors(query):
    async def exercise():
        tools = CircuitExperimentTools(vf.ToolsetConfig())

        async def attribute(prompt):
            return graph()

        tools._graph = attribute
        summary = await tools.attribute("prompt")
        result = await tools.query_graph(summary["graph_id"], **query)
        assert "error" in result
        tools._clear_graph()

    asyncio.run(exercise())


def test_replacement_failure_and_task_reset_delete_previous_file(monkeypatch):
    async def exercise():
        tools = CircuitExperimentTools(vf.ToolsetConfig())
        assert "error" in await tools.query_graph("missing")
        old_path = None

        async def attribute(prompt):
            if old_path is not None:
                assert not old_path.exists()
            if prompt == "fail":
                raise ValueError("attribution failed")
            return graph()

        tools._graph = attribute
        first = await tools.attribute("first")
        old_path = tools._graph_store.path
        second = await tools.attribute("second")
        assert first["graph_id"] != second["graph_id"]
        assert "error" in await tools.query_graph(first["graph_id"])
        assert (await tools.query_graph(second["graph_id"]))["items"]
        old_path = tools._graph_store.path
        assert await tools.attribute("fail") == {"error": "attribution failed"}
        assert tools._graph_store is None
        assert "error" in await tools.query_graph(second["graph_id"])
        await tools.attribute("third")
        old_path = tools._graph_store.path
        monkeypatch.setattr(circuit_tracer, "subject_from_config", lambda config: None)
        await tools.setup_task(SimpleNamespace(subject_config=None, corruption_spec=[]))
        assert not old_path.exists()
        assert tools._graph_store is None

    asyncio.run(exercise())
