import pytest
from circuit_supernodes.models import (
    Circuit,
    CircuitDatasetRow,
    SamplingRequest,
    TracerGraphPayload,
    TracerNode,
)
from pydantic import BaseModel, ValidationError


def graph_payload() -> dict:
    return {
        "metadata": {"prompt_tokens": ["x"], "exporter_field": True},
        "nodes": [
            {
                "node_id": "1_2_0",
                "feature_type": "feature",
                "ctx_idx": 0,
                "layer": "1",
                "influence": 0.5,
                "activation": None,
                "unused": "discarded",
            }
        ],
        "links": [],
        "tokens": ["legacy"],
        "edges": [{"source": "legacy", "target": "legacy"}],
    }


def test_tracer_models_are_canonical_strict_and_ignore_exporter_extras() -> None:
    assert issubclass(TracerNode, BaseModel)
    graph = TracerGraphPayload.model_validate(graph_payload())
    assert graph.metadata.prompt_tokens == ["x"]
    assert graph.nodes[0].activation is None
    assert graph.model_dump() == {
        "metadata": {"prompt_tokens": ["x"]},
        "nodes": [
            {
                "node_id": "1_2_0",
                "feature_type": "feature",
                "ctx_idx": 0,
                "layer": "1",
                "influence": 0.5,
                "clerp": "",
                "label": "",
                "activation": None,
                "token_prob": 0.0,
            }
        ],
        "links": [],
    }
    with pytest.raises(ValidationError):
        TracerNode.model_validate(
            {"id": "1_2_0", "type": "feature", "ctx": 0, "layer": 1}
        )
    with pytest.raises(ValidationError):
        TracerNode(
            node_id="1_2_0",
            feature_type="feature",
            layer=1,
            ctx_idx=0,
            influence="0.5",
        )


def test_circuit_is_the_single_weighted_representation() -> None:
    circuit = Circuit(supernodes={"role": {"1_2": 0.5, "3_4": -1.0}})
    assert circuit.model_dump() == {"supernodes": {"role": {"1_2": 0.5, "3_4": -1.0}}}
    with pytest.raises(ValidationError, match="names must be unique"):
        Circuit(supernodes={"Role": {"1_2": 1.0}, "role": {"3_4": 1.0}})
    with pytest.raises(ValidationError, match="duplicate stable FEATURE"):
        Circuit(supernodes={"a": {"1_2": 1.0}, "b": {"1_2": -1.0}})
    with pytest.raises(ValidationError):
        Circuit(supernodes={"role": {"not-a-feature": 1.0}})
    with pytest.raises(ValidationError):
        Circuit(supernodes={"role": {"1_2": 9.0}})
    with pytest.raises(ValidationError):
        Circuit(supernodes={"role": {"1_2": 0.0}})


def test_dataset_row_contains_only_runtime_fields() -> None:
    example = {
        "prompt": "prompt",
        "target_token": " token",
        "graph": graph_payload(),
    }
    row = CircuitDatasetRow.model_validate(
        {
            "family": "test",
            "goal": "goal",
            "desired_answer": None,
            "example_prompts": [],
            "heldout_prompts": [example],
            "control_prompts": [],
        }
    )
    assert row.heldout_prompts[0].corruption == []
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CircuitDatasetRow.model_validate(
            {
                "task_type": "visible",
                "family": "test",
                "example_prompts": [],
                "heldout_prompts": [example],
                "control_prompts": [],
            }
        )
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CircuitDatasetRow.model_validate(
            {
                "family": "test",
                "example_prompts": [],
                "heldout_prompts": [{**example, "sampling": {}}],
                "control_prompts": [],
            }
        )


def test_live_sampling_boundary_remains_strict() -> None:
    with pytest.raises(ValidationError):
        SamplingRequest.model_validate(
            {
                "prompt": "prompt",
                "possible_answers": ["yes"],
                "max_new_tokens": "10",
            }
        )
