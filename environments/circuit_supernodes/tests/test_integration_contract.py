import json

import pytest
from circuit_supernodes.runtime.graphs import feature_nodes
from circuit_supernodes.runtime.modal import SUBJECT
from circuit_supernodes.tasksets import rediscovery, restoration, visible


def graph(token: str = " output") -> dict:
    return {
        "metadata": {"prompt_tokens": ["prompt"]},
        "nodes": [
            {
                "node_id": "1_2_0",
                "feature_type": "feature",
                "ctx_idx": 0,
                "layer": 1,
                "influence": 0.4,
                "activation": 0.2,
            },
            {
                "node_id": "logit",
                "feature_type": "logit",
                "ctx_idx": 0,
                "layer": 2,
                "influence": 0.4,
                "clerp": token,
                "token_prob": 0.9,
            },
        ],
        "links": [],
    }


def example(prompt: str, *, corruption=None) -> dict:
    payload = {"prompt": prompt, "target_token": " output", "graph": graph()}
    if corruption is not None:
        payload["corruption"] = corruption
    return payload


def write_row(path, row: dict) -> None:
    path.write_text(json.dumps(row) + "\n")


def test_visible_taskset_loads_minimal_three_cohort_row(tmp_path, monkeypatch) -> None:
    write_row(
        tmp_path / "visible.jsonl",
        {
            "family": "family",
            "goal": "semantic goal",
            "example_prompts": [example("example")],
            "heldout_prompts": [example("heldout")],
            "control_prompts": [example("control")],
        },
    )
    monkeypatch.setattr(visible, "DATA_DIR", tmp_path)
    task = next(
        iter(
            visible.CircuitTaskset(
                visible.CircuitTasksetConfig(dataset_file="visible.jsonl")
            ).load()
        )
    )
    assert task.data.goal == "semantic goal"
    assert [row.prompt for row in task.data.example_prompts] == ["example"]
    assert [row.prompt for row in task.data.heldout_prompts] == ["heldout"]
    assert "prompt='example'" in task.data.prompt
    assert "semantic goal" not in task.data.prompt
    assert task.data.min_heldout_success == 1.0


def test_rediscovery_derives_prompt_metadata_from_canonical_row(
    tmp_path, monkeypatch
) -> None:
    write_row(
        tmp_path / "rediscovery.jsonl",
        {
            "family": "family",
            "goal": "capital-city",
            "example_prompts": [],
            "heldout_prompts": [example("secret positive")],
            "control_prompts": [example("secret control")],
        },
    )
    monkeypatch.setattr(rediscovery, "DATA_DIR", tmp_path)
    task = next(
        iter(
            rediscovery.DiscoveryTaskset(
                rediscovery.DiscoveryTasksetConfig(dataset_file="rediscovery.jsonl")
            ).load()
        )
    )
    assert "capital-city" in task.data.prompt
    assert "secret positive" not in task.data.prompt
    assert task.data.goal == "capital-city"


def test_restoration_uses_first_heldout_corruption_spec(tmp_path, monkeypatch) -> None:
    first = [[1, 2, 0, 4.0], [1, 3, 0, 2.0]]
    second = [[1, 2, 7, 4.0], [1, 3, 7, 2.0]]
    write_row(
        tmp_path / "restoration.jsonl",
        {
            "family": "family",
            "example_prompts": [],
            "heldout_prompts": [
                example("heldout-one", corruption=first),
                example("heldout-two", corruption=second),
            ],
            "control_prompts": [example("control")],
        },
    )
    monkeypatch.setattr(restoration, "DATA_DIR", tmp_path)
    task = next(
        iter(
            restoration.CircuitTaskset(
                restoration.CircuitTasksetConfig(dataset_file="restoration.jsonl")
            ).load()
        )
    )
    assert task.data.corruption_spec == [(1, 2, 4.0), (1, 3, 2.0)]
    assert "corruption_spec" not in task.data.model_dump()
    assert "4.0" not in task.data.prompt
    assert "2.0" not in task.data.prompt
    assert len(feature_nodes(task.data.heldout_prompts[0].graph)) == 1


def test_old_rows_fail_clearly_instead_of_being_partially_loaded(
    tmp_path, monkeypatch
) -> None:
    old = {
        "task_type": "visible",
        "family": "family",
        "goal": "goal",
        "example_prompts": [],
        "heldout_prompts": [
            {
                **example("heldout"),
                "family": "family",
                "sampling": {"disposition": "keep_correct"},
            }
        ],
        "control_prompts": [example("control")],
    }
    write_row(tmp_path / "old.jsonl", old)
    monkeypatch.setattr(visible, "DATA_DIR", tmp_path)
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        next(
            iter(
                visible.CircuitTaskset(
                    visible.CircuitTasksetConfig(dataset_file="old.jsonl")
                ).load()
            )
        )


def test_all_tasksets_share_one_subject_backend() -> None:
    assert SUBJECT.app_name == "circuit-supernodes"
    assert SUBJECT.class_name == "Attributor"


def test_all_plugin_entrypoints_remain_independent() -> None:
    from circuit_supernodes import CircuitTaskset as VisibleTaskset
    from circuit_supernodes.runtime.circuit_tracer import CircuitExperimentTools

    assert (
        len(
            {
                VisibleTaskset,
                rediscovery.DiscoveryTaskset,
                restoration.CircuitTaskset,
            }
        )
        == 3
    )
    assert (
        visible.CircuitTask.tools
        == rediscovery.DiscoveryTask.tools
        == restoration.CircuitTask.tools
        == (CircuitExperimentTools,)
    )
