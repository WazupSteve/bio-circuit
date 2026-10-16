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
    assert task.data.negate_interventions is False
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


@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
@pytest.mark.parametrize("family", ["visible", "rediscovery", "restoration"])
@pytest.mark.parametrize(
    "failure",
    [None, "positive_baseline", "secondary_gate", "malformed", "circuit_miss"],
)
def test_reward_handles_replay_validation_failures(
    tmp_path, monkeypatch, family, failure, lazy
) -> None:
    import asyncio
    from types import SimpleNamespace

    from circuit_supernodes.tasksets import shared

    module = {
        "visible": visible,
        "rediscovery": rediscovery,
        "restoration": restoration,
    }[family]
    write_row(
        tmp_path / "task.jsonl",
        {
            "family": "family",
            "goal": "goal",
            "example_prompts": [example("visible")],
            "heldout_prompts": [example("heldout", corruption=[[1, 2, 0, 4.0]])],
            "control_prompts": [example("control")],
        },
    )
    monkeypatch.setattr(module, "DATA_DIR", tmp_path)
    if lazy:
        monkeypatch.setattr(shared, "LAZY_DATASET_BYTES", 0)
    taskset_type = (
        module.DiscoveryTaskset if family == "rediscovery" else module.CircuitTaskset
    )
    config_type = (
        module.DiscoveryTasksetConfig
        if family == "rediscovery"
        else module.CircuitTasksetConfig
    )
    task = next(taskset_type(config_type(dataset_file="task.jsonl")).load())
    assert (task.data.dataset_path is not None) is lazy

    class Subject:
        calls = 0

        async def call(self, method, *args):
            self.calls += 1
            if method == "restore_output":
                result = {
                    "target_token": " output",
                    "standard_target_is_top": failure != "positive_baseline",
                    "corruption_changed_top_token": failure != "secondary_gate",
                    "restored_target_is_top": failure != "circuit_miss",
                    "restored_standard_kl": 0.0,
                }
                if failure == "malformed":
                    del result["restored_standard_kl"]
                return result
            assert method == "evaluate_ablation_cohort"
            positives, controls = args
            result = {
                "positives": [
                    {
                        "expected_token": row["expected_token"],
                        "baseline_expected_is_top": failure != "positive_baseline",
                        "changed_top_token": failure != "circuit_miss",
                    }
                    for row in positives
                ],
                "controls": [
                    {
                        "expected_token": row["expected_token"],
                        "baseline_expected_is_top": failure != "secondary_gate",
                        "preserved_top_token": True,
                    }
                    for row in controls
                ],
            }
            if failure == "malformed":
                del result["positives"][0]["changed_top_token"]
            return result

    subject = Subject()
    monkeypatch.setattr(shared, "subject_from_config", lambda config: subject)
    monkeypatch.setattr(restoration, "subject_from_config", lambda config: subject)
    trace = SimpleNamespace(
        info={},
        last_reply=json.dumps(
            {
                "supernodes": [
                    {"name": "goal", "nodes": ["1_2"], "intervention_weight": 1.0}
                ]
            }
        ),
    )

    async def exercise():
        reward = getattr(
            task,
            {
                "visible": "circuit_faithfulness",
                "rediscovery": "goal_supernode_intervention",
                "restoration": "suppressed_feature_restoration",
            }[family],
        )
        assert await reward(trace) == (1.0 if failure is None else 0.0)
        result = trace.info[task.score_cache_key]
        validation_failed = failure in {
            "positive_baseline",
            "secondary_gate",
            "malformed",
        }
        drift = failure in {"positive_baseline", "secondary_gate"}
        assert result["valid"] is not validation_failed
        if validation_failed:
            assert result["failure"] == (
                "baseline_drift" if drift else "grading_validation_error"
            )
            assert result["validation_errors"]
            assert all("input" not in error for error in result["validation_errors"])
            json.dumps(result)
        calls = subject.calls
        assert await task.grading_validation_metrics(trace) == {
            "grading_validation_error": float(validation_failed),
            "baseline_drift": float(drift),
            "non_feature_node_ids": 0.0,
        }
        assert await reward(trace) == result["reward"]
        assert subject.calls == calls

    asyncio.run(exercise())


def lazy_and_eager_tasks(tmp_path, monkeypatch, module, rows: list[dict]):
    from circuit_supernodes.tasksets import shared

    (tmp_path / "task.jsonl").write_text(
        "\n\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n"
    )
    monkeypatch.setattr(module, "DATA_DIR", tmp_path)
    taskset_type = getattr(module, "DiscoveryTaskset", None) or module.CircuitTaskset
    config_type = (
        getattr(module, "DiscoveryTasksetConfig", None) or module.CircuitTasksetConfig
    )
    eager = list(taskset_type(config_type(dataset_file="task.jsonl")).load())
    monkeypatch.setattr(shared, "LAZY_DATASET_BYTES", 0)
    lazy = list(taskset_type(config_type(dataset_file="task.jsonl")).load())
    return eager, lazy


@pytest.mark.parametrize("module", [visible, rediscovery, restoration])
def test_lazy_tasks_match_eager_tasks(tmp_path, monkeypatch, module) -> None:
    rows = [
        {
            "family": f"family-{index}",
            "goal": f'goal "{index}"',
            "example_prompts": [example(f"example {index}")] if index else [],
            "heldout_prompts": [
                example(
                    f"heldout {index}-{position}",
                    corruption=[[1, index, position, 2.5], [3, 4, 5, -1e-3]],
                )
                for position in range(6)
            ],
            "control_prompts": [example(f"control {index}")],
        }
        for index in range(3)
    ]
    eager, lazy = lazy_and_eager_tasks(tmp_path, monkeypatch, module, rows)

    assert len(lazy) == len(eager) == 3
    for eager_task, lazy_task in zip(eager, lazy, strict=True):
        assert lazy_task.data.dataset_path is not None
        assert not lazy_task.data.heldout_prompts
        assert lazy_task.data.model_dump() == eager_task.data.model_dump()
        assert lazy_task.data.corruption_spec == eager_task.data.corruption_spec
        assert lazy_task.grading_cohorts() == eager_task.grading_cohorts()


def test_lazy_restoration_without_corruption_matches_eager(
    tmp_path, monkeypatch
) -> None:
    rows = [
        {
            "family": "family",
            "example_prompts": [],
            "heldout_prompts": [
                example("no corruption"),
                example("later", corruption=[[1, 2, 0, 4.0]]),
            ],
            "control_prompts": [example("control")],
        }
    ]
    eager, lazy = lazy_and_eager_tasks(tmp_path, monkeypatch, restoration, rows)
    assert lazy[0].data.corruption_spec == eager[0].data.corruption_spec == []
