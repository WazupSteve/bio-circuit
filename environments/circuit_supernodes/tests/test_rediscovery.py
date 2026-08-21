from circuit_supernodes import CircuitTaskset as PackageCircuitTaskset
from circuit_supernodes.runtime.circuit_tracer import CircuitExperimentTools
from circuit_supernodes.tasksets.rediscovery import (
    SYSTEM_PROMPT,
    DiscoveryTask,
    DiscoveryTaskset,
    cohort_prompt,
)
from circuit_supernodes.tasksets.visible import CircuitTaskset as VisibleCircuitTaskset
from scripts.build_promptless_dataset import cohort_rows


def graph(token: str) -> dict:
    return {
        "metadata": {"prompt_tokens": ["secret"]},
        "nodes": [
            {
                "node_id": "1_2_0",
                "feature_type": "feature",
                "ctx_idx": 0,
                "layer": 1,
                "influence": 0.5,
            },
            {
                "node_id": "logit",
                "feature_type": "logit",
                "ctx_idx": 0,
                "layer": 2,
                "influence": 0.5,
                "clerp": token,
                "token_prob": 0.8,
            },
        ],
        "links": [],
    }


def record(
    prompt: str,
    target: str = "answer",
    *,
    family: str = "test",
    goal: str | None = None,
    target_token: str | None = None,
) -> dict:
    token = f" {target}" if target_token is None else target_token
    return {
        "family": family,
        "goal": goal or family,
        "prompt": prompt,
        "target": target,
        "target_token": token,
        "graph": graph(token),
    }


def test_promptless_is_a_separate_hidden_cohort_taskset() -> None:
    assert PackageCircuitTaskset is VisibleCircuitTaskset
    assert DiscoveryTask.tools == (CircuitExperimentTools,)
    assert DiscoveryTaskset is not VisibleCircuitTaskset
    prompt = cohort_prompt("capital-city goal", "Austin")
    assert "capital-city goal" in prompt
    assert "Austin" in prompt
    assert "arbitrary prompts" in SYSTEM_PROMPT
    assert "verifier" not in f"{SYSTEM_PROMPT}\n{prompt}".casefold()


def test_builder_separates_goals_answers_tokens_and_emits_minimal_examples() -> None:
    positives = [record(f"positive {index}") for index in range(4)]
    unrelated_same_answer = [
        record(f"unrelated {index}", family="unrelated") for index in range(4)
    ]
    controls = [record(f"control {index}", "other") for index in range(4)]
    rows = cohort_rows(
        [*positives, *unrelated_same_answer, *controls],
        tasks=2,
        cohort_size=3,
        control_size=2,
        seed=7,
    )
    assert {row.goal for row in rows} == {"test", "unrelated"}
    for row in rows:
        assert row.desired_answer == "answer"
        serialized = row.model_dump(exclude_defaults=True, exclude_none=True)
        assert all(
            set(example) == {"prompt", "target_token", "graph"}
            for example in [
                *serialized["heldout_prompts"],
                *serialized["control_prompts"],
            ]
        )
        assert all(example.target_token != " answer" for example in row.control_prompts)


def test_builder_excludes_control_with_colliding_first_token() -> None:
    positives = [record(f"positive {index}") for index in range(3)]
    colliding = record(
        "colliding", "another", family="collision", target_token=" answer"
    )
    safe = record("safe", "other", family="safe")
    row = cohort_rows(
        [*positives, colliding, safe],
        tasks=1,
        cohort_size=3,
        control_size=1,
        seed=2,
    )[0]
    assert [example.prompt for example in row.control_prompts] == ["safe"]
