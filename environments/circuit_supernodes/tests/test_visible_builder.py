from collections import defaultdict

from circuit_supernodes.models import TracerGraphPayload, TracerNode
from scripts.build_visible_dataset import (
    _attribution_batch,
    cohort_capacity,
    cohort_rows,
)


def graph() -> TracerGraphPayload:
    return TracerGraphPayload(
        metadata={"prompt_tokens": ["x"]},
        nodes=[
            TracerNode(
                node_id="0_0_0",
                feature_type="feature",
                influence=1.0,
            ),
            TracerNode(
                node_id="logit",
                feature_type="logit",
                clerp='"y"',
                token_prob=1.0,
                influence=1.0,
            ),
        ],
    )


def test_visible_builder_separates_goals_and_emits_minimal_examples() -> None:
    records = [
        {
            "family": family,
            "goal": goal,
            "prompt": f"{goal}-{index}",
            "target_token": "y",
            "graph": graph(),
            "sampling": {"builder-only": True},
        }
        for family, goal in (("a", "goal-a"), ("b", "goal-b"))
        for index in range(4)
    ]
    rows = cohort_rows(
        records,
        tasks=2,
        example_size=1,
        holdout_size=2,
        null_size=2,
        seed=4,
    )
    assert len(rows) == 2
    for row in rows:
        assert len(row.example_prompts) == 1
        assert all(
            example.prompt.startswith(row.goal) for example in row.example_prompts
        )
        assert all(
            example.prompt.startswith(row.goal) for example in row.heldout_prompts
        )
        assert all(
            not example.prompt.startswith(row.goal) for example in row.control_prompts
        )
        serialized = row.model_dump(exclude_defaults=True, exclude_none=True)
        assert all(
            set(example) == {"prompt", "target_token", "graph"}
            for example in [
                *serialized["example_prompts"],
                *serialized["heldout_prompts"],
                *serialized["control_prompts"],
            ]
        )


def test_capacity_counts_disjoint_positive_blocks_and_available_controls() -> None:
    assert cohort_capacity({"a": 60, "b": 24}, positive_size=30, null_size=25) == 0
    assert cohort_capacity({"a": 60, "b": 25}, positive_size=30, null_size=25) == 2
    assert (
        cohort_capacity(
            {f"goal-{index}": 60 for index in range(9)},
            positive_size=30,
            null_size=25,
        )
        == 18
    )


def test_attribution_batch_skips_work_beyond_each_goals_remaining_cap() -> None:
    a_items = [object() for _ in range(10)]
    b_items = [object() for _ in range(10)]
    queues = {"a": list(a_items), "b": list(b_items)}
    selected = _attribution_batch(
        queues,
        defaultdict(int, {"a": 9}),
        max_records_per_goal=10,
        batch_size=10,
    )

    assert len(selected) == 10
    assert sum(item in a_items for item in selected) == 1
    assert sum(item in b_items for item in selected) == 9
