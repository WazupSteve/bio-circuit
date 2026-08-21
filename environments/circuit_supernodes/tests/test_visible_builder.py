from circuit_supernodes.models import TracerGraphPayload, TracerNode
from scripts.build_visible_dataset import cohort_rows


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
