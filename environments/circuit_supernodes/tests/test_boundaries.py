import asyncio
from typing import Any

from circuit_supernodes.grading.ablation import AblationHoldoutGrader, ablation_score
from circuit_supernodes.models import Circuit, DatasetExample, TracerGraphPayload
from circuit_supernodes.runtime.modal import ModalSubject


class FakeCall:
    def __init__(self, value: Any) -> None:
        self.value = value

    def get(self) -> Any:
        return self.value


class FakeModalSubject:
    def __init__(self, responses: dict[str, Any]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def call(self, method: str, *args: Any) -> Any:
        self.calls.append((method, args))
        return self.responses[method]

    def call_sync(self, method: str, *args: Any) -> Any:
        self.calls.append((method, args))
        return self.responses[method]

    def map(self, method: str, *inputs: Any, order_outputs: bool = True):
        assert order_outputs
        self.calls.append((method, inputs))
        return iter(self.responses[method])

    def spawn(self, method: str, *args: Any) -> FakeCall:
        self.calls.append((method, args))
        return FakeCall(self.responses[method])


def graph() -> TracerGraphPayload:
    return TracerGraphPayload.model_validate(
        {
            "metadata": {"prompt_tokens": ["x"]},
            "nodes": [
                {
                    "node_id": "1_2_0",
                    "feature_type": "feature",
                    "layer": 1,
                    "ctx_idx": 0,
                    "influence": 0.5,
                },
                {
                    "node_id": "logit",
                    "feature_type": "logit",
                    "clerp": " target",
                    "token_prob": 0.9,
                    "influence": 0.5,
                },
            ],
            "links": [],
        }
    )


def example(prompt: str, *, corruption=None) -> DatasetExample:
    return DatasetExample(
        prompt=prompt,
        target_token=" clean",
        graph=graph(),
        corruption=corruption or [],
    )


def test_modal_subject_centralizes_named_batch_operations() -> None:
    subject = ModalSubject("app", "Class")
    modal_subject = FakeModalSubject(
        {
            "sample_until_answer": [{"status": "matched"}],
            "attribute_prompt": [{"graph": 1}],
        }
    )
    subject.map = modal_subject.map  # ty: ignore[invalid-assignment]
    subject.spawn = modal_subject.spawn  # ty: ignore[invalid-assignment]

    assert subject.sample_answers([{"prompt": "p"}]) == [{"status": "matched"}]
    assert subject.attribution_graphs(["p"]) == [{"graph": 1}]
    assert subject.spawn_attribution_graph("p").get() == [{"graph": 1}]


def test_grader_accepts_cohorts_and_thresholds_directly() -> None:
    async def exercise() -> None:
        subject = FakeModalSubject(
            {
                "evaluate_ablation_cohort": {
                    "positives": [
                        {
                            "expected_token": " target",
                            "baseline_expected_is_top": True,
                            "changed_top_token": True,
                        },
                        {
                            "expected_token": " target",
                            "baseline_expected_is_top": True,
                            "changed_top_token": True,
                        },
                    ],
                    "controls": [
                        {
                            "expected_token": " target",
                            "baseline_expected_is_top": True,
                            "preserved_top_token": True,
                        }
                    ],
                }
            }
        )
        result = await AblationHoldoutGrader(subject).grade(
            Circuit(supernodes={"goal": {"1_2": 0.5}}),
            [example("visible")],
            [example("heldout")],
            [example("control")],
            min_visible_success=1.0,
            min_holdout_success=1.0,
            max_control_failure=0.0,
        )
        assert result["reward"] == 1.0
        positives, controls = subject.calls[0][1]
        assert positives[0]["expected_token"] == " target"
        assert positives[0]["interventions"] == [(1, 2, 0, -0.5)]
        assert controls[0]["expected_token"] == " target"

    asyncio.run(exercise())


def test_ablation_score_keeps_three_threshold_gates() -> None:
    result = ablation_score(
        {
            "positives": [
                {"changed_top_token": True},
                {"changed_top_token": True},
            ],
            "controls": [{"preserved_top_token": True}],
        },
        visible_count=1,
        min_visible_success=1.0,
        min_holdout_success=1.0,
        max_control_failure=0.0,
    )
    assert result["reward"] == 1.0
