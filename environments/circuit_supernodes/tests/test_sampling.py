from circuit_supernodes.datagen import CircuitSample
from circuit_supernodes.sampling import (
    MAX_NEW_TOKENS,
    answer_only_prompt,
    classify_result,
    sampling_request,
)
from modal_app.attribution import _first_answer
from scripts.build_dataset import prepare_samples


class FakeRemoteMethod:
    def __init__(self, batches: list[list[dict]]) -> None:
        self.batches = iter(batches)
        self.requests: list[list[dict]] = []

    def map(self, requests: list[dict], *, order_outputs: bool):
        assert order_outputs
        self.requests.append(requests)
        return iter(next(self.batches))


class FakeAttributor:
    def __init__(self, batches: list[list[dict]]) -> None:
        self.sample_until_answer = FakeRemoteMethod(batches)


def test_classify_result_has_three_outcomes() -> None:
    assert classify_result(" fepal", None) == "reject"
    assert (
        classify_result(" fepal", {"status": "exhausted", "generated_text": "which"})
        == "reject"
    )
    assert (
        classify_result(" fepal", {"status": "matched", "matched_answer": "FEPAL"})
        == "keep_correct"
    )
    assert (
        classify_result(" fepal", {"status": "matched", "matched_answer": "prax"})
        == "keep_incorrect"
    )


def test_answer_match_skips_narration_and_observes_boundaries() -> None:
    assert _first_answer("which eventually reaches Sable", ["Sable", "Pine"]) == (
        "Sable",
        25,
        30,
    )
    assert _first_answer("This isomorphic", ["is", "are"]) is None


def test_answer_match_defers_prefix_ambiguity() -> None:
    assert _first_answer("north", ["north", "northeast"]) is None
    assert _first_answer("northeast", ["north", "northeast"]) == (
        "northeast",
        0,
        9,
    )
    assert _first_answer("12", ["1", "12", "16"], final=True) == ("12", 0, 2)


def test_sampling_request_has_hard_limit() -> None:
    request = sampling_request("prompt", ("yes", "no"))
    assert request == {
        "prompt": answer_only_prompt("prompt", ("yes", "no")),
        "possible_answers": ["yes", "no"],
        "max_new_tokens": MAX_NEW_TOKENS,
    }
    assert request["prompt"].endswith("\n\nprompt")
    assert "Answer with only the answer label" in request["prompt"]
    assert "Do not explain" in request["prompt"]
    assert '"yes"' not in request["prompt"]
    assert '"no"' not in request["prompt"]


def test_prepare_samples_keeps_correct_and_rejects_wrong_without_retry() -> None:
    samples = [
        CircuitSample("path", "P1", " Sable", ("Sable", "Pine")),
        CircuitSample("rule", "P2", " fepal", ("fepal", "prax")),
        CircuitSample("rule", "P3", " quell", ("quell", "prax")),
    ]
    initial = [
        {
            "status": "matched",
            "generated_text": " which leads to Sable",
            "matched_answer": "Sable",
            "provided_prompt": "P1 which leads to",
            "target_token": " S",
        },
        {
            "status": "matched",
            "generated_text": " prax",
            "matched_answer": "prax",
            "provided_prompt": "P2",
            "target_token": " pr",
        },
        {"status": "exhausted", "generated_text": " which"},
    ]
    attributor = FakeAttributor([initial])

    prepared, counts = prepare_samples(attributor, samples)

    assert counts == {"keep_correct": 1, "keep_incorrect": 1, "reject": 1}
    assert [item.sample.family for item in prepared] == ["path"]
    assert prepared[0].sample.prompt == "P1 which leads to"
    assert prepared[0].target_token == " S"
    assert len(attributor.sample_until_answer.requests) == 1
