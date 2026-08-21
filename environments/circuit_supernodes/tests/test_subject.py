from circuit_supernodes.models import SamplingResult
from circuit_supernodes.runtime.answers import first_answer
from circuit_supernodes.runtime.answers import first_answer as _first_answer
from circuit_supernodes.runtime.modal import ModalSubject, SubjectConfig
from circuit_supernodes.runtime.sampling import MAX_NEW_TOKENS
from scripts.prompt_generation import CircuitSample
from scripts.sample_preparation import (
    answer_only_prompt,
    classify_result,
    prepare_samples,
    sampling_request,
)


class FakeGenerator:
    def __init__(self, batches: list[list[dict]]) -> None:
        self.batches = iter(batches)
        self.requests: list[list[dict]] = []

    def sample_answers(self, requests: list[dict]) -> list[dict]:
        self.requests.append(requests)
        return next(self.batches)


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


def test_classify_result_rejects_any_result_processing_failure(monkeypatch) -> None:
    def fail(cls, value):
        raise RuntimeError("unexpected result failure")

    monkeypatch.setattr(SamplingResult, "model_validate", classmethod(fail))
    assert classify_result("fepal", {"status": "matched"}) == "reject"


def test_answer_match_skips_narration_and_observes_boundaries() -> None:
    assert _first_answer is first_answer
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
    generator = FakeGenerator([initial])

    prepared, counts = prepare_samples(generator, samples)

    assert counts == {"keep_correct": 1, "keep_incorrect": 1, "reject": 1}
    assert [item.sample.family for item in prepared] == ["path"]
    assert prepared[0].sample.prompt == "P1 which leads to"
    assert prepared[0].target_token == " S"
    assert len(generator.requests) == 1


def test_modal_subject_instantiates_parameterized_backend(monkeypatch) -> None:
    import sys
    from types import SimpleNamespace

    captured = {}

    class RemoteClass:
        def __call__(self, **kwargs):
            captured.update(kwargs)
            return object()

    class Cls:
        @staticmethod
        def from_name(app_name, class_name):
            assert (app_name, class_name) == ("circuit-supernodes", "Attributor")
            return RemoteClass()

    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(Cls=Cls))
    config = SubjectConfig(
        model_name="org/model",
        transcoder_set="org/transcoders",
        node_threshold=0.7,
        edge_threshold=0.9,
    )
    subject = ModalSubject("circuit-supernodes", "Attributor", config)
    subject._subject()
    assert captured == config.model_dump()
