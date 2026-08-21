"""Shared answer-aware preparation for every dataset builder."""

from typing import Literal

from circuit_supernodes.models import (
    FrozenStrictModel,
    SamplingRequest,
    SamplingResult,
    TracerGraphPayload,
)
from circuit_supernodes.runtime.answers import answer_key
from circuit_supernodes.runtime.graphs import top_outputs
from circuit_supernodes.runtime.modal import ModalSubject
from circuit_supernodes.runtime.sampling import MAX_NEW_TOKENS
from pydantic import Field
from scripts.prompt_generation import CircuitSample


class InitialSampling(FrozenStrictModel):
    initial: SamplingResult
    max_new_tokens: int = Field(ge=1, le=MAX_NEW_TOKENS)


class PreparedSample(FrozenStrictModel):
    sample: CircuitSample
    target_token: str
    disposition: Literal["keep_correct", "keep_incorrect", "reject"]
    sampling: InitialSampling


def answer_only_prompt(prompt: str, possible_answers: tuple[str, ...]) -> str:
    if not any(answer.strip() for answer in possible_answers):
        raise ValueError("possible_answers must contain at least one non-empty label")
    return (
        "Answer with only the answer label and nothing else. "
        "Do not explain your answer or write any other words.\n\n"
        f"{prompt}"
    )


def sampling_request(
    prompt: str, possible_answers: tuple[str, ...]
) -> dict[str, object]:
    request = SamplingRequest(
        prompt=answer_only_prompt(prompt, possible_answers),
        possible_answers=list(possible_answers),
        max_new_tokens=MAX_NEW_TOKENS,
    )
    return request.model_dump(exclude={"debug"})


def classify_result(
    target: str, result: SamplingResult | dict[str, object] | None
) -> Literal["keep_correct", "keep_incorrect", "reject"]:
    if result is None:
        return "reject"
    try:
        parsed = (
            result
            if isinstance(result, SamplingResult)
            else SamplingResult.model_validate(result)
        )
    except Exception:
        return "reject"
    if parsed.status != "matched" or parsed.matched_answer is None:
        return "reject"
    return (
        "keep_correct"
        if answer_key(parsed.matched_answer) == answer_key(target)
        else "keep_incorrect"
    )


def _sample_answers(
    generator: ModalSubject, samples: list[CircuitSample]
) -> list[SamplingResult]:
    requests = [
        sampling_request(sample.prompt, sample.possible_answers) for sample in samples
    ]
    return [
        SamplingResult.model_validate(result)
        for result in generator.sample_answers(requests)
    ]


def _prepared(
    sample: CircuitSample,
    result: SamplingResult,
    disposition: Literal["keep_correct", "keep_incorrect", "reject"],
    sampling: InitialSampling,
) -> PreparedSample:
    return PreparedSample(
        sample=CircuitSample(
            family=sample.family,
            prompt=str(result.provided_prompt),
            target=sample.target,
            possible_answers=sample.possible_answers,
            source_family=sample.source_family,
            goal=sample.goal,
        ),
        target_token=str(result.target_token),
        disposition=disposition,
        sampling=sampling,
    )


def prepare_samples(
    generator: ModalSubject, samples: list[CircuitSample]
) -> tuple[list[PreparedSample], dict[str, int]]:
    """Keep correct answers and record, but reject, all other outcomes."""

    counts = {"keep_correct": 0, "keep_incorrect": 0, "reject": 0}
    prepared: list[PreparedSample] = []
    initial_results = _sample_answers(generator, samples)
    for sample, result in zip(samples, initial_results, strict=True):
        disposition = classify_result(sample.target, result)
        counts[disposition] += 1
        if disposition == "keep_correct":
            prepared.append(
                _prepared(
                    sample,
                    result,
                    disposition,
                    InitialSampling(initial=result, max_new_tokens=MAX_NEW_TOKENS),
                )
            )
    return prepared, counts


def is_faithful(graph: TracerGraphPayload, target: str) -> tuple[bool, str]:
    """Require the sampled answer to equal the attributed graph's top token."""

    outputs = top_outputs(graph)
    if not outputs:
        return False, ""
    top_token = outputs[0][0]
    return bool(top_token) and target == top_token, top_token
