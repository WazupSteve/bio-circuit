"""Answer-aware subject-model sampling and dataset triage helpers."""

from __future__ import annotations

from typing import Literal, TypedDict

MAX_NEW_TOKENS = 50

Disposition = Literal["keep_correct", "keep_incorrect", "reject"]


class SamplingResult(TypedDict, total=False):
    status: Literal["matched", "exhausted", "error"]
    generated_text: str
    matched_answer: str
    provided_prompt: str
    target_token: str
    generated_tokens: int
    error: str


def answer_key(value: str) -> str:
    """Compare labels independently of presentation whitespace and casing."""

    return value.strip().casefold()


def classify_result(target: str, result: SamplingResult | None) -> Disposition:
    """Turn a bounded sampler result into the builder's three-way decision."""

    if not result or result.get("status") != "matched":
        return "reject"
    if answer_key(result.get("matched_answer", "")) == answer_key(target):
        return "keep_correct"
    return "keep_incorrect"


def answer_only_prompt(prompt: str, possible_answers: tuple[str, ...]) -> str:
    """Tell the subject model to emit one vocabulary label without narration."""

    answers = tuple(answer.strip() for answer in possible_answers if answer.strip())
    if not answers:
        raise ValueError("possible_answers must contain at least one non-empty label")
    return (
        "Answer with only the answer label and nothing else. "
        "Do not explain your answer or write any other words.\n\n"
        f"{prompt}"
    )


def sampling_request(prompt: str, possible_answers: tuple[str, ...]) -> dict:
    return {
        "prompt": answer_only_prompt(prompt, possible_answers),
        "possible_answers": list(possible_answers),
        "max_new_tokens": MAX_NEW_TOKENS,
    }
