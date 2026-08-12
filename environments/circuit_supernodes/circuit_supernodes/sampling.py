"""Answer-aware subject-model sampling and dataset triage helpers."""

from __future__ import annotations

import re
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


def first_answer(
    text: str, possible_answers: list[str], *, final: bool = False
) -> tuple[str, int, int] | None:
    """Find the earliest complete answer label in generated text."""

    answers = [answer.strip() for answer in possible_answers if answer.strip()]
    matches: list[tuple[int, int, str]] = []
    for answer in answers:
        match = re.search(
            rf"(?<!\w){re.escape(answer)}(?!\w)", text, flags=re.IGNORECASE
        )
        if match is None:
            continue
        if match.end() == len(text) and not final:
            key = answer.casefold()
            if any(
                other.casefold() != key and other.casefold().startswith(key)
                for other in answers
            ):
                continue
        matches.append((match.start(), -len(answer), answer))
    if not matches:
        return None
    start, negative_length, answer = min(matches)
    return answer, start, start - negative_length


def token_prefix_at_char(tokenizer, token_ids: list[int], char_index: int) -> int:
    for index in range(len(token_ids) + 1):
        decoded = tokenizer.decode(
            token_ids[:index],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if len(decoded) > char_index:
            return max(0, index - 1)
    return len(token_ids)


def sample_until_answer(model, request: dict) -> SamplingResult:
    """Run the shared bounded greedy answer sampler on a subject model."""

    import torch

    prompt = str(request.get("prompt", ""))
    possible_answers = [
        str(answer).strip()
        for answer in request.get("possible_answers", [])
        if str(answer).strip()
    ]
    max_new_tokens = int(request.get("max_new_tokens", MAX_NEW_TOKENS))
    if not prompt or not possible_answers or not 1 <= max_new_tokens <= MAX_NEW_TOKENS:
        return {
            "status": "error",
            "generated_text": "",
            "error": "prompt, possible_answers, and 1-50 new tokens are required",
        }

    tokenizer = model.tokenizer
    tokens = model.ensure_tokenized(prompt)
    generated_ids: list[int] = []
    match = None
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            logits = model(tokens, return_type="logits")
            next_id = int(logits.reshape(-1, logits.shape[-1])[-1].argmax())
            generated_ids.append(next_id)
            next_token = torch.tensor(
                [next_id] if tokens.dim() == 1 else [[next_id]],
                device=tokens.device,
                dtype=tokens.dtype,
            )
            tokens = torch.cat((tokens, next_token), dim=-1)
            generated_text = tokenizer.decode(
                generated_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            match = first_answer(generated_text, possible_answers)
            if match is not None or next_id == tokenizer.eos_token_id:
                break
        else:
            generated_text = tokenizer.decode(
                generated_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

    match = match or first_answer(generated_text, possible_answers, final=True)
    if match is None:
        return {
            "status": "exhausted",
            "generated_text": generated_text,
            "generated_tokens": len(generated_ids),
        }
    answer, start, _ = match
    answer_token_index = token_prefix_at_char(tokenizer, generated_ids, start)
    prefix_text = tokenizer.decode(
        generated_ids[:answer_token_index],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    target_token = tokenizer.decode(
        [generated_ids[answer_token_index]],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return {
        "status": "matched",
        "generated_text": generated_text,
        "matched_answer": answer,
        "provided_prompt": prompt + prefix_text,
        "target_token": target_token,
        "generated_tokens": len(generated_ids),
    }
