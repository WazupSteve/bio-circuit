"""Pure answer-label text matching and token-boundary helpers."""

from __future__ import annotations

import re


def answer_key(value: str) -> str:
    return value.strip().casefold()


def first_answer(
    text: str, possible_answers: list[str], *, final: bool = False
) -> tuple[str, int, int] | None:
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
