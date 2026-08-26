"""Bounded greedy sampling until the first complete answer label."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..models import SamplingRequest, SamplingResult
from .answers import first_answer, token_prefix_at_char

MAX_NEW_TOKENS = 50


def sample_until_answer(
    model,
    request: SamplingRequest | dict[str, object],
    *,
    logits_fn: Callable[[Any], Any] | None = None,
) -> dict[str, object]:
    """Run the shared bounded greedy answer sampler on a subject model."""

    import torch  # ty: ignore[unresolved-import] -- installed in the Modal GPU image

    try:
        parsed_request = (
            request
            if isinstance(request, SamplingRequest)
            else SamplingRequest.model_validate(request)
        )
    except Exception:
        return SamplingResult(
            status="error",
            generated_text="",
            generated_tokens=0,
            error="prompt, possible_answers, and 1-50 new tokens are required",
        ).model_dump(exclude_none=True)

    generated_ids: list[int] = []
    generated_text = ""
    match = None
    try:
        tokenizer = model.tokenizer
        tokens = model.ensure_tokenized(parsed_request.prompt)
        with torch.inference_mode():
            for _ in range(parsed_request.max_new_tokens):
                logits = (
                    logits_fn(tokens)
                    if logits_fn is not None
                    else model(tokens, return_type="logits")
                )
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
                match = first_answer(generated_text, parsed_request.possible_answers)
                if match is not None or next_id == tokenizer.eos_token_id:
                    break

        match = match or first_answer(
            generated_text, parsed_request.possible_answers, final=True
        )
        if match is None:
            return SamplingResult(
                status="exhausted",
                generated_text=generated_text,
                generated_tokens=len(generated_ids),
            ).model_dump(exclude_none=True)
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
        return SamplingResult(
            status="matched",
            generated_text=generated_text,
            matched_answer=answer,
            provided_prompt=parsed_request.prompt + prefix_text,
            target_token=target_token,
            generated_tokens=len(generated_ids),
        ).model_dump(exclude_none=True)
    except Exception as exc:
        error = str(exc)
        if parsed_request.debug:
            import traceback

            error = traceback.format_exc()
        return SamplingResult(
            status="error",
            generated_text=generated_text,
            generated_tokens=len(generated_ids),
            error=error,
        ).model_dump(exclude_none=True)
