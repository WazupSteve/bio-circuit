"""Backend-neutral model operations shared by Modal service adapters."""


def tokens_and_target(
    model,
    prompt: str,
    target_token: str,
    *,
    require_single: bool,
    require_round_trip: bool,
):
    """Tokenize a prompt and validate the intervention target token."""

    token_ids = model.tokenizer.encode(target_token, add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"target token {target_token!r} encodes to nothing")
    if require_single and len(token_ids) != 1:
        raise ValueError(
            f"target token {target_token!r} must encode to exactly one token; "
            f"got {len(token_ids)}"
        )
    if require_round_trip and model.tokenizer.decode(token_ids) != target_token:
        raise ValueError(f"target token {target_token!r} does not round-trip")
    return model.ensure_tokenized(prompt), token_ids[0]


def normalize_activations(activations, *, validate_shape: bool = False):
    """Normalize Circuit Tracer activations to layers/positions/features."""

    normalized = activations[:, 0] if activations.dim() == 4 else activations
    if validate_shape and normalized.dim() != 3:
        raise ValueError(
            "expected activation shape [layers, positions, features], "
            f"got {tuple(normalized.shape)}"
        )
    return normalized


def target_probability(logits, target_id: int) -> float:
    """Return the final-position probability assigned to one target token."""

    import torch  # ty: ignore[unresolved-import] -- installed in the Modal image

    final_logits = logits.reshape(-1, logits.shape[-1])[-1].float()
    return torch.softmax(final_logits, dim=-1)[target_id].item()
