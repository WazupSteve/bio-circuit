"""Validation and output-based scoring for persistent feature corruption."""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .graphs import Graph
from .interventions import (
    MAX_ABS_WEIGHT,
    MAX_CORRUPTION_FACTOR,
    MAX_CORRUPTION_NODES,
    MAX_INTERVENTIONS,
    MIN_CORRUPTION_FACTOR,
    MIN_CORRUPTION_NODES,
    Intervention,
    coordinate_from_feature_id,
    validate_interventions,
)

REWARD_VERSION = "v3-persistent-corruption-verified-gated-kl"
MODAL_APP = "circuit-supernodes-persistent-restoration"
MODAL_CLS = "RestorationAttributor"
RETRY_DELAYS = (1.0, 4.0, 15.0)

Corruption = list[Intervention]


class InvalidRestorationResult(RuntimeError):
    """The deployed backend did not uphold the scoring protocol."""


def parse_intervention_graph(text: str) -> dict[str, Any] | None:
    """Parse the exact single fenced JSON block required by the task prompt."""

    match = re.fullmatch(r"\s*```(?:json)?\s*(\{.*\})\s*```\s*", text, re.DOTALL)
    if match is None:
        return None
    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    valid = isinstance(data, dict) and isinstance(data.get("nodes"), list)
    return data if valid else None


def validate_corruption(raw: Sequence[Sequence[object]]) -> Corruption:
    """Validate the compact, hidden `(layer, feature, position, factor)` payload."""

    return validate_interventions(
        raw,
        label="corruption",
        require_nonempty=True,
        min_count=MIN_CORRUPTION_NODES,
        max_count=MAX_CORRUPTION_NODES,
        min_factor=MIN_CORRUPTION_FACTOR,
        max_abs_factor=MAX_CORRUPTION_FACTOR,
        require_positive=True,
    )


def validate_intervention_graph(
    data: Mapping[str, Any], graph: Graph
) -> tuple[list[tuple[str, float]] | None, str]:
    """Return bounded unique FEATURE ids from the complete exported graph."""

    raw_nodes = data.get("nodes")
    if not isinstance(raw_nodes, list):
        return None, "nodes must be a list"
    if not 1 <= len(raw_nodes) <= MAX_INTERVENTIONS:
        return None, f"use 1-{MAX_INTERVENTIONS} intervention nodes"

    interventions: list[tuple[str, float]] = []
    seen: set[str] = set()
    for index, raw_node in enumerate(raw_nodes):
        if not isinstance(raw_node, Mapping):
            return None, f"node {index} must be an object"
        node_id = str(raw_node.get("id", "")).strip()
        if not node_id:
            return None, f"node {index} needs an id"
        node = graph.nodes.get(node_id)
        if node is None or node.kind != "feature":
            return None, f"{node_id!r} is not an exported FEATURE id"
        try:
            coordinate_from_feature_id(node_id)
        except ValueError:
            return None, f"{node_id!r} is not a valid FEATURE coordinate"
        if node_id in seen:
            return None, f"duplicate intervention node {node_id!r}"
        raw_weight = raw_node.get("intervention_weight")
        if isinstance(raw_weight, bool):
            return None, f"{node_id!r} needs a numeric intervention_weight"
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError, OverflowError):
            return None, f"{node_id!r} needs a numeric intervention_weight"
        if not math.isfinite(weight):
            return None, f"{node_id!r} intervention_weight must be finite"
        if abs(weight) > MAX_ABS_WEIGHT:
            return None, (
                f"{node_id!r} intervention_weight must lie in "
                f"[-{MAX_ABS_WEIGHT:g}, {MAX_ABS_WEIGHT:g}]"
            )
        seen.add(node_id)
        interventions.append((node_id, weight))
    return interventions, ""


def validate_replay_invariants(
    result: Mapping[str, Any], expected_target: str | None = None
) -> None:
    """Fail closed when the deployed service and dataset no longer agree."""

    if expected_target is not None and result.get("target_token") != expected_target:
        raise InvalidRestorationResult("restoration backend returned the wrong target")
    if result.get("standard_target_is_top") is not True:
        raise InvalidRestorationResult("clean replay does not have the target at top-1")
    if result.get("corruption_changed_top_token") is not True:
        raise InvalidRestorationResult(
            "hidden corruption no longer changes the top token"
        )


def reward_from_restoration(
    result: Mapping[str, Any], expected_target: str | None = None
) -> tuple[float, str]:
    """Apply the exact-token gate, then clean-distribution similarity.

    `exp(-KL(clean || restored))` is one only for the clean distribution and is
    always in `(0, 1]`. Protocol mismatches fail as infrastructure errors; they
    must never be converted into rewards.
    """

    validate_replay_invariants(result, expected_target)
    if result.get("restored_target_is_top") is not True:
        return 0.0, "target_token_gate"
    try:
        divergence = float(result["restored_standard_kl"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise InvalidRestorationResult(
            "restoration backend returned no numeric clean KL"
        ) from exc
    if not math.isfinite(divergence) or divergence < -1e-6:
        raise InvalidRestorationResult(
            "restoration backend returned an invalid clean KL"
        )
    return math.exp(-max(0.0, divergence)), "exact_gated_clean_similarity"


async def restore_with_retries(attributor, *args) -> dict:
    for delay in RETRY_DELAYS:
        try:
            return await attributor.restore_output.remote.aio(*args)
        except Exception:
            await asyncio.sleep(delay)
    return await attributor.restore_output.remote.aio(*args)


class RestorationScoring:
    """Replay hidden corruption, compose a proposal, and score clean recovery."""

    def __init__(self) -> None:
        self._attributor = None

    async def score(
        self,
        prompt: str,
        graph: Graph,
        corruption: Sequence[Sequence[object]],
        target_token: str,
        data: Mapping[str, Any],
    ) -> dict[str, Any]:
        interventions, reason = validate_intervention_graph(data, graph)
        if interventions is None:
            return {
                "reward": 0.0,
                "gated": reason,
                "parse": True,
                "reward_version": REWARD_VERSION,
            }
        hidden = validate_corruption(corruption)
        if not target_token:
            return {
                "reward": 0.0,
                "gated": "task has no original target token",
                "parse": True,
                "reward_version": REWARD_VERSION,
            }

        if self._attributor is None:
            import modal

            self._attributor = modal.Cls.from_name(MODAL_APP, MODAL_CLS)()
        result = await restore_with_retries(
            self._attributor,
            prompt,
            hidden,
            [
                (*coordinate_from_feature_id(node_id), weight)
                for node_id, weight in interventions
            ],
            target_token,
        )
        reward, mode = reward_from_restoration(result, target_token)
        return {
            **result,
            "reward": reward,
            "score_mode": mode,
            "intervention_count": len(interventions),
            "parse": True,
            "reward_version": REWARD_VERSION,
        }
