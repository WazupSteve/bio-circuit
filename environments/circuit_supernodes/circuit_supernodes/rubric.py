"""Proposal validation and causal minimum-description-length reward."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Mapping

from .graphs import Graph, feature_coords
from .mdl import (
    MAX_GROUPS,
    OUTPUT,
    ResponseBank,
    score_circuit,
)

REWARD_VERSION = "v4.0-exponential-mdl"
"""The v3 score is normalized held-out causal MDL saving.

Numbers are intentionally incomparable with v1/v2, which multiplied separation,
coverage, and completeness.  V3 measures distortion on one fixed hidden feature
universe and charges a canonical literal circuit code in the same unit: bits.
"""

MODAL_APP = "circuit-supernodes-attribution"
MODAL_CLS = "Attributor"
MIN_GROUP_SIZE = 2
"""Mechanical floor for leave-one-member-out block estimation."""

CACHE_MAX = 512
RETRY_DELAYS = (1.0, 4.0, 15.0)
TOOL_ABLATION = "zero"


def relative_drop(base: float, changed: float) -> float | None:
    return max(0.0, (base - changed) / base) if base > 1e-6 else None


def parse_proposal(text: str) -> dict | None:
    """Read the last well-formed proposal; agents often show drafts first."""

    blobs = [
        match.group(1)
        for match in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    ]
    if not blobs:
        start, end = text.find("{"), text.rfind("}")
        blobs = [text[start : end + 1]] if start != -1 and end > start else []
    for blob in reversed(blobs):
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and isinstance(data.get("supernodes"), list):
            return data
    return None


def validate_members(
    data: dict, graph: Graph
) -> tuple[dict[str, list[str]] | None, str]:
    """Reject malformed membership without imposing a reward-like heuristic gate."""

    supernodes: dict[str, list[str]] = {}
    names_seen: set[str] = set()
    members_used: set[str] = set()
    for raw_supernode in data["supernodes"]:
        if not isinstance(raw_supernode, dict):
            return None, "every supernode must be an object"
        name = str(raw_supernode.get("name", "")).strip()
        if not name:
            return None, "supernode names must be non-empty"
        if name.lower() == OUTPUT:
            return None, f"{OUTPUT!r} is the reserved terminal node"
        if name.lower() in names_seen:
            return None, f"duplicate supernode name {name!r}"
        names_seen.add(name.lower())

        raw_members = raw_supernode.get("nodes", [])
        if not isinstance(raw_members, list):
            return None, f"{name!r} nodes must be a list"
        members = [str(member) for member in raw_members]
        unknown = sorted(
            {
                member
                for member in members
                if graph.nodes.get(member) is None
                or graph.nodes[member].kind != "feature"
            }
        )
        if unknown:
            return None, f"{name!r} lists non-FEATURE ids: {unknown[:4]}"
        if len(set(members)) != len(members):
            return None, f"{name!r} repeats a member"
        if overlap := sorted(set(members) & members_used):
            return None, f"{name!r} overlaps another supernode: {overlap[:4]}"
        if len(members) < MIN_GROUP_SIZE:
            return None, (
                f"{name!r} has {len(members)} members; "
                f"leave-one-member-out scoring needs at least {MIN_GROUP_SIZE}"
            )
        members_used.update(members)
        supernodes[name] = members

    if not 1 <= len(supernodes) <= MAX_GROUPS:
        return None, f"use 1-{MAX_GROUPS} supernodes, not {len(supernodes)}"
    return supernodes, ""


def validate_topology(
    names: list[str], edges: object
) -> tuple[dict[str, set[str]] | None, str]:
    """Return transitive reachability for a strict, duplicate-free DAG."""

    if edges is None:
        edges = []
    if not isinstance(edges, list):
        return None, "edges must be a list"
    lookup = {name.lower(): name for name in names} | {OUTPUT: OUTPUT}
    adjacency: dict[str, set[str]] = {name: set() for name in names}
    seen_edges: set[tuple[str, str]] = set()
    for edge in edges:
        if not (isinstance(edge, (list, tuple)) and len(edge) == 2):
            return None, f"every edge must be a [source, target] pair, got {edge!r}"
        source, target = (lookup.get(str(value).strip().lower()) for value in edge)
        if source is None or target is None:
            return None, f"edge {list(edge)!r} names an unknown supernode"
        if source == OUTPUT:
            return None, f"{OUTPUT!r} is terminal and cannot be an edge source"
        if source == target:
            return None, f"{source!r} has a self-loop"
        canonical = (source, target)
        if canonical in seen_edges:
            return None, f"duplicate edge {list(canonical)!r}"
        seen_edges.add(canonical)
        adjacency[source].add(target)

    closure: dict[str, set[str]] = {}
    for start in names:
        reached: set[str] = set()
        stack = list(adjacency[start])
        while stack:
            node = stack.pop()
            if node in reached:
                continue
            reached.add(node)
            stack.extend(adjacency.get(node, ()))
        if start in reached:
            return None, f"{start!r} lies on a cycle; the circuit must be a DAG"
        closure[start] = reached
    return closure, ""


async def intervene_with_retries(attributor, *args) -> dict:
    """Retry tool interventions, then propagate infrastructure failure."""

    for delay in RETRY_DELAYS:
        try:
            return await attributor.intervene.remote.aio(*args)
        except Exception:
            await asyncio.sleep(delay)
    return await attributor.intervene.remote.aio(*args)


async def response_bank_with_retries(attributor, *args) -> dict:
    """Retry fixed-bank construction, then propagate infrastructure failure."""

    for delay in RETRY_DELAYS:
        try:
            return await attributor.causal_response_bank.remote.aio(*args)
        except Exception:
            await asyncio.sleep(delay)
    return await attributor.causal_response_bank.remote.aio(*args)


class InterventionScoring:
    """Score every proposal against the same cached hidden response bank."""

    def __init__(self) -> None:
        self._attributor = None
        self._cache: dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()

    async def _response_bank(
        self, prompt: str, feature_ids: list[str], target_token: str
    ) -> dict:
        key = json.dumps([prompt, target_token, feature_ids])
        async with self._lock:
            if self._attributor is None:
                import modal

                self._attributor = modal.Cls.from_name(MODAL_APP, MODAL_CLS)()
            future = self._cache.get(key)
            if future is None:
                if len(self._cache) >= CACHE_MAX:
                    self._cache.clear()
                future = asyncio.ensure_future(
                    response_bank_with_retries(
                        self._attributor,
                        prompt,
                        feature_coords(feature_ids),
                        target_token,
                    )
                )
                self._cache[key] = future
        try:
            return await asyncio.shield(future)
        except BaseException:
            failed = future.cancelled() or (
                future.done() and future.exception() is not None
            )
            if failed:
                async with self._lock:
                    if self._cache.get(key) is future:
                        del self._cache[key]
            raise

    async def score(
        self,
        prompt: str,
        graph: Graph,
        data: dict,
        nonce: str = "",
        response_bank: Mapping | ResponseBank | None = None,
    ) -> dict:
        """Validate, decode, and score one proposal.

        ``nonce`` remains accepted for task-interface compatibility, but v4 does
        not rotate proposal-dependent splits.  Every sibling proposal must see
        exactly the same hidden bank.
        """

        del nonce

        def gated(
            reason: str, membership: bool = False, topology: bool = False
        ) -> dict:
            return {
                "reward": 0.0,
                "net_reward": 0.0,
                "gated": reason,
                "parse": True,
                "membership": membership,
                "topology": topology,
                "reward_version": REWARD_VERSION,
            }

        supernodes, reason = validate_members(data, graph)
        if supernodes is None:
            return gated(reason)
        names = list(supernodes)
        closure, reason = validate_topology(names, data.get("edges"))
        if closure is None:
            return gated(reason, membership=True)

        target_token = graph.top_outputs[0][0] if graph.top_outputs else ""
        if not target_token.strip():
            return gated("graph has no target token", membership=True, topology=True)

        feature_ids = sorted(node.id for node in graph.feature_nodes)
        if response_bank is None:
            response_bank = await self._response_bank(prompt, feature_ids, target_token)
        bank = (
            response_bank
            if isinstance(response_bank, ResponseBank)
            else ResponseBank.from_dict(response_bank)
        )
        if set(bank.feature_ids) != set(feature_ids):
            raise RuntimeError(
                "response-bank feature universe does not match the displayed graph"
            )

        score = score_circuit(
            bank,
            supernodes,
            closure,
            edge_count=len(data.get("edges") or []),
        )
        return {
            **score.to_dict(),
            "reward_version": REWARD_VERSION,
            "parse": True,
            "membership": True,
            "topology": True,
            "bank_id": bank.metadata.get("bank_id"),
        }
