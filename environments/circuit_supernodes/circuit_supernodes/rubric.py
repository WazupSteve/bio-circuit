import asyncio
import hashlib
import hmac
import json
import math
import os
import re

from .graphs import Graph, feature_coords

MODAL_APP = "circuit-supernodes-attribution"
MODAL_CLS = "Attributor"
MAX_GROUPS = 8  # bound GPU cost per rollout
MIN_GROUP_SIZE = 6  # see split_members
MAX_DOMINANCE = 0.7  # max fraction of grouped influence in one supernode
TAU_RANK = 0.05  # soft-comparison width for the rank AUC (relative-drop units)
DELTA_ABS = 0.3  # relative drop treated as a full-size causal effect
TAU_LOGODDS = 1.0  # completeness decay scale (log-odds units)
EXP_SEP, EXP_COV, EXP_COMP = 0.6, 0.2, 0.2
CACHE_MAX = 4096
OUTPUT = "output"
P_CLIP = 1e-6
RETRY_DELAYS = (1.0, 4.0, 15.0)  # infra backoff; exhausting these raises
SALT_ENV = "CIRCUIT_SPLIT_SALT"
DEFAULT_SALT = "circuit-supernodes-v2"
FLOOR = (("parse", 0.01), ("membership", 0.02), ("topology", 0.02))


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def relative_drop(base: float, changed: float) -> float | None:
    return max(0.0, (base - changed) / base) if base > 1e-6 else None


def completion_text(state) -> str:
    # Historical traces stored an OpenAI-style ``completion`` list. Verifiers
    # v1 traces store messages in graph ``nodes`` and identify model
    # responses with ``sampled=true``. Accept both representations.
    messages = state.get("completion") or []
    if not messages:
        messages = [
            node.get("message", {})
            for node in state.get("nodes", [])
            if isinstance(node, dict) and node.get("sampled")
        ]
    for m in reversed(messages):
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", "")
        if role == "assistant":
            content = (
                m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
            )
            if isinstance(content, str):
                return content
    return ""


def parse_proposal(text: str) -> dict | None:
    """Read the *last* well-formed proposal: agents routinely show drafts first."""
    blobs = [
        m.group(1)
        for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
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


def valid_supernodes(data: dict, g: Graph) -> dict[str, list[str]]:
    """Lenient membership for *display* (Neuronpedia export): keep whatever renders.

    Scoring uses ``validate_members`` instead, which rejects rather than repairs.
    """
    out: dict[str, list[str]] = {}
    used: set[str] = set()
    for sn in data.get("supernodes", []):
        if not isinstance(sn, dict):
            continue
        name = str(sn.get("name", "")).strip()
        if name.lower() == OUTPUT:
            continue  # reserved terminal node
        members = [
            str(x)
            for x in sn.get("nodes", [])
            if str(x) in g.nodes
            and g.nodes[str(x)].kind == "feature"
            and str(x) not in used
        ]
        if name and members:
            out[name] = members
            used.update(members)
    return out


def validate_members(
    data: dict, g: Graph, min_group_size: int = MIN_GROUP_SIZE
) -> tuple[dict[str, list[str]] | None, str]:
    """Strict membership gate. Silently repairing a malformed proposal teaches the
    policy that malformed proposals are fine, so every violation is a rejection.

    ``min_group_size`` is a parameter so the oracle audit can measure how far the
    gate has to sit before small groups stop defeating the hidden split.
    """
    sns: dict[str, list[str]] = {}
    seen: set[str] = set()
    used: set[str] = set()
    for sn in data["supernodes"]:
        if not isinstance(sn, dict):
            return None, "every supernode must be an object"
        name = str(sn.get("name", "")).strip()
        if not name:
            return None, "supernode names must be non-empty"
        if name.lower() == OUTPUT:
            return None, f"{OUTPUT!r} is the reserved terminal node"
        if name.lower() in seen:
            return None, f"duplicate supernode name {name!r}"
        seen.add(name.lower())
        members = [str(x) for x in sn.get("nodes", [])]
        unknown = sorted(
            {
                m
                for m in members
                if g.nodes.get(m) is None or g.nodes[m].kind != "feature"
            }
        )
        if unknown:
            return None, f"{name!r} lists non-FEATURE ids: {unknown[:4]}"
        if len(set(members)) != len(members):
            return None, f"{name!r} repeats a member"
        if clash := sorted(set(members) & used):
            return None, f"{name!r} overlaps another supernode: {clash[:4]}"
        if len(members) < min_group_size:
            return None, (
                f"{name!r} has {len(members)} members; "
                f"supernodes need at least {min_group_size}"
            )
        used.update(members)
        sns[name] = members

    if not 2 <= len(sns) <= MAX_GROUPS:
        return None, f"use 2-{MAX_GROUPS} supernodes, not {len(sns)}"
    total = sum(n.share for n in g.feature_nodes)
    shares = [sum(g.nodes[m].share for m in ms) for ms in sns.values()]
    if total <= 0 or sum(shares) <= 0:
        return None, "no influence in proposal"
    if max(shares) / sum(shares) > MAX_DOMINANCE:
        return None, "catch-all supernode dominates grouped influence"
    return sns, ""


def validate_topology(
    names: list[str], edges
) -> tuple[dict[str, set[str]] | None, str]:
    """Strict topology gate returning each supernode's downstream closure.

    The reward scores *reachability*, so this is what a proposal actually claims:
    a DAG in which every supernode reaches ``output``.
    """
    lookup = {n.lower(): n for n in names} | {OUTPUT: OUTPUT}
    adj: dict[str, set[str]] = {n: set() for n in names}
    for e in edges or []:
        if not (isinstance(e, (list, tuple)) and len(e) == 2):
            return None, f"every edge must be a [source, target] pair, got {e!r}"
        source, target = (lookup.get(str(x).strip().lower()) for x in e)
        if source is None or target is None:
            return None, f"edge {list(e)!r} names an unknown supernode"
        if source == OUTPUT:
            return None, f"{OUTPUT!r} is terminal and cannot be an edge source"
        if source == target:
            return None, f"{source!r} has a self-loop"
        adj[source].add(target)

    closure: dict[str, set[str]] = {}
    for start in names:
        seen: set[str] = set()
        stack = list(adj[start])
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            stack.extend(adj.get(n, ()))
        if start in seen:
            return None, f"{start!r} lies on a cycle; the circuit must be a DAG"
        if OUTPUT not in seen:
            return None, f"{start!r} has no path to {OUTPUT!r}"
        closure[start] = seen
    return closure, ""


def split_seed(graph_id: str, nonce: str, salt: str | None = None) -> bytes:
    """Committed-but-unrevealed seed for the hidden split.

    Keyed on the training step (via ``nonce``), so it is identical for every
    rollout of a GRPO group by construction and rotates between steps — a fixed
    per-task split would just become another thing to memorize. HMAC rather than
    ``hash()``: string hashing is randomized per process, so sibling rollouts in
    different workers would silently disagree.
    """
    key = (salt or os.environ.get(SALT_ENV, DEFAULT_SALT)).encode()
    return hmac.new(key, f"{graph_id}\x00{nonce}".encode(), hashlib.sha256).digest()


def split_members(members: list[str], seed: bytes) -> tuple[list[str], list[str]]:
    """Halve a supernode under the hidden seed.

    A group admits ``C(m, m//2)/2`` distinct halvings — 1 at m=2, 3 at m=4, 10 at
    m=6, 462 at m=12. Below ``MIN_GROUP_SIZE`` the split is effectively known in
    advance and hides nothing, which is why membership gates on it.
    """
    order = sorted(
        members, key=lambda m: hmac.new(seed, m.encode(), hashlib.sha256).digest()
    )
    cut = len(order) // 2
    return order[:cut], order[cut:]


async def intervene_with_retries(attributor, *args) -> dict:
    """Retry transient infra faults, then let the failure raise.

    Turning a Modal outage into a 0.0 would poison every advantage in the GRPO
    group; raising records a trace error the trainer can drop instead.
    """
    for delay in RETRY_DELAYS:
        try:
            return await attributor.intervene.remote.aio(*args)
        except Exception:
            await asyncio.sleep(delay)
    return await attributor.intervene.remote.aio(*args)


def shaped_reward(result: dict) -> float:
    """Training reward: a small structured floor so a policy that cannot yet earn
    causal credit still has gradient toward a valid proposal. Eval stays strict."""
    floor = sum(weight for stage, weight in FLOOR if result.get(stage))
    return floor + (1.0 - sum(weight for _, weight in FLOOR)) * result["reward"]


class InterventionScoring:
    """Scores a proposal by intervening on the subject model.

    ``split_halves`` selects the reward version: v1 ablates each supernode whole
    (which lets an agent pre-measure the entire scored matrix and transcribe it),
    v2 ablates two hidden halves and credits an edge only when both reproduce the
    effect. The oracle audit runs the same attacks through both.
    """

    def __init__(
        self,
        *,
        split_halves: bool = True,
        salt: str | None = None,
        min_group_size: int = MIN_GROUP_SIZE,
    ):
        self.split_halves = split_halves
        self.salt = salt
        self.min_group_size = min_group_size
        self._attributor = None
        self._cache: dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()

    async def _intervene(
        self,
        prompt: str,
        groups: list,
        target_token: str,
        complement: list,
        indices=None,
    ) -> dict:
        key = json.dumps([prompt, target_token, groups, complement, indices])
        async with self._lock:
            if self._attributor is None:
                import modal

                self._attributor = modal.Cls.from_name(MODAL_APP, MODAL_CLS)()
            fut = self._cache.get(key)
            if fut is None:
                if len(self._cache) >= CACHE_MAX:
                    self._cache.clear()
                fut = asyncio.ensure_future(
                    intervene_with_retries(
                        self._attributor,
                        prompt,
                        groups,
                        target_token,
                        complement or None,
                        indices,
                    )
                )
                self._cache[key] = fut
        try:
            # One cancelled rollout must not cancel the shared remote call.
            return await asyncio.shield(fut)
        except BaseException:
            failed = fut.cancelled() or (fut.done() and fut.exception() is not None)
            if failed:
                async with self._lock:
                    if self._cache.get(key) is fut:
                        del self._cache[key]
            raise

    def _plan(self, sns: dict, names: list[str], seed: bytes):
        """Build the groups to send, which of them to ablate, and which ablation
        records combine into each supernode's measured effect."""
        groups = [feature_coords(sns[n]) for n in names]
        if not self.split_halves:
            return groups, None, [[(i, 1.0)] for i in range(len(names))]
        halves, sources = [], []
        for i, name in enumerate(names):
            parts = split_members(sns[name], seed)
            sources.append(
                [(2 * i + h, len(p) / len(sns[name])) for h, p in enumerate(parts)]
            )
            halves.extend(feature_coords(p) for p in parts)
        return (
            groups + halves,
            list(range(len(groups), len(groups) + len(halves))),
            sources,
        )

    @staticmethod
    def _effect(parts: list[tuple[float, float, float]]) -> float | None:
        """Geometric mean of the fraction-normalized drop of each ablated part.

        One whole-group part reproduces the v1 relative drop exactly. Two halves
        turn the score into a claim that the causal role is *distributed* across
        the group: if one half carries everything, the geometric mean collapses.
        """
        product = 1.0
        for base, changed, fraction in parts:
            drop = relative_drop(base, changed)
            if drop is None:
                return None
            product *= min(drop / fraction, 1.0)
        return product ** (1.0 / len(parts))

    def _observations(
        self, result: dict, names: list[str], sources: list, closure: dict
    ) -> list[tuple[str, bool, float]]:
        base_prob, base_acts = result["base_prob"], result["base_group_acts"]
        ablations = result["ablations"]
        obs: list[tuple[str, bool, float]] = []
        for i, src in enumerate(names):
            parts = [(ablations[idx], fraction) for idx, fraction in sources[i]]
            for j, dst in enumerate(names):
                if i == j:
                    continue
                effect = self._effect(
                    [(base_acts[j], a["group_acts"][j], f) for a, f in parts]
                )
                if effect is not None:
                    obs.append((src, dst in closure[src], effect))
            effect = self._effect([(base_prob, a["target_prob"], f) for a, f in parts])
            if effect is not None:
                obs.append((src, OUTPUT in closure[src], effect))
        return obs

    @staticmethod
    def _separation(
        obs: list[tuple[str, bool, float]], names: list[str]
    ) -> float | None:
        predicted = [d for _, z, d in obs if z]
        unpredicted = [d for _, z, d in obs if not z]
        if not predicted or not unpredicted:
            return None

        rank = _mean(
            [
                1.0 / (1.0 + math.exp(-(p - u) / TAU_RANK))
                for p in predicted
                for u in unpredicted
            ]
        )

        def full(d: float) -> float:
            return min(d / DELTA_ABS, 1.0)

        def macro_err(want_predicted: bool) -> float:
            per_src = []
            for src in names:
                errs = [
                    (1.0 - full(d)) ** 2 if z else full(d) ** 2
                    for s, z, d in obs
                    if s == src and z == want_predicted
                ]
                if errs:
                    per_src.append(_mean(errs))
            return _mean(per_src)

        calibration = 1.0 - 0.5 * (macro_err(True) + macro_err(False))

        def skill(x: float) -> float:
            return max(0.0, 2.0 * x - 1.0)

        return math.sqrt(skill(rank) * skill(calibration))

    @staticmethod
    def _completeness(result: dict) -> float:
        if result.get("complement_prob") is None:
            return 1.0

        def logit(p: float) -> float:
            p = min(max(p, P_CLIP), 1.0 - P_CLIP)
            return math.log(p / (1.0 - p))

        shift = abs(logit(result["complement_prob"]) - logit(result["base_prob"]))
        return math.exp(-shift / TAU_LOGODDS)

    async def measure(self, prompt: str, g: Graph, sns: dict) -> dict:
        """Whole-group effect matrix for a partition — what an attacker with
        unlimited measurement learns. Used only by the oracle audit to upper-bound
        measure-then-copy; the reward never calls it.
        """
        names = list(sns)
        groups = [feature_coords(sns[n]) for n in names]
        target_token = g.top_outputs[0][0] if g.top_outputs else ""
        result = await self._intervene(prompt, groups, target_token, [], None)
        base_prob, base_acts = result["base_prob"], result["base_group_acts"]
        return {
            "names": names,
            "target": [
                relative_drop(base_prob, a["target_prob"]) or 0.0
                for a in result["ablations"]
            ],
            "groups": [
                [
                    relative_drop(base_acts[j], a["group_acts"][j]) or 0.0
                    for j in range(len(names))
                ]
                for a in result["ablations"]
            ],
        }

    async def score(self, prompt: str, g: Graph, data: dict, nonce: str = "") -> dict:
        def gated(
            reason: str, membership: bool = False, topology: bool = False
        ) -> dict:
            return {
                "reward": 0.0,
                "gated": reason,
                "parse": True,
                "membership": membership,
                "topology": topology,
            }

        sns, reason = validate_members(data, g, self.min_group_size)
        if sns is None:
            return gated(reason)
        names = list(sns)
        closure, reason = validate_topology(names, data.get("edges"))
        if closure is None:
            return gated(reason, membership=True)

        target_token = g.top_outputs[0][0] if g.top_outputs else ""
        if not target_token.strip():
            return gated("graph has no target token", membership=True, topology=True)

        member_ids = {m for ms in sns.values() for m in ms}
        complement = feature_coords(
            [n.id for n in g.feature_nodes if n.id not in member_ids]
        )
        groups, indices, sources = self._plan(
            sns, names, split_seed(prompt, nonce, self.salt)
        )
        result = await self._intervene(
            prompt, groups, target_token, complement, indices
        )

        separation = self._separation(
            self._observations(result, names, sources, closure), names
        )
        if separation is None:
            return gated("no falsifiable prediction", membership=True, topology=True)
        coverage = sum(g.nodes[m].share for m in member_ids) / sum(
            n.share for n in g.feature_nodes
        )
        completeness = self._completeness(result)
        return {
            "reward": separation**EXP_SEP * coverage**EXP_COV * completeness**EXP_COMP,
            "separation": separation,
            "coverage": coverage,
            "completeness": completeness,
            "parse": True,
            "membership": True,
            "topology": True,
        }
