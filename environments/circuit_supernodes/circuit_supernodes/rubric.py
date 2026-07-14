import asyncio
import json
import math
import re

from .graphs import Graph, feature_coords

MODAL_APP = "circuit-supernodes-attribution"
MODAL_CLS = "Attributor"
MAX_GROUPS = 8  # bound GPU cost per rollout
MAX_DOMINANCE = 0.7  # max fraction of grouped influence in one supernode
TAU_RANK = 0.05  # soft-comparison width for the rank AUC (relative-drop units)
DELTA_ABS = 0.3  # relative drop treated as a full-size causal effect
TAU_LOGODDS = 1.0  # completeness decay scale (log-odds units)
EXP_SEP, EXP_COV, EXP_COMP = 0.6, 0.2, 0.2
CACHE_MAX = 4096
OUTPUT = "output"
P_CLIP = 1e-6


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def relative_drop(base: float, changed: float) -> float | None:
    return max(0.0, (base - changed) / base) if base > 1e-6 else None


def completion_text(state) -> str:
    messages = state.get("completion") or []
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
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    blob = m.group(1) if m else None
    if blob is None:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        blob = text[start : end + 1]
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("supernodes"), list):
        return None
    return data


def valid_supernodes(data: dict, g: Graph) -> dict[str, list[str]]:
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


class InterventionScoring:
    def __init__(self):
        self._attributor = None
        self._cache: dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()

    def _reachable(self, names: list[str], edges: list) -> dict[str, set[str]]:
        vocab = set(names) | {OUTPUT}
        # Output is terminal; allowing it as a source creates false reachability.
        adj: dict[str, set[str]] = {n: set() for n in names}
        for e in edges or []:
            if isinstance(e, (list, tuple)) and len(e) == 2:
                a, b = str(e[0]).strip(), str(e[1]).strip()
                if b.lower() == OUTPUT:
                    b = OUTPUT
                if a in adj and b in vocab and a != b:
                    adj[a].add(b)
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
            closure[start] = seen
        return closure

    async def _intervene(
        self, prompt: str, groups: list, target_token: str, complement: list
    ) -> dict:
        key = json.dumps([prompt, target_token, groups, complement])
        async with self._lock:
            if self._attributor is None:
                import modal

                self._attributor = modal.Cls.from_name(MODAL_APP, MODAL_CLS)()
            fut = self._cache.get(key)
            if fut is None:
                if len(self._cache) >= CACHE_MAX:
                    self._cache.clear()
                fut = asyncio.ensure_future(
                    self._attributor.intervene.remote.aio(
                        prompt, groups, target_token, complement or None
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

    @staticmethod
    def _separation(result: dict, names: list[str], closure: dict) -> float | None:
        base_prob, base_acts = result["base_prob"], result["base_group_acts"]

        obs: list[tuple[str, bool, float]] = []
        for i, src in enumerate(names):
            abl = result["ablations"][i]
            for j, dst in enumerate(names):
                if i == j:
                    continue
                d = relative_drop(base_acts[j], abl["group_acts"][j])
                if d is not None:
                    obs.append((src, dst in closure[src], d))
            d = relative_drop(base_prob, abl["target_prob"])
            if d is not None:
                obs.append((src, OUTPUT in closure[src], d))

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

    async def score(self, prompt: str, g: Graph, data: dict) -> dict:
        def gated(reason: str) -> dict:
            return {"reward": 0.0, "gated": reason}

        sns = valid_supernodes(data, g)
        names = list(sns)[:MAX_GROUPS]
        if len(names) < 2:
            return gated("needs >= 2 valid supernodes")
        sns = {n: sns[n] for n in names}

        total = sum(n.share for n in g.feature_nodes)
        group_share = {n: sum(g.nodes[m].share for m in sns[n]) for n in names}
        grouped = sum(group_share.values())
        if total <= 0 or grouped <= 0:
            return gated("no influence in proposal")
        if max(group_share.values()) / grouped > MAX_DOMINANCE:
            return gated("catch-all supernode dominates grouped influence")

        target_token = g.top_outputs[0][0] if g.top_outputs else ""
        if not target_token.strip():
            return gated("graph has no target token")

        member_ids = {m for ms in sns.values() for m in ms}
        groups = [feature_coords(sns[n]) for n in names]
        complement = feature_coords(
            [n.id for n in g.feature_nodes if n.id not in member_ids]
        )
        closure = self._reachable(names, data.get("edges"))
        if any(not closure[n] for n in names):
            return gated("every supernode needs an outgoing edge")

        result = await self._intervene(prompt, groups, target_token, complement)
        separation = self._separation(result, names, closure)
        if separation is None:
            return gated("no falsifiable prediction")
        coverage = grouped / total
        completeness = self._completeness(result)
        return {
            "reward": separation**EXP_SEP * coverage**EXP_COV * completeness**EXP_COMP,
            "separation": separation,
            "coverage": coverage,
            "completeness": completeness,
        }
