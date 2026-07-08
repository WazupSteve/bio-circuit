"""Causal reward for proposed supernode circuits.

Parses a proposal, ablates each group on Modal and rewards cases where
measured drops follow the proposal's reachable edges.
"""

import asyncio
import json
import re

import verifiers as vf

from .graphs import Graph, load_graph_json

MODAL_APP = "circuit-supernodes-attribution"
MODAL_CLS = "Attributor"
MAX_GROUPS = 8  # bound GPU cost per rollout
OUTPUT = "output"


class InterventionScoring:
    """Expose the ``circuit_faithfulness`` reward signal."""

    def __init__(self):
        self._attributor = None
        self._lock = asyncio.Lock()

    # proposal extraction

    def _completion_text(self, state) -> str:
        """Return the latest assistant message text."""
        messages = state.get("completion") or []
        for m in reversed(list(messages)):
            role = m.get("role") if isinstance(m, dict) else getattr(m, "role", "")
            if role == "assistant":
                content = (
                    m.get("content")
                    if isinstance(m, dict)
                    else getattr(m, "content", "")
                )
                if isinstance(content, str):
                    return content
        return ""

    def _parse(self, state) -> dict | None:
        """Extract fenced or bare answer JSON."""
        text = self._completion_text(state)
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

    def _supernodes(self, data: dict, g: Graph) -> dict[str, list[str]]:
        """Keep valid, disjoint feature-node groups."""
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

    # Topology of circuit

    def _reachable(self, names: list[str], edges: list) -> dict[str, set[str]]:
        """Compute transitive reachability over valid proposed edges."""
        vocab = set(names) | {OUTPUT}
        adj: dict[str, set[str]] = {n: set() for n in vocab}
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

    @staticmethod
    def _parse_node_id(node_id: str) -> tuple[int, int, int]:
        """Convert ``layer_feature_ctx`` ids into integer triples."""
        layer, feat, pos = node_id.split("_")
        return int(layer), int(feat), int(pos)

    # Modal client

    async def _get_attributor(self):
        """Connect to the deployed Modal ``Attributor`` once."""
        async with self._lock:
            if self._attributor is None:
                import modal

                cls = await asyncio.to_thread(modal.Cls.from_name, MODAL_APP, MODAL_CLS)
                self._attributor = cls()
        return self._attributor

    # Scoring info

    @staticmethod
    def _score(result: dict, names: list[str], closure: dict[str, set[str]]) -> float:
        """Rank reachable ablation drops above unreachable drops."""
        base_prob = result["base_prob"]
        base_acts = result["base_group_acts"]

        def rel_drop(base: float, ablated: float) -> float | None:
            if base <= 1e-6:
                return None  # no baseline signal to drop from
            return (base - ablated) / base  # unclamped: increases rank lowest

        predicted, unpredicted = [], []
        for i, src in enumerate(names):
            abl = result["ablations"][i]
            for j, dst in enumerate(names):
                if i == j:
                    continue
                d = rel_drop(base_acts[j], abl["group_acts"][j])
                if d is None:
                    continue
                (predicted if dst in closure[src] else unpredicted).append(d)
            d = rel_drop(base_prob, abl["target_prob"])
            if d is not None:
                (predicted if OUTPUT in closure[src] else unpredicted).append(d)

        if not predicted or not unpredicted:
            return 0.0  # circuit makes no falsifiable distinction
        wins = 0.0
        for p in predicted:
            for u in unpredicted:
                if abs(p - u) < 1e-4:
                    wins += 0.5
                elif p > u:
                    wins += 1.0
        auc = wins / (len(predicted) * len(unpredicted))
        return max(0.0, min(1.0, 2.0 * auc - 1.0))

    # Reward

    @vf.reward(weight=1.0)
    async def circuit_faithfulness(self, task, state) -> float:
        """Parse, intervene, and score one rollout."""
        g = load_graph_json(task["info"]["graph"])
        data = self._parse(state)
        if not data:
            return 0.0
        sns = self._supernodes(data, g)
        if len(sns) < 2:
            return 0.0
        names = list(sns)[:MAX_GROUPS]
        sns = {n: sns[n] for n in names}
        closure = self._reachable(names, data.get("edges"))

        prompt = task["info"]["subject_prompt"]
        target_token = g.top_outputs[0][0] if g.top_outputs else ""
        if not target_token.strip():
            return 0.0
        groups = [[self._parse_node_id(m) for m in sns[n]] for n in names]

        attributor = await self._get_attributor()
        try:
            result = await attributor.intervene.remote.aio(prompt, groups, target_token)
        except Exception as e:
            state["intervention_error"] = f"{type(e).__name__}: {e}"
            return 0.0
        state["intervention"] = result
        return self._score(result, names, closure)

    def signals(self) -> list:
        """Return verifier reward callables."""
        return [self.circuit_faithfulness]
