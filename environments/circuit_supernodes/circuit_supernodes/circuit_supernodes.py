"""Taskset loader for supernode circuit identification."""

import json
from pathlib import Path

import verifiers as vf

from .graphs import graph_to_json, load_graph_json, render_graph, slim_graph
from .rubric import InterventionScoring

DATA_DIR = Path(__file__).parent / "data"

SYSTEM_PROMPT = """\
You are an interpretability researcher analyzing attribution graphs of a
language model (Qwen3-4B) computed with cross-layer transcoders, following
Anthropic's "Circuit Tracing" methodology.

You are given the prompt tokens, the model's top output tokens, the graph's
nodes (transcoder features, embeddings, error and logit nodes) and the
strongest edges (direct linear attributions).

Group the FEATURE nodes into supernodes — clusters of features that represent
the same concept and play the same causal role — and describe the circuit as
directed edges between supernodes.

Your proposal is verified CAUSALLY: each supernode you propose will be ablated
in the actual model (features clamped to 0), and the measured effects must
propagate exactly along your edges — ablating a supernode should strongly
reduce the supernodes and output it can reach through your edges, and leave
unreachable supernodes untouched. You are rewarded for the separation between
the two, so both missing edges and superfluous edges cost you.

Rules:
- Supernode members must be FEATURE node ids only — never embedding, error,
  or logit node ids.
- Each feature node belongs to at most one supernode; use 2-8 supernodes.
- "output" is a reserved terminal node: draw an edge ["<supernode>", "output"]
  from any supernode that directly drives the predicted token.
- Name supernodes by their concept (e.g. "Texas", "say a capital", "ones
  digit is 5").
- Prefer high-influence feature nodes; ignore near-zero-influence ones.

Respond with a short analysis, then a single JSON code block:
```json
{
  "supernodes": [{"name": "<concept>", "nodes": ["<node_id>", ...]}, ...],
  "edges": [["<source supernode>", "<target supernode or output>"], ...],
  "narrative": "<one-paragraph mechanistic story from input tokens to output>"
}
```"""


class CircuitTasksetConfig(vf.TasksetConfig):
    dataset_file: str = "dataset.jsonl"
    max_features: int = 60


class CircuitEnvConfig(vf.EnvConfig):
    taskset: CircuitTasksetConfig | None = None


def load_taskset(config: CircuitTasksetConfig | None = None) -> vf.Taskset:
    config = CircuitTasksetConfig.from_config(config)
    dataset_path = DATA_DIR / config.dataset_file

    def rows():
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"{dataset_path} missing — generate it with "
                "`uv run python scripts/build_dataset.py` (runs attribution on Modal)."
            )
        with dataset_path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                g = slim_graph(
                    load_graph_json(rec["graph"]), max_features=config.max_features
                )
                question = (
                    f"Subject prompt: {rec['prompt']!r}\n"
                    f"Expected continuation: {rec['target']!r}\n\n"
                    f"{render_graph(g)}\n\n"
                    "Identify the supernodes and the circuit."
                )
                yield {
                    "prompt": [{"role": "user", "content": question}],
                    "info": {
                        "family": rec["family"],
                        "subject_prompt": rec["prompt"],
                        "expected_supernodes": rec["expected_supernodes"],
                        "expected_edges": rec["expected_edges"],
                        "graph": graph_to_json(g),
                    },
                    "max_turns": 1,
                }

    return vf.Taskset(
        source=rows,
        system_prompt=SYSTEM_PROMPT,
        rewards=InterventionScoring().signals(),
        config=config,
    )


def load_environment(config: CircuitEnvConfig | None = None) -> vf.Env:
    config = config or CircuitEnvConfig()
    return vf.Env(taskset=load_taskset(config=config.taskset))
