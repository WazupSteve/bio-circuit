# circuit-supernodes

### Overview
- **Environment ID**: `circuit-supernodes`
- **Short description**: Identify supernodes and circuits in pruned attribution graphs of Qwen3-4B, computed with cross-layer transcoders via circuit-tracer on Modal.
- **Tags**: single-turn, interpretability, train, eval

### Datasets
- **Primary dataset(s)**: Synthetic circuit-eliciting prompts sampled from the prompt families in Anthropic's [attribution-graphs methods paper](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) (two-hop factual recall, addition, antonyms/multilingual, analogies, acronyms, sequences, factual recall, in-context pattern matching). Each prompt's pruned attribution graph is precomputed on a Modal GPU with [circuit-tracer](https://github.com/safety-research/circuit-tracer) and cached into `circuit_supernodes/data/dataset.jsonl`.
- **Generation**: `uv run python scripts/build_dataset.py --n 16 --seed 0` (requires Modal auth)
- **Split sizes**: configurable; default 16 eval examples

### Task
- **Type**: single-turn (v1 `vf.Env` / `vf.Taskset`, base harness)
- **Output format**: short analysis + one JSON code block: `{"supernodes": [{"name", "nodes"}], "edges": [[src, tgt-or-"output"]], "narrative"}`
- **Reward**: a single causal signal, `circuit_faithfulness` (GPU inference at scoring time).
  Each proposed supernode is ablated in the actual model (circuit-tracer
  `feature_intervention` on Modal, attention frozen); we measure relative drops in every
  other supernode's activation and in the target-token probability. The proposal's edges
  predict where drops should occur (transitive reachability, `output` terminal), and

  `reward = clamp(mean(drop | predicted) − mean(drop | unpredicted), 0, 1)`

  Mega-blob and fully-connected proposals score 0 by construction (nothing falsifiable).
  Requires the Modal app deployed (`uv run modal deploy modal_app/attribution.py`) and
  Modal credentials at eval time.

### Quickstart
```bash
# 1. Generate the dataset (runs attribution on Modal, one-time)
cd environments/circuit_supernodes && uv run python scripts/build_dataset.py

# 2. Install + eval
prime env install circuit-supernodes
prime eval run circuit-supernodes
```

### Environment Arguments
Passed as v1 config, e.g. `-a '{"config": {"taskset": {"dataset_file": "dataset.jsonl"}}}'`:

| Taskset config field | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `dataset_file` | str | `"dataset.jsonl"` | Which cached dataset file under `circuit_supernodes/data/` to load |
| `max_features` | int | `60` | Feature nodes kept in the slimmed graph shown to the model |

### Architecture
- `circuit_supernodes/circuit_supernodes.py` — v1 loader: `load_taskset` (rows + rewards) and `load_environment`
- `circuit_supernodes/datagen.py` — synthetic prompt sampler with expected conceptual circuits (weak ground truth)
- `modal_app/attribution.py` — Modal GPU service: Qwen3-4B + transcoders → pruned attribution graph JSON
- `circuit_supernodes/graphs.py` — graph normalization, influence shares, slimming, in-context rendering
- `circuit_supernodes/rubric.py` — `CircuitScoring`: structural reward signals per the paper's supernode criteria
