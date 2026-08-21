# Repository map and feature-creep inventory

The requested product core is the tool environment, its three TaskSets, the
graders, data generation, and optional Neuronpedia labels. The repository also
contains substantial supporting and adjacent machinery accumulated during
development. It is listed here explicitly rather than folded invisibly into the
core design.

## Primary layout

```text
configs/
  eval/                         Prime evaluation configurations
  rl/                           Prime training configurations
environments/circuit_supernodes/
  circuit_supernodes/
    data/                       generated/checked-in dataset artifacts
    grading/                    discovery ablation and restoration KL reward
    runtime/                    schemas, graphs, sampling, tools, Modal clients
    tasksets/                   three Verifiers adapters
  modal_app/                    GPU deployment and unified backend
  scripts/                      prompt and dataset generation
  tests/                        unit and contract tests
docs/                           this documentation
```

## Modal deployment layer

One deployed service owns all subject-model execution:

| Service | Model/transcoders | Responsibilities |
| --- | --- | --- |
| `circuit-supernodes/Attributor` | Qwen3-4B / `mwhanna/qwen3-4b-transcoders` | sampling, clean and corrupted attribution, corruption search, clean and corrupted-baseline ablation, and KL restoration replay |

It uses A100-80GB GPUs, shared Hugging Face cache volumes, container scaling,
and long timeouts. Deploying it changes external Modal state and is not part of
local tests.

```bash
cd environments/circuit_supernodes
uv run modal deploy modal_app/attribution.py
```

## Runner and packaging layer

There are two Python projects. The root pins Prime, Modal, and Verifiers for
repository-wide running and checks. The nested project is a buildable
`circuit-supernodes` package with its own dependencies and lockfile. Prime eval
and RL TOML files configure policy models, rollout concurrency, turn/token
budgets, TaskSet thresholds, training filters, and periodic evaluation.

This is operational infrastructure, not part of circuit semantics, but it is
required to reproduce training and evaluation.

## Graph presentation and compatibility machinery

The repository now owns more than a minimal attribution wrapper:

- a maximal-graph storage policy and derived top-k views;
- cumulative-influence-to-delta ranking;
- graph quality accounting and error-node screening;
- textual graph rendering and response-shape normalization;
- model tokenization, activation-shape, and logits adapters;
- custom corrupted-attribution context construction that reaches into Circuit
  Tracer and NNSight internals;
- pinned cloudpickle/Circuit Tracer variants for backend compatibility; and
- retry, streaming gzip, atomic-write, digest, and provenance utilities.

These are valid support systems, but they are evidence of the major feature
creep mentioned in the request: the project has grown into a data/deployment and
compatibility platform, not only an interactive patching environment.

## Evaluation canary

`configs/eval/suppressed-features-canary-egress.toml` is a one-task,
external-policy-model smoke test. It is operational validation and is outside
the scientific task/grader contract.

## Tests

The test suite covers schemas, boundaries, graph views, sampling/data generation,
builder support, visible cohort construction, rediscovery grouping, restoration,
ablation wire contracts, and integration contracts. Tests mock or isolate remote
behavior; they do not deploy the Modal service.

```bash
cd environments/circuit_supernodes
uv sync --group dev
uv run ruff format --check .
uv run ruff check .
uv run pytest
```

## Known architectural seams

- Neuronpedia enrichment is Qwen-only, build-time-only, and best-effort.

These seams should be considered before extending the system again: new work
should either reinforce the shared contract or be documented as another
adjacent subsystem.
