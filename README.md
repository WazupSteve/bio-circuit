# Bio-circuit tasks

This integration exposes three independently analyzable TaskSets from one core
package, plus one graph policy shared by all of them:

| Contract | TaskSet ID |
| --- | --- |
| Five visible examples with 25 positive and 25 neutral holdouts | `circuit-supernodes` |
| Hidden-prompt goal-supernode cohorts with specificity controls | `supernode-rediscovery` |
| Shared corruption with discovery/held-out positives and clean Qwen nulls | `suppressed-features` |
| Maximal stored graphs with policy-selected top-k views | shared by all three |

All generated prompt families use the bounded answer-aware sampler introduced by
the eval-family work. Discovery TaskSets use negative-intervention ablation;
restoration is rewarded by gated clean-distribution KL similarity. See
[the system documentation](docs/README.md) for the core runtime, TaskSets,
graders, data generation, Neuronpedia support, and feature-creep inventory. See
[the environment guide](environments/circuit_supernodes/README.md) for exact
builder and deployment commands.

## Project boundaries

This repository deliberately has two Python projects. The root project is the
Prime runner and repository-wide lint/type-check environment. The nested
`environments/circuit_supernodes` project is the independently buildable
Verifiers TaskSet package and owns its runtime, builders, tests, and Modal
deployments. Each project has its own lock file; run environment commands from
the nested directory unless a command explicitly targets the repository root.

```bash
cd environments/circuit_supernodes
uv sync --group dev
uv run pytest
uv run ruff check .
```
