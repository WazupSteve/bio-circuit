# Bio-circuit tasks

This integration implements the four requested changes as three independently
loadable tasksets plus one graph-policy change shared by all of them:

| Contract | Taskset/plugin |
| --- | --- |
| Vanilla causal circuit MDL with all train and sealed eval families | `circuit-supernodes` |
| Hidden-prompt goal-supernode cohorts with specificity controls | `supernode-rediscovery` |
| Shared suppressed-feature corruption with discovery and held-out restoration cohorts | `suppressed-features` |
| Maximal stored graphs with policy-selected top-k views | shared by all three |

All generated prompt families use the bounded answer-aware sampler introduced by
the eval-family work. See
[the environment guide](environments/circuit_supernodes/README.md) for exact
contracts, builders, configs, deployment commands, and validation.

```bash
cd environments/circuit_supernodes
uv sync --group dev
uv run pytest
uv run ruff check .
```
