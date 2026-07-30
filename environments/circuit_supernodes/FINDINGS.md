# What the intervention reward actually measures

An audit of the `circuit-supernodes` reward, run before training anything on it.
The audit scripts and archived run data behind these numbers were removed in a
repo cleanup (2026-07-30); both remain in git history at commit `99e1334`, from
which every command quoted below still reproduces every number.

The short version: **the reward's causal signal is dominated by the layer
ordering of the transformer, which requires no experiments to predict.** Once
that is accounted for, three frontier models add approximately nothing, and
scripted partitions that never look at what a feature means add more.

---

## 1. The reward, as built

A policy sees a pruned attribution graph of Qwen3-4B, may run bounded feature
ablations, and proposes 2–8 supernodes plus directed edges. Scoring runs
independent interventions:

```text
reward = separation^0.6 x coverage^0.2 x completeness^0.2
```

`separation` is the only causal term. It asks whether groups the proposal claims
are downstream of a source drop more, when that source is ablated, than groups it
does not claim.

## 2. Two terms were decorative

Grouping every displayed feature pins `coverage` at 1.0 and leaves the complement
empty, which scores `completeness` 1.0 by construction. The reward collapses to
`separation^0.6`. This is not hypothetical — it is what the published frontier
comparison measured:

```bash
uv run python scripts/reward_components.py archive/eval15/*/traces.jsonl
```

| model | separation | coverage | completeness | reward |
|---|---:|---:|---:|---:|
| Claude Fable 5 | **0.580** | 1.000 | 1.000 | **0.663** |
| GPT-5.6 Sol | **0.662** | 0.659 | 0.289 | 0.481 |
| Claude Opus 4.8 | 0.558 | 0.701 | 0.321 | 0.276 |

The causal term is flat across all three. The entire 2.4x spread comes from the
two degenerate terms, and the model with the *worst* separation has the best
reward. The v1 frontier chart this table replaces measured "did you group
every feature you were shown", not circuit quality.

## 3. Blind partitions score well, and a policy can afford them

`scripts/oracle_audit.py` scores scripted attacks that pick a partition without
understanding it, measure the true effect matrix, and fit the best reachability
claim the measurement supports.

```bash
uv run python scripts/oracle_audit.py --dataset heldout.jsonl --rows 10 --multistart 16
```

These are not oracle fantasies. `MAX_GROUPS x MAX_ABLATED_FEATURES` is 96, so at
60 displayed features a partition into five groups of twelve is fully measurable
in **five of the seven** available experiments, with two to spare. The audit
computes this per attack and reports it.

Reporting the best of N restarts against a single-shot policy compares a search
to one draw, so the audit reports the whole curve:

| k | E[best of k] | |
|---:|---:|---|
| 1 | 0.384 | one rollout |
| 8 | 0.592 | a GRPO group |
| 16 | 0.625 | |

Like for like, a single blind partition scores 0.384 and Claude Fable 5 scores
0.523 on the same graphs under the same reward. A good circuit does beat a
random one, by about 36%, and with a within-graph spread of 0.162 the group mean
separates at two standard errors from about 5.4 rollouts — so G=8 clears it,
without much margin.

## 4. The layer ordering is doing most of the work

Later layers are downstream of earlier ones by construction. A partition that
respects the layer ordering and claims forward edges is therefore *causally
correct for free*.

The control holds features, group count and sizes fixed, and differs only in
whether a group sits inside a layer band or spans the whole range:

```bash
uv run python scripts/oracle_audit.py --rows 20 --max-features 150
```

Over all twenty held-out graphs:

| partition | reward | separation |
|---|---:|---:|
| layer-bands | **0.695** | **0.551** |
| layer-spanning | 0.217 | 0.127 |

**4.3x on the causal term from layer structure alone**, and layer-bands beats
every model measured. A layer-band partition needs no experiments at all; it is
readable straight off the graph. The effect was 3.8x on the first eight graphs,
so it strengthens rather than regresses with sample size.

## 5. Correcting for it leaves the models with nothing

Scoring a second time against the closure that "later layers are downstream"
already predicts, and reporting the increment (`separation_over_null`):

Attacks over twenty graphs, models over their fifteen proposals:

| | separation | layer null | over null |
|---|---:|---:|---:|
| coherence-matched | 0.206 | 0.017 | **0.189** |
| community partition | 0.563 | 0.456 | 0.108 |
| layer-spanning | 0.127 | 0.023 | 0.104 |
| budget-feasible | 0.116 | 0.016 | 0.100 |
| random, best of 2 | 0.255 | 0.155 | 0.099 |
| blind-balanced | 0.162 | 0.084 | 0.078 |
| layer-bands | 0.551 | 0.521 | **0.030** |
| Claude Opus 5 (xhigh) | 0.515 | 0.467 | **0.048** |
| GPT-5.6 Sol (xhigh) | 0.509 | 0.476 | **0.033** |
| Kimi K3 (xhigh) | 0.431 | 0.431 | **0.000** |
| Claude Fable 5 | 0.389 | 0.359 | **0.030** |
| GPT-5.6 Sol | 0.459 | 0.455 | **0.004** |
| Claude Opus 4.8 | 0.456 | 0.475 | **−0.019** |

Layer-bands goes from first to last, because a layer-banded partition *is* the
null. The frontier models go with it — Claude Fable 5 lands on exactly the same
0.030, which is to say that in these terms it is indistinguishable from a
partition built by sorting on layer number. Every blind attack adds more over
the null than any model does.

The layer-bands figure was 0.031 on six graphs and 0.030 on twenty.

Claude Opus 5, GPT-5.6 Sol and Kimi K3 were then run at xhigh reasoning on all
twenty graphs, two rollouts each, at 150 displayed features — the strongest
models and the most favourable settings tested, under a prompt that states the
contract. All 120 rollouts completed (six of Kimi K3's needed one retry after
provider rate-limiting); one Kimi K3 proposal was rejected by the dominance
gate, against 9 of 15 rejected for the archived Opus 4.8 under the older
prompt.

Kimi K3 was run under identical settings once it became reachable on the
inference endpoint. None of the three escape the null:

  model                 reward   sep    null   over null
  Claude Opus 5          0.296  0.515   0.467      0.048
  GPT-5.6 Sol            0.303  0.509   0.476      0.033
  Kimi K3                0.356  0.431   0.431      0.000

Kimi K3's separation and layer-null figure are equal to three decimal places:
every unit of causal credit its proposals earned is accounted for by the layer
ordering alone, with nothing left over. Opus 5 and Sol are not far from that —
91% and 93% of their separation is the same free structure. All three sit at or
below a pure layer-band partition (0.030 over the null) and below every blind
attack that is not one (0.078 to 0.189). Their rewards are indistinguishable
from the 0.290 a single blind partition scores on the same graphs.

A stronger model, more reasoning, a wider slice and a corrected prompt therefore
do not change the conclusion. They raise the separation term and leave the part
of it that is not layer ordering where it was.

The reason is visible in what the models proposed. Claude Fable 5's supernodes
include *"Mid-layer capital retrieval at ' is' (L24-30)"* and *"Late-layer
Atlanta output promotion (L31-35)"*. They are layer-banding, in as many words.

```bash
uv run python scripts/score_archived.py --runs archive/eval15/* --max-features 150
```

## 6. Why the correction is a diagnostic and not the reward

Making `separation_over_null` the objective would be worse. Blind partitions beat
frontier models on it *because* a layer-crossing partition with measured edges
leaves the null far behind — so it rewards measurement rather than
understanding.

That is the shape of the underlying problem. The reward asks a policy to predict
intervention effects, and the policy can measure those effects within its budget.
Two mitigations were tried and measured:

- **Hidden split-half scoring.** Each supernode is split under a seed the agent
  cannot know and both halves ablated, so a group earns credit only when its
  causal role is shared across members. This binds against blind partitions
  (−36% for a fixed partition) but does not stop partition *search*, and an
  attacker that matches members by measured per-feature effect neutralises it
  (the split costs it 7%, against 44% for a blind partition).
- **A held-out intervention operator.** The tool zero-clamps; scoring
  mean-ablates, so a circuit fitted to what the agent measured has to generalise.
  The operators genuinely differ, including in *ordering* — but it costs a real
  proposal about as much as a blind one (7% against 5%), so it does not
  discriminate on real graphs.

Neither the raw term nor any correction tried here isolates circuit discovery.
This is a property of the construct, not a bug with a patch.

## 7. What the environment does measure, stated honestly

- Causal consistency of a **downstream-reachability hypothesis** — edge sets with
  the same transitive closure are indistinguishable to it — within a
  frozen-attention replacement model, on a pruned slice.
- The slice is small relative to the circuit. The top 60 features carry about
  **31%** of total feature influence; reaching half would take ~196 features and
  80% ~722. The complement ablation touches only ungrouped *displayed* features,
  so the rest is never intervened on. `coverage = 1.0` means "grouped everything
  shown".
- Most of the causal signal it does capture is the transformer's forward flow.

## 8. Corrections made during the audit

Recorded because two of them nearly became changes to the environment.

- **`share` looked wrong and is right.** The top 60 features by `share` and by
  raw `influence` share *no* nodes, which reads as a serious bug.
  circuit-tracer's `influence` is cumulative — every graph's maximum is exactly
  the exporter's `node_threshold` — so the smallest value marks the most
  important node. Top nodes by `share` sit at mean layer 28 with 90% at the final
  token position; by raw influence, mean layer 10 and 28%. "Fixing" it would have
  shown the agent the sixty *least* important features.
- **A first discrimination result was an artefact.** A synthetic harness appeared
  to show blind shuffles beating the ground-truth partition. Its simulator
  propagated effects one hop while the reward scores transitive reachability, so
  the true claim looked wrong in the world the harness built. With a transitive
  subject the true partition scores 1.000 and the argmax edge set *is* the truth
  (`scripts/partition_discrimination.py`).
- **The topology gate forced false claims.** Requiring every supernode to reach
  `output` made the true answer unstateable when a group is genuinely off-path.
  Fixed; worth 14% to a correct proposal.
- **A minimum group size of 6** was justified by a halving-count argument that
  measurement refuted, and is now the mechanical floor of 2.

## 9. Reproducing all of it

The audit scripts, archived data and test suite live in git history, not this
checkout:

```bash
git checkout 99e1334 -- environments/circuit_supernodes/scripts \
    environments/circuit_supernodes/archive environments/circuit_supernodes/tests
cd environments/circuit_supernodes && uv sync && uv run --with pytest pytest
uv run python scripts/partition_discrimination.py          # no Modal, seconds
uv run python scripts/reward_components.py archive/eval15/*/traces.jsonl
uv run python scripts/oracle_audit.py --rows 10 --multistart 16   # needs Modal
uv run python scripts/score_archived.py --runs archive/eval15/*   # needs Modal
```
