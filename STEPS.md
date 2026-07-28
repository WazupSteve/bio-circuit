# STEPS — can RL teach a small model to do real interpretability experiments?

*Rewritten 2026-07-25 after a code audit (Claude + GPT-5.6-sol, 3 rounds) and a
literature sweep. Revised 2026-07-28 after a second independent audit (Claude +
GPT-5.6-sol, read-only, against the current tree) that verified every claim below
in code and found four new defects. Supersedes all previous versions.*

---

## 1. The idea

An LLM agent gets an attribution graph of Qwen3-4B (circuit-tracer + per-layer
transcoders), runs up to 7 causal ablation experiments on the real subject model, and
submits a circuit hypothesis: 2–8 feature groups ("supernodes") plus directed causal
edges. The reward is not a judge's opinion — it is computed by **actually running
held-out interventions** and checking that the predicted structure anticipates what
happens. We then RL-train a 4B policy on this reward.

**The one-sentence claim we are trying to earn:**

> RL with an intervention-verified reward taught a 4B model to *use causal
> experiments*: on unseen task families its circuits get better when it is allowed
> more experiments, and that gain disappears when the experiment tool secretly
> returns scrambled results.

**Why this is worth doing (what the literature says is missing):**

- Anthropic grouped features into supernodes **manually** — "no automated method was
  sufficient" (Circuit Tracing, 2025) — and the multi-org Circuits Landscape report
  (Aug 2025) names automating graph analysis and validation as an open problem.
- Existing interpretability agents are **prompted, not trained** (MAIA, HyVE,
  InterpAgent, probe-prompting pipelines); trained investigators exist but for
  behavior elicitation or static self-explanation (Transluce), not circuit
  hypotheses; the one RL-for-circuits result (MechRL, May 2026) trains a small PPO
  policy over GPT-2 attention-head ablations — not a language agent reading graphs
  and proposing grouped, named hypotheses. **To the best of our literature search,
  no one has RL-trained an LLM agent whose reward is a held-out causal-intervention
  score of its circuit hypothesis.** That is our slot. (Phrase it as a search
  result, never as an absolute — we cannot prove a negative over a fast-moving
  field.)
- Anthropic's May 2026 Circuits Update found that feature labels and activation
  similarity do *not* predict causal effects — downstream causal behavior does. A
  reward built purely from interventions (never from labels) is exactly the
  correction the field is converging on.
- FIND showed prompted agents fail ~48% of even simple interpretation tasks; the
  "Pitfalls in Evaluating Interpretability Agents" paper (Mar 2026) shows agents can
  fake circuit "discovery" via memorization. Training against verifiable
  interventions on **sealed task families** is the answer to both.

**What actually works today** (don't rebuild — but read the caveats, they are load-bearing):

- The env runs end-to-end on Verifiers 0.2.1 + Modal; multi-turn tool loop, Modal
  intervention backend, and Neuronpedia export with supernode overlays all work.
- 33 tests pass locally and Ruff is clean — but **the tests are not in the repo**.
  Commit `d7dcb72` ("chore:vf 0.2.1 port") deleted `tests/test_scoring.py` and
  `tests/test_dataset_and_export.py` and added `/tests/` to the environment's
  `.gitignore`. `git ls-files tests/` returns 0, and `[tool.hatch.build] include`
  excludes them too, so a fresh clone — and anything we publish to the Hub — has no
  test suite at all. Fix this before anything else (Stage 0).
- The frontier chart (`media/01_model_comparison.png`: GPT-5.6-sol 0.68 > Opus 4.8
  0.64 > Fable 5 0.54 >> Qwen3.5-9B 0.14, random 0.24, 16 examples × 48 rollouts) is
  **not reproducible from this checkout** — only the PNG survives; the traces are
  gitignored and the rendering script was deleted. The one archived run with
  recoverable traces (`archive/eval15`, 15 tasks × 1 rollout) *reverses* Fable and
  Sol. Treat the chart as an undocumented preliminary until re-derived (see §2,
  finding 2 — there is a much worse problem with it than irreproducibility).

---

## 2. What the audits found (why we can't train on the reward as-is)

Seven findings. All verified in the current tree.

1. **The score is pre-measurable.** With 60 shown features, 5 groups of 12, and 7
   experiments (12 ablated + 48 probe features each), the agent can measure the
   entire effect matrix the scorer will compute, then transcribe edges from it
   (`experiments.py:6-9` vs `rubric.py:151-200`). "Discovery" reduces to
   measure-then-copy. The archived Fable trace does exactly this: 6 supernodes,
   60/60 features grouped, 7 experiments.

2. **Two of three reward terms are degenerate — and they *are* the published
   ranking.** Grouping all 60 features makes coverage = 1.0 (`rubric.py:249`) and
   completeness = 1.0 by construction (empty complement skipped and scored 1.0,
   `rubric.py:204-205`). The reward collapses to `separation^0.6`. Recovering the
   per-component means from `archive/eval15`:

   | model | separation | coverage | completeness | reward |
   |---|---:|---:|---:|---:|
   | Claude Fable 5 | **0.580** | 1.000 | 1.000 | **0.663** |
   | GPT-5.6 Sol | **0.662** | 0.659 | 0.289 | 0.481 |
   | Claude Opus 4.8 | 0.558 | 0.701 | 0.321 | 0.276 |

   The only causal term is flat (0.56–0.66) across all three models. The entire
   spread — a 2.4× reward ratio — is produced by the two degenerate terms, and the
   model with the *worst* separation wins. **The v1 frontier chart measures "did you
   put all 60 features in a group," not circuit quality.** It cannot be kept as a
   "before adversarial audit" capability baseline; it can only be shown as an
   exhibit *in* the audit.

3. **Infra failures score 0.0.** Any Modal exception becomes reward 0
   (`circuit_supernodes.py:66-68`), and under GRPO one false zero corrupts the whole
   group's advantages.

4. **The reward validates reachability, not edges.** `_reachable` (`rubric.py:96-118`)
   scores the transitive closure, so `A→B→C`, `A→C plus A→B, B→C`, and any other
   edge set with the same closure receive identical separation. The system prompt is
   already honest about this ("Edges mean reachability"); our *framing* is not. We
   are eliciting **downstream-reachability hypotheses**, and every claim in the blog
   and the env card must say so — or we add a direct-edge intervention test, which
   we are not doing at this budget.

5. **The label story is self-contradictory in the code, and accidentally fine in the
   data.** `render_graph` prints every node's label into the prompt
   (`graphs.py:162-172`), and the documented build recipe passes
   `--neuronpedia-labels`, so Stage 5's "hidden Neuronpedia labels" analysis would be
   invalid as specified. But the datasets on disk carry **zero** labels (0 of 57,495
   train nodes, 0 of 36,372 heldout nodes; `clerp` is empty for all 1,709
   non-logit nodes in a sampled graph). So the archived frontier runs read purely
   structural graphs — id, kind, layer, position, token, importance, edges — and
   still separated. That is a *better* result than we claimed, and it makes the
   offline label analysis valid provided we keep labels out. Freeze it: **no feature
   labels in the prompt, ever** (§5).

6. **`share` is an unvalidated heuristic that everything depends on.**
   `graphs.py:123-128` sorts *all* nodes by cumulative influence and takes adjacent
   differences. That one line decides which 60 features the agent sees
   (`slim_graph`), what coverage means, and whether the dominance gate fires.
   Circuit-tracer's `influence` is a cumulative pruning score, so the recovery is
   plausible — but unverified against the raw graph tensors. If it is wrong, every
   number in this project is wrong.

7. **Validation silently repairs instead of rejecting.** `valid_supernodes`
   (`rubric.py:68-87`) drops unknown IDs, non-feature nodes and overlapping members
   without comment; duplicate names overwrite; `rubric.py:219` truncates to 8 groups.
   `parse_proposal` (`rubric.py:52`) takes the *first* JSON block, not the last.

Also: the "hack scores 0" calibration only proved a gate fires, never tested a
gate-compliant meaningless partition; the dataset (36 train / 20 heldout rows, pools
of 6–15 items) is memorization bait at RL visit counts; and the eval config leaves
thinking at default while `configs/rl/circuit-supernodes.toml:12` disables it.

---

## 3. The plan — stage-gated, each with a deliverable and a kill switch

Ceilings: **$4,000 Prime / $2,000 Modal** total. Stage-gated, not calendar-gated.

### Stage 0 — Make the repo reproducible (~$0)

Nothing below is meaningful if the environment can't be checked out and verified.

1. Remove `/tests/` from `environments/circuit_supernodes/.gitignore`, restore the
   33 tests deleted in `d7dcb72`, and add `tests/**` to `[tool.hatch.build] include`
   so the published package ships them.
2. Commit STEPS.md and tag the v1 environment (`env-v1`) so the audit below has a
   fixed referent.
3. Archive the artifacts behind `media/01_model_comparison.png` and
   `media/02_pass_at_k.png` — traces, config, and rendering script. **Not possible:**
   only the PNGs were ever tracked, so instead both charts are marked preliminary and
   not reproducible in `media/README.md`, with the §2 component table attached as the
   substantive reason they are retired.
4. Make finding 2 checkable from a clean checkout:
   `scripts/reward_components.py` recovers the per-component means from the tracked
   `archive/eval15` traces, and a test keeps those traces readable.

**Deliverable:** clean clone → `uv sync --group dev && uv run pytest` → tracked suite
passes (38 tests) with no Modal credentials and no generated dataset.

### Stage 1 — Audit v1, then build v2 (~$150 Modal)

*The environment is the research product; everything downstream inherits its
credibility. Note the order: **audit before redesign**. Designing v2 against an
unmeasured v1 means we cannot tell whether v2 helped.*

**1.1 Error-semantics canary (hard blocker, run first).** Cheap, has an external
dependency, and is the only item that can invalidate the architecture. Record the
hosted prime-rl build; inject one scoring failure; confirm the errored group is
dropped rather than zero-rewarded. The claim that this is prime-rl's confirmed
default is **not** demonstrated by this repo and the Prime CLI isn't installed
locally — the canary is what makes it true. Same run: measure end-to-end step
wall-clock with batch 8 × G=8 = 64 concurrent rollouts against
`max_containers=8` (`modal_app/attribution.py:32`), since queueing there sets Stage 4's
calendar, not its cost.

**1.2 The v1 oracle audit.** Script four attacks and run them against v1:
  - (a) blind balanced 12-feature groups + exact measurement + optimal edge fit;
  - (b) graph-community partition + edge fit;
  - (c) 16–32 random partitions, report max;
  - (d) **adaptive small-group attacker** — minimum-size groups, spend the budget
    measuring members, fit edges. This is the attack v2 must survive (see 1.5).

  Publish the table. If the blind oracle ≈ frontier on v1, that is the blog's "we
  red-teamed our own reward" section, and finding 2's component table is its
  centrepiece.

**1.3 Freeze and validate the intervention protocol** (Miller et al.: faithfulness
scores are *defined* by the ablation methodology). Document zero-clamp ablation,
frozen-attention replacement model, token positions, and pruning thresholds as part
of the env spec. Two audits, both **blockers, not spot checks**:
  - `share` vs raw circuit-tracer influence on 10 graphs (finding 6). Verify against
    graph tensors, not the exported JSON. Graphs are stored raw and slimmed at load
    time, so a fix here does *not* invalidate Stage 2's builds — which is exactly why
    it must happen before Stage 2 spends $450.
  - frozen-vs-unfrozen effect ranks on 10 graphs.

**1.4 Strict gates + shaped floor.** Reject, don't silently fix: duplicate names,
overlapping members, unknown IDs, >8 groups; require a DAG that reaches `output`;
case-insensitive name matching; parse the *last* valid JSON block. Training reward
gets a small structured floor — `0.01·parse + 0.02·membership + 0.02·topology +
0.95·causal` — so a weak policy has gradient toward validity; eval reward is strict
(floor removed). Infra failures get neither. Land the floor in the same change as the
gates: reject-instead-of-repair will spike gate failures for a 4B, and the floor is
what keeps that trainable.

**1.5 Reward v2 — hidden split-half scoring.** After the final proposal, a hidden
seed splits each supernode into two halves; both halves are ablated independently; a
predicted downstream relation scores well only if *both halves* of the source
reproduce the effect (fraction-normalized, geometric mean), fed into the existing
rank + calibration machinery. Precedents: split-half reliability; Shi et al.'s
minimality/independence tests; probe-prompting's matched-random controls.

  - **What this actually buys, stated honestly.** It does *not* make measure-then-copy
    impossible: an agent can still spend ~5 experiments measuring full-group effects
    and transcribe relations, and if its groups are internally homogeneous the
    half-effects track the full effects and the copy scores well. What v2 kills is
    copying with *heterogeneous* groups. It upgrades the construct — a supernode now
    has to be a causally coherent set, which is what the word is supposed to mean —
    rather than eliminating the strategy. Write it that way.
  - **Minimum group size ≥ 6 (gate).** The hidden seed only hides anything when the
    group is big enough to split many ways: a 2-member group has exactly **1**
    possible halving, a 4-member group has 3, a 6-member group has 10, a 12-member
    group has 462. With a ≥2 gate the split is fully known in advance for the
    smallest groups and the "hidden" seed hides nothing — which is precisely what
    attack (d) exploits. ≥6 keeps 8 groups feasible (8 × 6 = 48 ≤ 60) and puts the
    guess at ≤1-in-10.
  - **Seed derivation — HMAC, and it needs plumbing.** Never `hash()`: Python
    randomizes string hashing per process, so different verifiers workers would
    silently derive different splits for the same group. Use
    `HMAC-SHA256(secret_salt, graph_id ‖ budget_variant ‖ group_nonce)`, salt
    committed pre-run and revealed after for verifiability. The nonce must be
    **per-GRPO-group and rotating**, not a persistent `(graph_id, budget)` pair — a
    fixed per-task split is a memorization target across training visits. This
    contradicts the previous version's "identical for all rollouts by construction,
    no plumbing" claim: getting a rotating group-shared nonce into the scorer is real
    plumbing, and canary 1.1 must confirm siblings see the same one. Zero-plumbing
    fallback if that proves impossible: derive the seed from
    `HMAC(secret_salt, canonical(proposal) ‖ graph_id)` — deterministic and
    reproducible, no stable per-task target, unguessable without the salt, at the
    cost of extra within-group variance (mitigate with 2 folds).
  - **Re-fit the calibration constants.** `DELTA_ABS = 0.3` and `TAU_RANK = 0.05`
    were tuned for 12-feature ablations; halves are ~6 features and will produce
    systematically smaller drops. Fraction-normalization handles part of it;
    re-fit on dev graphs rather than inheriting.
  - Coverage/completeness terms and the 0.6/0.2/0.2 exponents stay.
  - Implementation reuses `intervene()` as-is: pass fulls + halves, ablate only the
    halves via `ablation_indices`. Cost is **2k + 2** forward passes per proposal
    (base + 2k halves + complement), against **k + 2** today — a ~2× increase in
    scoring cost that flows into every Stage 3–5 estimate.

**1.6 Error semantics.** Retry Modal calls with backoff; when retries exhaust,
**raise** so verifiers records `trace.error` and the hosted trainer drops the group
(per canary 1.1). Agent mistakes stay ordinary low rewards. Tool budget consumed only
on successful calls. Deadlines on scoring.

**1.7 Re-run the oracle suite against v2** and publish v1-vs-v2 side by side.

**Deliverable:** env v2 + v1/v2 oracle-audit table + extended, *tracked* test suite.
**Kill switch:** if v2 does not beat every oracle — especially (d) — stop and publish
the audit as the result. Do not train against a broken reward.

### Stage 2 — Data worth training on (~$450 Modal)

1. Expand the family pools (currently 6–15 items each), then build:
   - **~150 train graphs** across the 6 existing train families,
   - **~30 IID validation graphs** (unseen items, same families) — all tuning and
     checkpoint selection happens here,
   - **~36 sealed OOD graphs** from **three brand-new families** never used
     anywhere: subject–verb agreement with attractor nouns, two-digit numerical
     comparison (balanced yes/no, anti-shortcut strata), day-of-week arithmetic.
     Old `two_hop`/`addition` demote to dev (they steered the frontier chart).
2. **Graph-quality screening, not just answer screening.** Error nodes carry 20–40%
   of influence on typical prompts and only ~25% of prompts yield clean graphs.
   Replace the current prefix match (`build_dataset.py:15-21`) with **exact expected
   first-token ID**, and add circuit-tracer's replacement/completeness graph scores as
   a filter; record thresholds as env parameters.
3. **No feature labels in the prompt.** Drop `--neuronpedia-labels` from the build
   recipe (it is already a no-op in practice — the shipped data has zero labels) and
   keep any labels we fetch in a separate eval-only side-file keyed by feature id.
   This is what makes Stage 5's label-coherence analysis legitimate, and it matches
   what the archived frontier runs actually read.
4. Each train graph appears as **3 task variants with experiment budgets {0, 3, 7}**
   stated in the prompt (≈450 training tasks) — so the headline budget-curve
   evaluation is in-distribution, per-group budget is homogeneous, and the policy
   learns to act with and without experiments.
5. Freeze + hash everything: dataset manifests pin HF revisions (model,
   transcoders), circuit-tracer version, reward version, Modal app hash. Archive the
   JSONLs — they are gitignored today, which is why our own results aren't
   reproducible.
6. **Sealing is a Stage 2 act, not a Stage 5 one.** The sealed set is generated and
   frozen here by automated screening only: no manual inspection, no threshold
   tuning against it, no scoring, no eyeballing a single graph. After the freeze
   hash is recorded it is untouched until Stage 5.

**Deliverable:** three frozen splits + manifest + freeze hashes.

### Stage 3 — Can a 4B learn this? (~$700 Prime, $350 Modal)

1. **Feasibility eval, 8 rollouts/task on train+IID:** Qwen3.5-4B thinking-on vs
   thinking-off (the eval config leaves thinking at default while the RL config
   disables it — decide once, keep train/serve identical, including think-block
   retention across turns). Track per-task: gate-failure reasons, tool-call success,
   nonzero-variance group fraction, reward components.
   **Go/no-go:** ≥25–30% of groups show reward variance; else SFT must create it.
2. **Self-SFT (no frontier distillation — provider ToS):** rejection-sample the 4B
   (or 9B, labeled "open-family distillation") through the harness; keep the best
   valid trace per graph, rescored under 2 split seeds so we don't select lucky
   splits; target 100–200 traces, ≤2 per graph, adaptive spend capped at 1,000
   rollouts. SFT-then-RL is the evidence-backed pattern for 4B agents (WebSailor,
   UserRL, InfoAgent all show cold RL plateaus).
3. **Baseline the frontier under v2** on dev data (~$150 API). This is the *only*
   frontier baseline that goes in the paper — the v1 chart appears solely as an
   exhibit in the reward audit, captioned with the component table from §2.

**Deliverable:** feasibility memo + SFT checkpoint + v2 frontier baseline.
**Kill switch:** if even SFT can't produce ≥80% gate-valid rollouts and reward
variance, publish the negative result (4B trainability limits) — still a real
finding.

### Stage 4 — The RL run (~$2,050 Prime, $500 Modal)

- GRPO via prime-rl on Qwen3.5-4B from the SFT checkpoint. **Group size 8** (not 4),
  ~40 steps × **two seeds** (replication beats one long run), batch 8, mixed budget
  variants, IID-validation checkpoint selection only.
- **DAPO-style hygiene:** drop/resample zero-variance groups; log the zero-variance
  fraction, reward std, and entropy per step (Echo-Trap early warning). These plots
  go in the blog — "we instrumented the known pathologies" is table stakes.
- **Control run** (Spurious-Rewards protocol): one 40-step run rewarding *format
  only* (the floor, no causal term). Cheap on Modal (no interventions needed). If
  format-only RL recovers most of the gain, our reward isn't doing the work — better
  to learn that ourselves than from a reviewer.

**Deliverable:** two RL checkpoints + one control checkpoint + training-dynamics
plots.

### Stage 5 — The verdict (~$400 Prime, $300 Modal)

Pre-registered before unsealing (freeze this file and record its hash; cite
preregistration-for-agents norms):

- **Primary contrast — two pre-registered scalars, not one:**
  1. at budget 7 on sealed families, `(RL_real − RL_sham) − (SFT_real − SFT_sham)`,
     RL seeds averaged;
  2. the **budget slope**: the same difference-in-differences fitted across budgets
     {0, 3, 7}. The claim in §1 is about *gain growing with budget* — a single
     budget-7 contrast can't test it, and reporting only the point that looks best
     post hoc is the failure mode preregistration exists to prevent.
- Also pre-register, now: rollouts per task per arm, task-level (not
  rollout-level) bootstrap CIs, equal weighting across the three sealed families,
  and the exact definition of pass@k on a continuous score (threshold + unbiased
  estimator).
- **Headline figure:** v2 reward vs experiment budget {0, 3, 7}, sealed families,
  base / SFT / SFT+RL; solid = real tool, dashed = **sham tool** (backend swaps
  feature IDs by a hidden matched derangement — internally consistent, causally
  wrong; validated: true-vs-sham effect correlation ≈ 0 on dev). Sham-tool arms are
  the strong version of MAIA's tool ablations / Anthropic's affordance ablations.
- **Baselines on the same plot or table:**
  1. frontier models under v2 (prompted, from Stage 3),
  2. **non-agentic heuristic**: spectral/community partition of the attribution
     graph + influence-ranked relations, zero experiments — MIB found simple
     attribution methods embarrassingly strong; we must beat this or the RL story
     collapses,
  3. matched random partitions (size/layer/influence-matched, same budget — Shi
     et al. convention),
  4. the full scripted oracle suite from Stage 1, including attack (d).
- **Secondary analyses:** pass@k (k up to 16–32; Modal cost caps k) base vs RL on
  sealed tasks — speaks to the sharpening-vs-new-capability debate; robustness spot
  check of model rankings under a second ablation scheme (mean-ablation) on 10
  graphs; offline label-coherence of supernode names against the eval-only label
  side-file vs matched random groupings (valid only because §3 Stage 2.3 keeps
  labels out of the prompt; ~38% of auto-interp labels fail audits, so they are
  scenery, not ground truth).
- **Artifacts:** every sealed-eval proposal exported to Neuronpedia
  (`qParams.supernodes` + pinned IDs) with a **pre-registered selection rule** for
  the gallery (best/median/failure per family — no cherry-picking); environment
  published on the Environments Hub with the eval split, sham config, tracked tests,
  and a reproducible `vf-eval` baseline (hub-reviewed envs fed INTELLECT-3 —
  publication is itself a residency deliverable); blog post.

**Interpretation guide (written down now, before results):**
- RL > SFT under real tool, gain grows with budget, vanishes under sham → the claim.
- RL ≈ RL-sham → RL learned better priors/formatting, not experimentation (report
  as such).
- Heuristic ≈ RL → the graphs, not the agent, carry the information (report as
  such).
- Any of the negative outcomes + the oracle audit still make an honest, useful blog.

---

## 4. Budget

| Stage | Prime | Modal | Reserve notes |
|---|---:|---:|---|
| 0 Reproducibility | $0 | $0 | tests, tags, artifact archive |
| 1 Audit v1 + reward v2 | $50 | $150 | oracle runs, canary, share/frozen audits |
| 2 Data | $50 | $450 | ~220 graph builds + screening |
| 3 Feasibility + SFT | $700 | $350 | evals, ≤1k-rollout harvest, v2 baseline |
| 4 RL (2 seeds + control) | $2,050 | $500 | ~40 steps each, G=8 |
| 5 Sealed eval + artifacts | $400 | $300 | paired real/sham, exports |
| Reserve | $750 | $250 | failures only — not scope growth |
| **Ceiling** | **$4,000** | **$2,000** | |

Stage 5's Modal line is **$300** — an earlier draft said $550 in the section header
and $300 in this table; the table is authoritative and the column sums to the
ceiling.

Published small-model agentic RL runs land at 40–480 H100-hours (ART·E: ~$80), so
dollars are not the risk — reward integrity, dataset quality, and Modal *throughput*
are. Note that v2 roughly doubles scoring cost (2k+2 vs k+2 passes); Stage 4 still
fits, but 64 concurrent rollouts against 8 containers is a wall-clock risk that
canary 1.1 must quantify before we commit to two seeds.

## 5. What we will NOT do

1. No RL before Stage 1 lands (canary + oracle audit are hard blockers).
2. No Stage 2 spend before the `share` audit clears (Stage 1.3).
3. No frontier-model distillation (ToS) — self/open-family traces only.
4. No LLM judge, name scoring, or label term anywhere in the reward.
5. **No feature labels in the prompt** — they are eval-only evidence, and the
   moment they appear in the rendered graph the label-coherence analysis dies.
6. No touching the sealed families after the Stage 2 freeze, for anything, until
   Stage 5.
7. No claiming "interpretability", and no claiming *edges*. The reward measures
   causal consistency of a **downstream-reachability hypothesis** *within a
   frozen-attention replacement model on a pruned top-60 slice*; edge sets with
   identical transitive closure are indistinguishable to it. Labels are measured
   offline. Say all of this plainly (limitations: transcoder quality ceiling,
   faithfulness-metric fragility, dormant-pathway illusions, self-repair
   underestimation).
8. No dataset regeneration mid-experiment; no scope growth from the reserve.
9. No comparing models under mismatched thinking/sampling settings.
10. No reuse of the v1 frontier chart as a capability result — only as an audit
    exhibit, always with the component table from §2.

## 6. Sources (key ones)

*Circuit tracing:* Ameisen et al. "Circuit Tracing" + Lindsey et al. "Biology of an
LLM" (transformer-circuits.pub, 2025); open-sourcing post (anthropic.com, May 2025);
circuit-tracer (BlackboxNLP 2025); Circuits Research Landscape (neuronpedia.org/graph/info,
Aug 2025); Circuits Updates May/June 2026 (transformer-circuits.pub); CLT-unfaithfulness
critique (LessWrong, Feb 2026); mwhanna/qwen3-4b-transcoders.
*Nearest neighbors:* MAIA (arXiv:2404.14394); FIND (arXiv:2309.03886); HyVE/
AgenticInterpBench (arXiv:2606.24026); ADAG (arXiv:2604.07615); probe prompting
(arXiv:2511.07002); MechRL (arXiv:2605.26343); Transluce investigator agents
(arXiv:2502.01236) + self-explanation (arXiv:2511.08579); Pitfalls in Evaluating
Interpretability Agents (arXiv:2603.20101).
*Metrics & controls:* MIB (arXiv:2504.13151); Shi et al. circuit-hypothesis tests
(arXiv:2410.13032); Miller et al. faithfulness-not-robust (arXiv:2407.08734);
Makelov et al. illusions (arXiv:2311.17030); Hydra effect (arXiv:2307.15771); RAVEL
(arXiv:2402.17700); preregistration for AI agents (arXiv:2606.11217).
*RL practice:* Yue et al. RLVR limits (arXiv:2504.13837); Spurious Rewards
(arXiv:2506.10947); ProRL; DAPO/Dr.GRPO line (arXiv:2607.00152); RAGEN Echo Trap
(arXiv:2504.20073); WebSailor/UserRL/InfoAgent SFT-warm-start evidence; Qwen3 RL
recipe (arXiv:2505.09388); ART·E (openpipe.ai); prime-rl + verifiers docs +
Environments Hub program (primeintellect.ai).
