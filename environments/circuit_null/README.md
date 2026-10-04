# circuit-null

Bio-Circuit Verifiers v1 harness with pinned MCP and extended tool timeouts.

This is a small Hub distribution for the `circuit-null` harness. It depends on an
version-pinned public `circuit-supernodes==0.1.1` wheel, which owns
all implementation modules. The distribution adds no duplicate plugin code.

Install with `prime env install wazupsteve/circuit-null`. In Verifiers v1, select
`harness.id = "wazupsteve/circuit-null@0.1.2"`. This is the shared policy harness,
not an additional taskset. All Bio-Circuit modules come from the core package.

Public data: [amitprakash2005/bio-circuit](https://huggingface.co/datasets/amitprakash2005/bio-circuit).
Full setup: [circuit-supernodes](https://app.primeintellect.ai/dashboard/environments/wazupsteve/circuit-supernodes).

**Live tools and scoring require an external CUDA GPU backend.** This release
has 16 evaluation tasks per taskset and no training split or validated SFT
trajectories. Downloading and inspecting the data works without a GPU. Follow
the core package README to run your own backend; use credentials for the Modal workspace that owns your deployment. This release does not supply a hosted GPU endpoint or container image.
