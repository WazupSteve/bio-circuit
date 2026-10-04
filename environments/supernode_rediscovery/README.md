# supernode-rediscovery

Rediscover causal Qwen3-4B circuits for named semantic goals.

This is a small Hub distribution for the `supernode-rediscovery` taskset. It depends on an
version-pinned public `circuit-supernodes==0.1.1` wheel, which owns
all implementation modules. The distribution adds no duplicate plugin code.

Install with `prime env install wazupsteve/supernode-rediscovery`. In Verifiers v1, select
`taskset.id = "wazupsteve/supernode-rediscovery@0.1.2"` and use the `circuit-null` harness for
tasksets. All Bio-Circuit modules are installed with the shared core package.

Public data: [amitprakash2005/bio-circuit](https://huggingface.co/datasets/amitprakash2005/bio-circuit).
Full setup: [circuit-supernodes](https://app.primeintellect.ai/dashboard/environments/wazupsteve/circuit-supernodes).

**Live tools and scoring require an external CUDA GPU backend.** This release
has 16 evaluation tasks per taskset and no training split or validated SFT
trajectories. Downloading and inspecting the data works without a GPU. Follow
the core package README to run your own backend; use credentials for the Modal workspace that owns your deployment. This release does not supply a hosted GPU endpoint or container image.
