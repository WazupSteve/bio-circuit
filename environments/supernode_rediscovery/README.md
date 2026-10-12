# supernode-rediscovery

Find causal Qwen3-4B features for a named semantic goal, with no example prompts.

This is a thin Prime Hub distribution. It installs the pinned
`circuit-supernodes==0.1.1` package, which contains all implementation code;
this package adds none.

Install with `prime env install wazupsteve/supernode-rediscovery`, select
`taskset.id = "wazupsteve/supernode-rediscovery@0.1.2"` and use the `circuit-null`
harness.

Data: [amitprakash2005/bio-circuit](https://huggingface.co/datasets/amitprakash2005/bio-circuit)
(16 evaluation tasks per taskset, no training split). Setup:
[circuit-supernodes](https://app.primeintellect.ai/dashboard/environments/wazupsteve/circuit-supernodes).

**Tools and scoring need a CUDA GPU subject backend**, which this release does
not provide. Follow the core package README to deploy one with your own Modal
credentials. The data itself downloads and opens without a GPU.
