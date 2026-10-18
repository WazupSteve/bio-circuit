# suppressed-features

Find and undo a hidden feature corruption that changes Qwen3-4B's outputs.

This is a thin Prime Hub distribution that depends on
`circuit-supernodes==0.2.4`, which contains all implementation code; this
package adds none.

Install the core package first, running `prime` as a tool so it stays out of
your environment's venv:

```bash
uvx --from prime prime env install wazupsteve/circuit-supernodes
uvx --from prime prime env install wazupsteve/suppressed-features
```

Then select `env.taskset.id = "wazupsteve/suppressed-features@0.2.2"` and use the `circuit-null`
harness.

Data: [amitprakash2005/bio-circuit](https://huggingface.co/datasets/amitprakash2005/bio-circuit)
(16 evaluation tasks per taskset, no training split). Setup:
[circuit-supernodes](https://app.primeintellect.ai/dashboard/environments/wazupsteve/circuit-supernodes).

**Tools and scoring need a CUDA GPU subject backend**, which this release does
not provide. Follow the core package README to deploy one with your own Modal
credentials. The data itself downloads and opens without a GPU.
