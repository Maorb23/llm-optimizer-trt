# TensorRT-LLM and NVIDIA Dynamo

## Decision

Dynamo is not part of the initial TensorRT-LLM integration. The native managed
backend launches `trtllm-serve`, waits on its OpenAI-compatible health endpoint,
and benchmarks the same completions API used for vLLM and SGLang. That is the
least confounded path for comparing inference engines on one node.

Native TensorRT-LLM is appropriate for:

- single-node inference benchmarking;
- TP, PP, CP, batch, and sequence-limit tuning;
- KV-cache and executor configuration;
- checkpoint precision and quantization comparisons; and
- latency, throughput, and Pareto comparisons with vLLM and SGLang.

Dynamo adds value when the experiment is about deployment topology rather than
only engine behavior. NVIDIA documents TensorRT-LLM support for disaggregated
prefill/decode serving, KV-aware routing, distributed workers, multi-node
deployments, planners, and request routing. Those experiments should use a
future Dynamo integration because a single native server command cannot model
their component graph faithfully.

## Future integration seam

Managed frameworks are described separately from their benchmark client and
server lifecycle defaults. A future `deployment_mode=dynamo` can select a
Dynamo launcher while retaining `framework=trtllm`, native configuration
normalization, result serialization, and the common benchmark client. This
keeps these concepts distinct:

```text
inference engine:  vllm | sglang | trtllm
deployment mode:   native | dynamo (future)
```

The initial change does not add a general plugin framework or expose the future
CLI option. It leaves only the small descriptor seam needed to change the
launcher/readiness behavior later.

## Benchmark ownership

Keep direct engine tuning native. Use Dynamo when the independent variables are
worker pools, prefill/decode placement, KV-aware routing, router policy,
multi-node transport, or production-scale topology. Results from a future mode
should record both `framework` and `deployment_mode` so engine and orchestration
effects are never silently conflated.

References:

- [TensorRT-LLM backend overview](https://docs.nvidia.com/dynamo/dev/knowledge-base/modular-components/backends/tensor-rt-llm/overview)
- [Dynamo disaggregated serving](https://docs.nvidia.com/dynamo/components/router/disaggregated-serving)
- [Dynamo KV-aware routing](https://docs.nvidia.com/dynamo/cli/kv-aware-routing/overview)
