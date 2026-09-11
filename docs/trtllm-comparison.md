# Reproducible three-framework comparison

Run all commands on the same host with the same model revision and GPU set. The
example uses one GPU and a synthetic 1,024-input/256-output-token workload so
the workload shape is equivalent. Adjust capacity settings only when required
by the installed framework release.

```bash
COMMON_CLIENT="dataset_name=random;random_input_len=1024;random_output_len=256;max_concurrency=32;num_prompts=1000;seed=1"

llm-optimizer \
  --framework vllm \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --server-args "tensor_parallel_size=1;max_num_seqs=32;max_num_batched_tokens=8192" \
  --client-args "$COMMON_CLIENT" \
  --gpus 1 \
  --output-json vllm.json

llm-optimizer \
  --framework sglang \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --server-args "tp_size=1;max_running_requests=32;max_total_tokens=8192" \
  --client-args "$COMMON_CLIENT" \
  --gpus 1 \
  --output-json sglang.json

llm-optimizer \
  --framework trtllm \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --server-args "tp_size=1;max_batch_size=32;max_num_tokens=8192" \
  --client-args "$COMMON_CLIENT" \
  --gpus 1 \
  --output-json trtllm.json

llm-optimizer visualize \
  --data-file "vllm.json,sglang.json,trtllm.json" \
  --output three-frameworks.html
```

The native settings above serialize to common `normalized_config` fields:

| Meaning | vLLM | SGLang | TensorRT-LLM |
| --- | --- | --- | --- |
| Tensor parallel size | `tensor_parallel_size` | `tp_size` | `tp_size` |
| Concurrent sequences | `max_num_seqs` | `max_running_requests` | `max_batch_size` |
| Batched tokens | `max_num_batched_tokens` | `max_total_tokens` | `max_num_tokens` |

The benchmark client—not a framework-provided benchmark—measures all three, so
TTFT, ITL/TPOT, end-to-end latency, request throughput, and token throughput
retain the same definitions. Performance is not expected to be equal; the test
checks that equivalent experiments are represented and visualized equivalently.
