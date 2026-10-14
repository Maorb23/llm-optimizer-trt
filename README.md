# LLM Optimizer

llm-optimizer is a Python tool for benchmarking and optimizing inference performance of any open-source LLMs.

For remote GPU jobs, see [Run a Nebius GPU optimization job](#run-a-nebius-gpu-optimization-job).

- 🧩 Benchmark SGLang, vLLM, and NVIDIA TensorRT-LLM using native arguments
- ⚡️ Find the optimal setup automatically for your use case without endless trial and error
- 🎯 Apply SLO constraints to focus only on configurations that meet your performance goals
- 🧮 Estimate performance theoretically without running full benchmarks
- 📊 Visualize results interactively with dashboards for clear analysis

[![Release Blog](https://img.shields.io/badge/Release_Blog-d0bfff?style=for-the-badge)](https://www.bentoml.com/blog/announcing-llm-optimizer)
[![LLM Performance Explorer](https://img.shields.io/badge/LLM_Performance_Explorer-eefbe4?style=for-the-badge)](https://www.bentoml.com/llm-perf)

Interested in optimizing disaggregated LLM inference? [👉 Contact us](https://www.bentoml.com/contact)

## Installation

Install llm-optimizer with `pip`.

```bash
pip install -e .
```

For development:
```bash
pip install -e .[dev]
```

Optuna-based Bayesian search is an optional installation:

```bash
pip install -e '.[optuna]'
```

## Run a Nebius GPU optimization job

### Build framework worker images

The remote worker must include vLLM and this repository's Python package. Build
the supplied image once from the repository root. Pin the vLLM image tag you
have tested rather than relying on `latest`:

```bash
docker build \
  --file docker/vllm.Dockerfile \
  --build-arg VLLM_IMAGE=vllm/vllm-openai:latest \
  --tag llm-optimizer-vllm:local \
  .
```

The image you build locally must be pushed to a registry that Nebius can pull
from. Use the resulting registry image reference in `request.json` under
`images.vllm`.

SGLang and TensorRT-LLM use separate base images and Dockerfiles. Build only
the frameworks you select in the request. Build from the repository root:

```bash
docker build --file docker/sglang.Dockerfile \
  --tag llm-optimizer-sglang:local .

docker build --file docker/trtllm.Dockerfile \
  --tag llm-optimizer-trtllm:local .
```

The defaults use versioned upstream images. Override `SGLANG_IMAGE` or
`TRTLLM_IMAGE` with `--build-arg` only after checking GPU, CUDA, and optimizer
argument compatibility. The SGLang Dockerfile uses the official SGLang release
image; the TensorRT-LLM Dockerfile uses NVIDIA's prebuilt release image, so it
does not compile TensorRT-LLM from source. Each build checks that the framework
and optimizer worker import successfully. A GPU job is still needed to verify
serving and benchmarking.

For a Nebius registry, use the registry ID suffix (without `registry-`) in the
image path, not the project ID. For example, after creating a registry and
configuring Docker credentials:

```bash
nebius registry configure-helper
docker tag llm-optimizer-sglang:local \
  cr.eu-north1.nebius.cloud/REGISTRY_ID/llm-optimizer-sglang:v1
docker push cr.eu-north1.nebius.cloud/REGISTRY_ID/llm-optimizer-sglang:v1

docker tag llm-optimizer-trtllm:local \
  cr.eu-north1.nebius.cloud/REGISTRY_ID/llm-optimizer-trtllm:v1
docker push cr.eu-north1.nebius.cloud/REGISTRY_ID/llm-optimizer-trtllm:v1
```

Set `images.sglang` and `images.trtllm` to those full registry references when
selecting those frameworks. Check available disk space and Docker's footprint
before pulling another large framework image:

```bash
docker system df
df -h .
```

Remote execution is opt-in. `submit` queues a run in a persistent controller;
the controller creates one Nebius job per framework. A dry run creates the job
plan without starting a GPU or uploading artifacts.

### Set up the controller and worker images

1. Install the controller dependencies: `pip install -e '.[remote]'`.
2. [Install and configure the Nebius CLI](https://docs.nebius.com/cli/install)
   on the controller host. Its active profile must have permission to create,
   inspect, and cancel jobs in your Nebius project. For an unattended controller,
   use Nebius's [service-account setup](https://docs.nebius.com/cli/configure).
3. Create an S3-compatible bucket and credentials that can upload and download
   objects. The controller and GPU workers both need access to it. Install the
   AWS CLI if you want to use the download command below.
4. Build or select a **separate image for each framework**. Each image must
   contain this `llm-optimizer` package, that framework's GPU runtime, its
   benchmark dependencies, `boto3`, and `optuna` for Optuna requests. Check
   CUDA and driver compatibility for the requested GPU. Replace the image and
   bucket placeholders below with real values; the repository does not publish
   ready-to-run framework images.
5. Put the controller SQLite database on durable local storage and run one
   controller process for that database.

Copy the environment template, replace every placeholder, and source it in the
shell where you configure Nebius and start the controller:

```bash
cp .env.example .env
# Edit .env with your project, service-account, S3, and optional HF values.
set -a
. ./.env
set +a
mkdir -p .local
```

`.env` is ignored by Git. Sourcing it exports its values to the Nebius CLI and
optimizer processes; merely creating the file does not load it automatically.

`hf_validator` configures its CLI profile in
`scripts/run_prefect_controller_job.sh` in that repository.
This optimizer invokes the Nebius CLI directly, so configure a profile **once
on the controller host** before starting `serve`. Create a Nebius service
account and authorized key, then copy its service-account ID, public-key ID,
and PEM private-key path from your Nebius setup:

```bash
nebius profile create \
  --endpoint api.nebius.cloud \
  --service-account-id "$SA_ID" \
  --public-key-id "$PUBLIC_KEY_ID" \
  --private-key-file "$NEBIUS_PRIVATE_KEY_PATH" \
  --profile llm-optimizer-controller \
  --parent-id "$NB_PROJECT_ID"
nebius profile list
nebius ai job list --parent-id "$NB_PROJECT_ID" --format json
```

The last command checks CLI authentication without starting a GPU job. For a
local browser login, create the profile this way instead:

```bash
nebius profile create \
  --profile llm-optimizer-controller \
  --endpoint api.nebius.cloud \
  --federation-endpoint auth.nebius.com \
  --parent-id "$NB_PROJECT_ID"
nebius ai job list --parent-id "$NB_PROJECT_ID" --format json
```

The `NEBIUS_CONFIG_YAML_B64` and
`NEBIUS_SERVICE_ACCOUNT_KEY_B64` bootstrap inputs in `hf_validator` are handled
by its script; this optimizer does not decode them. It uses the active Nebius
CLI profile and `NB_PROJECT_ID`. The `AWS_*` variables in `.env` authenticate S3;
they do not authenticate the Nebius CLI.

`HF_TOKEN` is needed for gated models. The controller database path in `.env`
is relative to the repository root; use an absolute path when running the
controller from another directory.

Create `request.json` for one vLLM job. The GPU platform and preset must be
available in your Nebius project. The preset's GPU count must match `gpu_count`.
Nebius job timeouts must be between 60 and 10,080 minutes.

```json
{
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "frameworks": ["vllm"],
  "gpu_type": "gpu-h100-sxm",
  "gpu_count": 1,
  "preset": "1gpu-16vcpu-96gb",
  "images": {"vllm": "registry.example/vllm-optimizer:cuda-tag"},
  "bucket": "your-results-bucket",
  "prefix": "llm-optimizer",
  "client_args": ["dataset_name=random;random_input_len=128;random_output_len=32;num_prompts=16"],
  "server_args": {"vllm": ["max_num_seqs=[8,16]"]},
  "search_strategy": "optuna",
  "optuna": {"trials": 2, "seed": 7},
  "timeout_minutes": 180,
  "dry_run": true
}
```

Start the controller and submit from the same configured shell. For a lasting
deployment, run `serve` as a supervised service instead of a shell background
process:

```bash
python -m llm_optimizer.remote serve --interval 15 >.local/controller.log 2>&1 &
python -m llm_optimizer.remote submit request.json --idempotency-key example-dry-run-001
```

The `submit` response contains `run_id`. Use it to inspect the dry-run plan:

```bash
RUN_ID="run-id-from-submit-response"
python -m llm_optimizer.remote status "$RUN_ID"
```

To launch a **billable GPU job**, set `"dry_run": false` in `request.json` and
submit it with a **new idempotency key**. Keep the controller running; it submits
the queued job and polls its state. `submit` itself returns immediately.

```bash
python -m llm_optimizer.remote submit request.json --idempotency-key example-live-001
```

Set `RUN_ID` to the new `run_id` returned by this live submission before using
the status and download commands below.

For a multi-framework run, add framework names to `frameworks`, an image for
each one in `images`, and any framework-specific settings in `server_args`.
For example, add `"sglang"` and a compatible SGLang image to run vLLM and
SGLang as independent jobs. Start the controller with
`python -m llm_optimizer.remote --concurrency 2 serve` to allow both
jobs to run at once. Use `optuna_by_framework` if their Optuna search spaces
differ. Each framework gets its own S3 prefix and a failure in one does not
discard the other's results.

Check progress and retrieve results:

```bash
python -m llm_optimizer.remote status "$RUN_ID"
python -m llm_optimizer.remote results "$RUN_ID"
aws s3 sync "s3://your-results-bucket/llm-optimizer/$RUN_ID/" ./downloaded-run/ \
  --endpoint-url "$AWS_ENDPOINT_URL_S3"
```

The framework manifests and `aggregate.json` contain artifact references.
Use `python -m llm_optimizer.remote cancel "$RUN_ID"` to cancel the whole run,
or add `--framework vllm` for one job. See [Nebius remote optimization](docs/nebius-remote.md)
for the full request schema, checkpoint recovery, and controller behavior.

## Get started

The quickest way to try llm-optimizer is with performance estimation. This feature predicts latency, throughput, and concurrency limits, helping you identify optimal configurations without running full benchmarks:

```bash
llm-optimizer estimate \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --gpu A100 \  # No need to provide gpu type if llm-optimizer can detect the local machine's GPU type and it's supported
  --input-len 1024 \
  --output-len 512
```

> [!NOTE]
> For gated models, request access on Hugging Face and export your token in advance:
> ```bash
> export HF_TOKEN=<your token>
> ```

More examples:

```bash
# With GPU specification
llm-optimizer estimate \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --input-len 2048 \
  --output-len 1024 \
  --gpu H100 \
  --num-gpus 8

# With performance constraints and command generation
llm-optimizer estimate \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --input-len 1024 \
  --output-len 512 \
  --gpu H100 \
  --num-gpus 4 \
  --constraints "ttft:mean<300ms;itl:p95<50ms"
```

For guided setup, you can also use interactive mode:

```bash
llm-optimizer estimate --interactive
```

## Run your first benchmark

llm-optimizer supports SGLang, vLLM, and NVIDIA TensorRT-LLM (`trtllm`).

Here is an example using SGLang:

```bash
# SGLang with multiple TP/DP combinations
llm-optimizer \
  --framework sglang \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --server-args "tp_size*dp_size=[(1,4),(2,2),(4,1)];chunked_prefill_size=[2048,4096,8192]" \
  --client-args "max_concurrency=[50,100,200];num_prompts=1000" \
  --output-json sglang_results.json
```

This command will:

- Test 3 TP/DP combinations × 3 prefill sizes = 9 server configurations
- Test each against 3 concurrency values = 27 client configurations
- In total, 27 unique benchmarks will be run, with results saved in `sglang_results.json`

### Bayesian search with Optuna

Grid search remains the default. Add `--search-strategy optuna` to tune one
framework and model with Optuna's seeded TPE sampler. The existing scalar,
list, `range(...)`, boolean, and coupled `tp_size*dp_size` syntax is reused as a
finite categorical space. Optuna-only integer distributions use separate
`--optuna-server-args` and `--optuna-client-args` options. `--output-json` is
required. Samplers and pruners can be selected by Optuna class name or fully
qualified import path, with JSON keyword arguments passed through using
`--optuna-sampler-kwargs` and `--optuna-pruner-kwargs`.

```bash
# vLLM: maximize output throughput (the default objective)
llm-optimizer --framework vllm --model meta-llama/Llama-3.1-8B-Instruct \
  --server-args "tensor_parallel_size=[1,2];max_num_seqs=[16,32,64]" \
  --client-args "max_concurrency=[16,32,64,128]" \
  --search-strategy optuna --optuna-trials 20 \
  --output-json vllm_optuna.json

# SGLang: minimize p95 time to first token while enforcing SLOs
llm-optimizer --framework sglang --model meta-llama/Llama-3.1-8B-Instruct \
  --server-args "tp_size*dp_size=[(1,4),(2,2),(4,1)];chunked_prefill_size=[2048,4096]" \
  --client-args "max_concurrency=[32,64,128]" \
  --constraints "itl:p95<10ms" --search-strategy optuna \
  --optuna-objective p95_ttft_ms --optuna-timeout 3600 \
  --output-json sglang_optuna.json

# TensorRT-LLM: native argument spelling and a reproducible sampler seed
llm-optimizer --framework trtllm --model meta-llama/Llama-3.1-8B-Instruct \
  --server-args "tp_size=[1,2,4];max_batch_size=[32,64]" \
  --client-args "dataset_name=random;random_input_len=1024;random_output_len=256;max_concurrency=[32,64]" \
  --search-strategy optuna --optuna-seed 7 --optuna-trials 16 \
  --output-json trtllm_optuna.json
```

For a genuine Optuna integer search, keep fixed and categorical arguments in
the existing options and put numeric ranges in the Optuna-only options:

```bash
llm-optimizer --framework vllm --model meta-llama/Llama-3.1-8B-Instruct \
  --server-args "tensor_parallel_size=1" \
  --optuna-server-args "max_num_seqs=int(16,64);max_num_batched_tokens=int(4096,16384,4096)" \
  --client-args "dataset_name=random;num_prompts=128" \
  --optuna-client-args "max_concurrency=int(8,32,8)" \
  --search-strategy optuna --optuna-trials 20 \
  --output-json vllm_integer_optuna.json
```

`int(low,high[,step])` includes both bounds; the step defaults to 1 and must
land exactly on the high bound. These values use Optuna's `suggest_int`, not
categorical sampling. Do not repeat a parameter in both argument options.
Grid search does not accept the Optuna-only options. Changing a range changes
the study signature, so use a fresh output/study or the original range when
resuming with `--continue`.

Throughput objectives (`request_throughput`, `input_throughput`,
`output_throughput`, and `total_throughput`) are maximized. Mean, median, p95,
and p99 TTFT, ITL, TPOT, and end-to-end fields such as `p95_ttft_ms` and
`mean_e2e_latency_ms` are minimized. Completed trials that miss an SLO remain
useful to constrained TPE, but are excluded from the reported feasible best.
Startup errors, OOMs, benchmark errors, and missing or non-numeric objectives
are recorded as failed trials; the server is stopped before search continues.

By default, durable study state is stored in `<output-name>.optuna.db` beside
the JSON output. Resume both that study and its JSONL results with the exact
same arguments:

```bash
llm-optimizer --framework vllm --model meta-llama/Llama-3.1-8B-Instruct \
  --server-args "max_num_seqs=[16,32,64]" \
  --client-args "max_concurrency=[16,32,64]" \
  --search-strategy optuna --optuna-trials 9 \
  --output-json vllm_optuna.json --continue
```

A canonical signature prevents `--continue` from mixing a changed model,
command, framework, space, objective, constraints, GPU count, host, or port.
For PostgreSQL or MySQL, pass an Optuna/SQLAlchemy URL through
`--optuna-storage` and optionally set `--optuna-study-name`. Credentials are
never copied into result files. Optuna mode currently supports one active
worker per study; remote RDB storage provides durability, not distributed
parallel execution. Use `--dry-run` to print cardinality, objective direction,
budget, timeout, and storage kind without creating result or database files.

More examples:

```bash
# vLLM with batch size tuning
llm-optimizer \
  --framework vllm \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --server-args "tensor_parallel_size*data_parallel_size=[(1,2),(2,1)];max_num_batched_tokens=[4096,8192,16384]" \
  --client-args "max_concurrency=[32,64,128];num_prompts=1000;dataset_name=sharegpt" \
  --output-json vllm_results.json

# Complex parameter grid for throughput optimization
llm-optimizer \
  --framework sglang \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --server-args "tp_size*dp_size=[(1,8),(2,4),(4,2)];schedule_conservativeness=[0.3,0.6,1.0];chunked_prefill_size=range(2048,8193,2048)" \
  --client-args "max_concurrency=range(50,201,50);request_rate=[10,20,50]" \
  --gpus 8 \
  --output-json complex_benchmark.json

# TensorRT-LLM with native executor and batching parameters
llm-optimizer \
  --framework trtllm \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --server-args "tp_size=[1,2,4];max_batch_size=[32,64];max_num_tokens=[4096,8192]" \
  --client-args "dataset_name=random;random_input_len=1024;random_output_len=256;max_concurrency=[32,64,128];num_prompts=1000" \
  --gpus 4 \
  --output-json trtllm_results.json
```

### TensorRT-LLM installation

TensorRT-LLM remains an external, backend-specific dependency. Installing
llm-optimizer does not install it, so vLLM and SGLang users do not inherit its
CUDA, PyTorch, or NVIDIA library constraints. NVIDIA recommends its pre-built
[TensorRT-LLM NGC release container](https://nvidia.github.io/TensorRT-LLM/latest/installation/installation-guide.html).
Run llm-optimizer inside that container (mount this checkout and the model/cache
directories), or install the compatible `tensorrt_llm` wheel on Linux. Confirm
that `trtllm-serve --help` works before starting a managed benchmark.

The integration uses the current `trtllm-serve` OpenAI-compatible API and its
`/health` endpoint. Current TensorRT-LLM uses the PyTorch backend and loads a
Hugging Face checkpoint directly; it no longer requires the legacy
`trtllm-build` engine step. Nested settings such as quantization, KV block
reuse, scheduler configuration, CUDA graphs, and speculative decoding should
be placed in a native YAML file and swept by passing alternative files through
`config=[...]`.

See [TensorRT-LLM comparison benchmark](docs/trtllm-comparison.md) for an
equivalent three-framework experiment and [TensorRT-LLM and Dynamo](docs/trtllm-dynamo.md)
for the deployment-topology design.

## Apply constraints

Not every benchmark result is useful. You can apply constraints directly to your benchmarks so only configurations that meet your Service Level Objectives (SLOs) are returned.

```bash
# Latency-optimized configuration
llm-optimizer \
  --framework vllm \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --server-args "tensor_parallel_size*data_parallel_size=[(1,2),(2,1)];max_num_seqs=[16,32,64]" \
  --client-args "max_concurrency=[8,16,32];num_prompts=500" \
  --constraints "ttft<200ms;itl:p99<10ms" \
  --output-json latency_optimized.json
```

Currently llm-optimizer supports constraints on key performance metrics using mean, median, p95, or p99 values. Constraint syntax:

```bash
# Time to first token constraints
--constraints "ttft<300ms"                    # Mean TTFT under 300ms
--constraints "ttft:median<200ms"             # Median TTFT under 200ms
--constraints "ttft:p95<500ms"                # 95th percentile under 500ms

# Inter-token latency constraints
--constraints "itl:mean<20ms"                 # Mean ITL under 20ms
--constraints "itl:p99<50ms"                  # 99th percentile under 50ms

# End-to-end latency constraints
--constraints "e2e_latency:p95<2s"           # 95th percentile under 2s

# Combined constraints
--constraints "ttft:median<300ms;itl:p95<10ms;e2e_latency:p95<2s"
```

## Visualize benchmark results

llm-optimizer saves benchmark results in JSON format, including key inference metrics like TTFT, ITL, and concurrency. Since raw numbers can be difficult to interpret, it provides an interactive visualization tool to help you explore results more easily.

```bash
# Visualize results with Pareto frontier analysis
llm-optimizer visualize --data-file results.json --serve --port 8080

# Combine multiple result files
llm-optimizer visualize \
  --data-file "sglang_results.json,vllm_results.json,trtllm_results.json" \
  --serve \
  --port 8080
```

Open your browser at `http://localhost:8080/pareto_llm_dashboard.html`. The dashboard allows you to:

- Compare results from multiple runs side by side
- Explore trade-offs between different setups (e.g., latency vs. throughput)
- Identify the best-performing configurations for your workload

> [!NOTE]
> This feature is still experimental, and we’ll continue improving it in the coming days. For visualized results, check out the [LLM Performance Explorer](https://www.bentoml.com/llm-perf/).

## Use custom server commands

By default, llm-optimizer manages server startup for supported frameworks. If you want more control, you can provide your own server command.

```bash
# Custom SGLang server
llm-optimizer \
  --server-cmd "python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --host 0.0.0.0 --port 30000" \
  --client-args "max_concurrency=[25,50,100];num_prompts=1000" \
  --host 0.0.0.0 \
  --port 30000

# Custom vLLM server with specific GPU allocation
llm-optimizer \
  --server-cmd "vllm serve meta-llama/Llama-3.1-8B-Instruct --tensor-parallel-size 4" \
  --client-args "max_concurrency=[64,128];num_prompts=2000" \
  --port 8000
```

## Tune inference parameters

llm-optimizer exposes native server parameters and common client parameters so
you can measure the same workload across supported frameworks.

### SGLang
- `tp_size*dp_size`: Tensor/Data parallelism combinations
- `chunked_prefill_size`: Prefill chunk size for throughput
- `schedule_conservativeness`: Request scheduling aggressiveness
- `schedule_policy`: Scheduling policy (fcfs, priority)

### vLLM
- `tensor_parallel_size`: Tensor parallelism degree
- `max_num_batched_tokens`: Maximum batch size in tokens
- `max_num_seqs`: Maximum concurrent sequences

### TensorRT-LLM

- `tp_size`, `pp_size`, `cp_size`, `ep_size`: native parallel dimensions
- `max_batch_size`: maximum requests the executor can schedule
- `max_num_tokens`: maximum unpadded input tokens in a batch
- `max_seq_len`: maximum prompt plus output length
- `free_gpu_memory_fraction`: free GPU memory reserved for KV cache
- `kv_cache_dtype`: `auto`, `fp8`, or `nvfp4` where supported
- `enable_chunked_prefill`, `enable_attention_dp`: native scheduling/parallel options
- `config`: native YAML for nested engine, KV cache, quantization, scheduler,
  CUDA graph, and speculative-decoding settings

TRT-LLM CLI argument names use underscores exactly as documented. The optimizer
also accepts the documented long aliases such as `tensor_parallel_size`.

Known limitations: model, quantization, precision, and GPU-architecture support
are determined by the installed TensorRT-LLM release. Multi-node and
prefill/decode-disaggregated topologies are outside the native backend's scope.
Use the Dynamo configuration pipeline for multi-node deployment validation.

## Validate NVIDIA Dynamo deployments

Install the optional AI Configurator integration with Python 3.11 through 3.13:

```bash
pip install -e '.[ai-configurator]'
```

The two Dynamo commands have separate responsibilities:

- `dynamo-check` expands a native argument grid, validates it against the
  Kubernetes topology, and generates deterministic DGD manifests.
- `dynamo-optimize` runs NVIDIA AI Configurator's support check and GPU-free
  offline sizing sweep. It ranks aggregated and disaggregated layouts and
  writes CSV reports, Dynamo manifests, and AIPerf validation artifacts.

Batch settings are reported as distributed-worker limits because vLLM,
SGLang, and TRT-LLM use continuous batching; they are not independent per-node
batch sizes.

```bash
llm-optimizer dynamo-check \
  --framework vllm \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --node-count 2 \
  --gpus-per-node 4 \
  --server-args "tensor_parallel_size=4;pipeline_parallel_size=2;max_num_seqs=[32,64];max_num_batched_tokens=8192" \
  --output results/dynamo-vllm-check.json
```

```bash
llm-optimizer dynamo-optimize \
  --framework vllm \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --node-count 2 \
  --gpus-per-node 4 \
  --gpu-system h100_sxm \
  --backend-version 0.26.0 \
  --input-len 4000 \
  --output-len 500 \
  --ttft 600 \
  --tpot 20 \
  --output results/dynamo-vllm-optimized.json
```

The GPU system is an analytical hardware model; AI Configurator does not
allocate or require local GPUs. The node and GPU counts define its total GPU
budget.

Any explicitly supplied server-argument grid must satisfy
`TP × PP × CP = node-count × GPUs-per-node`. Add `--check-cluster` to pass each
generated manifest to the active Kubernetes API with
`kubectl apply --dry-run=server`; this validates the installed Dynamo CRD and
cluster-side admission rules without deploying anything.

### Client parameters
- `max_concurrency`: Maximum concurrent requests
- `num_prompts`: Total number of requests to send
- `dataset_name`: Dataset for request generation (`sharegpt`, `random`)
- `random_input/random_output`: Random sequence lengths

### Supported GPUs

H100, H200, A100, L20, L40, B100, B200 with accurate TFLOPS specifications.

## Development

```bash
# Code formatting and linting
ruff format
ruff check

# Type checking
mypy src/
```

## Community

llm-optimizer is actively maintained by the BentoML team. Feel free to reach out and [join our Slack community!](https://l.bentoml.com/join-slack)

## Contributing

As an open-source project, we welcome contributions of all kinds, such as new features, bug fixes, and documentation. Here are some of the ways to contribute:

- Repost a bug by [creating a GitHub issue](https://github.com/bentoml/llm-optimizer/issues/new/choose).
- [Submit a pull request](https://github.com/bentoml/llm-optimizer/compare) or help review other developers’ [pull requests](https://github.com/bentoml/llm-optimizer/pulls).


## Acknowledgements

This project uses the following open-source projects:

- [vllm-project/vllm](https://github.com/vllm-project/vllm) for production level LLM backend and benchmark client codes
- [sgl-project/sglang](https://github.com/sgl-project/sglang) for production level LLM backend and benchmark client codes

We are grateful to the developers and contributors of these projects for their hard work and dedication.

### Optuna lab commands

Use `llm-optimizer experiment run`, `experiment validate`, and `experiment report`
with `--config` and `--run-dir` to run the notebook lab from Python commands.
See [the workflow guide](docs/optuna_experiment.md) and
[the launcher notebook](docs/optuna_experiment_cli.ipynb).

[Optuna run metadata notebook](docs/optuna_run_metadata.ipynb) scans local studies
and optionally reads W&B, lists the exact runs used, and summarizes Phase B and
Phase A metadata per framework. Results are organized by phase, framework,
concurrency, and objective, with scalar best results and paired Pareto frontiers.
It includes only studies with at least five
minutes of recorded optimization time (or a documented duration fallback).
The bundled `experiment_results/wandb_run_snapshot.json` includes exported W&B
results automatically, without login or network access. To refresh live data,
install `llm-optimizer[notebook,wandb]`, authenticate with `wandb login`, and set
`LOAD_WANDB=True`; failed refreshes retain cached results. Source errors, cache
export timestamps, and excluded runs remain visible.
