# LLM Optimizer — Bento Extend

This repository extends BentoML's LLM Optimizer in two parts:

| Workflow | What it does | Start here |
| --- | --- | --- |
| **Bento Extend: original benchmark workflow + TRT-LLM** | Starts an inference server, benchmarks every configuration in a grid, applies SLO constraints, and visualizes results. The original vLLM/SGLang workflow now also supports TensorRT-LLM. | [Grid benchmark](#1-bento-extend-original-workflow--trt-llm) |
| **Added Optuna workflow** | Searches configurations within a trial/time budget using measured results. Includes direct CLI tuning and a separate experiment lab for sampler comparisons and validation. | [Optuna tuning](#2-added-optuna-workflow) |

Both workflows use the same framework adapters and benchmark metrics. Grid search remains the default; Optuna is optional.

## 1. Bento Extend: original workflow + TRT-LLM

### Install

From the repository root, with Python 3.9 or newer:

```bash
pip install -e .
```

For measured benchmarks, run on a compatible NVIDIA GPU host and install the
selected serving framework and its benchmark dependencies in the same environment.
Installing this package does **not** install vLLM, SGLang, or TensorRT-LLM.
For gated models, obtain Hugging Face access and export `HF_TOKEN` first.

### Run a grid benchmark

This single-GPU example tests 2 server settings × 2 concurrency levels = 4 configurations:

```bash
llm-optimizer \
  --framework vllm \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --server-args "tensor_parallel_size=1;max_num_seqs=[16,32]" \
  --client-args "dataset_name=random;random_input_len=128;random_output_len=32;num_prompts=64;max_concurrency=[8,16]" \
  --output-json vllm_grid.json
```

The optimizer starts and stops the server, runs each configuration, and saves
benchmark metrics. Add `--dry-run` to inspect the configurations without running
a benchmark. Repeat the command with `--continue` to resume saved work.

Arguments use `;` between parameters. Scalars are fixed; lists and `range(...)`
are swept. Coupled settings use syntax such as
`tensor_parallel_size*data_parallel_size=[(1,2),(2,1)]`.
Keep input/output lengths and request count fixed when comparing engine settings.

### Use SGLang or TensorRT-LLM

Use the framework's native server parameter names:

| Framework | `--framework` | Example `--server-args` |
| --- | --- | --- |
| vLLM | `vllm` | `tensor_parallel_size=1;max_num_seqs=[16,32]` |
| SGLang | `sglang` | `tp_size=1;chunked_prefill_size=[2048,4096]` |
| TensorRT-LLM | `trtllm` | `tp_size=1;max_batch_size=[16,32];max_num_tokens=4096` |

For TRT-LLM, run inside a compatible NVIDIA TensorRT-LLM release environment.
The supplied worker Dockerfile includes the framework and optimizer:

```bash
docker build -f docker/trtllm.Dockerfile -t llm-optimizer-trtllm:local .
```

Inside the GPU environment, confirm `trtllm-serve --help` works, then run:

```bash
llm-optimizer \
  --framework trtllm \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --server-args "tp_size=1;max_batch_size=[16,32];max_num_tokens=4096" \
  --client-args "dataset_name=random;random_input_len=128;random_output_len=32;num_prompts=64;max_concurrency=[8,16]" \
  --output-json trtllm_grid.json
```

The adapter uses `trtllm-serve` with Hugging Face checkpoints. Nested TRT-LLM
settings go in native YAML files, selectable with `config=[...]`. Model, GPU,
and precision support depend on the installed framework release. See the
[three-framework comparison](docs/trtllm-comparison.md) for a larger example.

### Apply SLOs and view results

Add constraints to either grid or Optuna commands:

```bash
--constraints "ttft:p95<500ms;itl:p99<50ms"
```

Supported statistics include mean, median, p95, and p99. For example,
`e2e_latency:p95<2s` constrains end-to-end latency.

```bash
llm-optimizer visualize --data-file vllm_grid.json --serve --port 8080
```

Open `http://localhost:8080/pareto_llm_dashboard.html`. To compare frameworks,
pass comma-separated files, such as `--data-file "vllm_grid.json,trtllm_grid.json"`.

### Estimate performance without a full benchmark

```bash
llm-optimizer estimate \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --gpu H100 --num-gpus 1 \
  --input-len 1024 --output-len 512
```

Use `llm-optimizer estimate --interactive` for guided setup.

## 2. Added Optuna workflow

### Tune one model and framework

Install the optional search dependency:

```bash
pip install -e '.[optuna]'
```

Use the same benchmark command and argument syntax, adding an Optuna budget:

```bash
llm-optimizer \
  --framework vllm \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --server-args "tensor_parallel_size=1;max_num_seqs=[16,32,64]" \
  --client-args "dataset_name=random;random_input_len=128;random_output_len=32;num_prompts=64;max_concurrency=[8,16,32]" \
  --search-strategy optuna \
  --optuna-trials 6 --optuna-seed 7 \
  --optuna-objective output_throughput \
  --output-json vllm_optuna.json
```

This uses seeded TPE to select configurations rather than exhaustively running
the grid. The same approach works with `sglang` and `trtllm` using their native
arguments. Measured tuning requires the same GPU/runtime setup as grid search.

| Control | Usage |
| --- | --- |
| Trial budget | `--optuna-trials 20` |
| Time limit (seconds) | `--optuna-timeout 3600` |
| Objective | Throughput metrics are maximized; latency metrics such as `p95_ttft_ms` are minimized. Default: `output_throughput`. |
| SLOs | Add `--constraints`; completed trials that miss SLOs are excluded from the feasible best. |
| Integer ranges | Use `--optuna-server-args "max_num_seqs=int(16,64,16)"` or `--optuna-client-args "max_concurrency=int(8,32,8)"`. |
| Resume | Repeat the original command with `--continue`. Keep the model, workload, search space, objective, and constraints unchanged. |

Lists and `range(...)` in ordinary argument options are categorical choices.
Optuna-only `int(low,high[,step])` ranges use integer sampling, include both
bounds, and require the step to land on the high bound. Do not define the same
parameter in both ordinary and Optuna-only argument options.

`--output-json` is required. Results include a JSON file, incremental JSONL
records, and a persistent `<output-name>.optuna.db` study. Use a fresh output
path when changing the experiment. One active worker per study is supported.
Use `--dry-run` to inspect the search plan without launching benchmarks.
See `llm-optimizer --help` for sampler, pruner, and storage controls, and the
[multi-objective guide](docs/multi_objective.md) for Pareto optimization.

### Run the experiment lab

The lab is a separate workflow for comparing samplers across models,
frameworks, phases, and objectives, then independently validating candidates.
The supplied configuration defaults to **CPU simulation**, so its results
are not measured GPU performance.

```bash
pip install -e '.[notebook]'

llm-optimizer experiment run \
  --config docs/optuna_experiment.json --run-dir experiment_results/my-run

llm-optimizer experiment validate \
  --config experiment_results/my-run/config.json --run-dir experiment_results/my-run

llm-optimizer experiment report \
  --config experiment_results/my-run/config.json --run-dir experiment_results/my-run
```

Edit a copy of the example config before starting a new run. Repeating `run`
resumes its SQLite studies; `validate` measures shortlisted candidates;
`report` rebuilds exports without launching benchmarks. The run directory stores
the frozen config, environment metadata, SQLite studies, logs, and `results.json`.

For measured GPU runs, configure `run_real_gpu`, pinned local model snapshots,
and verified runtime/model preflight settings as described in the
[experiment guide](docs/optuna_experiment.md). Use a new run directory when
changing configuration.

- [Launcher notebook](docs/optuna_experiment_cli.ipynb): configure, launch, and explore the lab.
- [Run metadata notebook](docs/optuna_run_metadata.ipynb): summarize saved studies and optional W&B data.
- [Trial stability notebook](docs/optuna_trial_stability.ipynb): inspect historical GPU repeatability without launching benchmarks.

## Remote execution and deployment tools

- **Nebius GPU jobs:** [controller setup, request schema, submission, and results](docs/nebius-remote.md). Supports grid and Optuna searches with one job per framework. Worker Dockerfiles are in [docker/](docker/); build and push images before submission. Install controller dependencies with `pip install -e '.[remote]'`. Configure the Nebius CLI and S3 credentials, load `.env`, and start with a dry-run request.
- **NVIDIA Dynamo:** [deployment topology and validation](docs/trtllm-dynamo.md). `dynamo-check` validates grids and generates manifests; `dynamo-optimize` performs offline AI Configurator sizing. Install with `pip install -e '.[ai-configurator]'` on Python 3.11–3.13.

## Development and upstream

```bash
pip install -e '.[dev]'
pytest
ruff format --check .
ruff check .
mypy src/
```

Based on [BentoML LLM Optimizer](https://github.com/bentoml/llm-optimizer),
with [vLLM](https://github.com/vllm-project/vllm),
[SGLang](https://github.com/sgl-project/sglang),
[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), and
[Optuna](https://github.com/optuna/optuna).

Custom HF ShareGPT workloads: see [dataset preparation and worker rollout](docs/hf-datasets.md) and `optimization_hf_dataset.ipynb`. Install optional CPU preparation dependencies with `pip install ".[dataset]"`.
