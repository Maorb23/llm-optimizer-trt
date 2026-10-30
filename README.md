# LLM Optimizer — Bento Extend

Tune LLM serving configurations with Bayesian optimization instead of benchmarking
an entire parameter grid. This repository extends
[BentoML LLM Optimizer](https://github.com/bentoml/llm-optimizer) with Optuna-based
search, persistent experiment history, independent validation, execution monitoring,
and a native TensorRT-LLM adapter alongside vLLM and SGLang.

## Bayesian optimization

Each trial selects server and client settings, starts the inference engine,
measures a fixed workload, and feeds the objective back to Optuna. The default
seeded TPE sampler uses previous observations to guide later proposals. The CLI
also accepts configurable samplers, including Gaussian-process `GPSampler`;
the experiment lab compares TPE, multivariate TPE, GP, random, QMC, and grid arms.
Sampler availability and dependencies depend on the installed Optuna environment.

Maximize output throughput, minimize latency such as p95 time to first token
(TTFT), or search throughput/latency tradeoffs with multiple objectives. SLO
constraints identify feasible candidates; inspect feasibility as well as the
objective score. Multi-objective reports retain infeasible completed trials with
their feasibility flag: see the [multi-objective guide](docs/multi_objective.md).
Grid search remains the CLI default; select `--search-strategy optuna` explicitly.

### Effectiveness and how to assess it

The goal is to find a strong feasible configuration with fewer expensive GPU
measurements than exhaustive search. A trial budget caps search work; a timeout
bounds the search loop but is not a hard deadline for an already running benchmark.
Adaptive sampling does not guarantee a global optimum or a win over random search
on every workload, especially with small budgets or noisy measurements.

The experiment lab supports sampler comparisons across seeds and independent
validation of shortlisted candidates using reserved request seeds and repeated
measurements. Compare best feasible performance against benchmark attempts and
total allocation wall time, including startup, failures, cleanup, and rest. Keep
model revisions, workload, search space, hardware, and runtime fixed. GPU-seconds
in lab accounting are allocation-time proxies, not measured GPU utilization.

Saved GPU trial artifacts are present under `experiment_results/`, but this README
does not claim a repository-wide numerical speedup from them. Use the
[trial stability notebook](docs/optuna_trial_stability.ipynb) to assess exact-repeat
variability before interpreting small gains. The supplied experiment configuration
defaults to CPU simulation; simulated scores are not measured GPU performance.

### Install

From the repository root, with Python 3.9 or newer:

```bash
pip install -e '.[optuna]'
```

Measured benchmarks require a compatible NVIDIA GPU host and the selected serving
framework and benchmark dependencies in the same environment. This package does
not install vLLM, SGLang, or TensorRT-LLM. For gated models, obtain Hugging Face
access and export `HF_TOKEN` first.

### Tune one model and framework

Use the same benchmark command and argument syntax, adding an Optuna budget:

```bash
llm-optimizer \
  --framework vllm \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --server-args "tensor_parallel_size=1;max_num_seqs=[16,32,64,96,128]" \
  --client-args "dataset_name=random;random_input_len=128;random_output_len=32;num_prompts=64;max_concurrency=[8,16,32,48,64]" \
  --search-strategy optuna \
  --optuna-trials 20 --optuna-seed 7 \
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

**Start with 20 trials per study.** The default CLI TPE sampler uses random
sampling during its initial exploration phase (`n_startup_trials=10`). Choose
more than 10 trials to leave room for guided optimization after that phase;
failed trials and duplicate proposals can reduce the useful observations collected.
The example above offers 25 configurations, leaving room for a 20-trial search.
A search space with 10 or fewer distinct configurations is better suited to a
full grid comparison than a default TPE study.

The startup count is configurable through `--optuna-sampler-kwargs`, and other
samplers and experiment-lab settings may use different counts. Twenty trials is
a starting recommendation, not a guarantee of convergence or improved performance.
See [Optuna's TPE documentation](https://optuna.readthedocs.io/en/v4.9.0/reference/samplers/generated/optuna.samplers.TPESampler.html)
for startup sampling behavior.

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

## Stored results, caches, and resume

Optimization history and workload caches serve different purposes:

| Storage | Contents and reuse |
| --- | --- |
| CLI output JSON and incremental JSONL | Benchmark metrics, configurations, and search metadata; JSONL preserves successful trial records incrementally. |
| `<output-name>.optuna.db` | Default persistent CLI study, including trial state and observations. Resume with the original command plus `--continue`. |
| Lab run directory | Frozen `config.json`, `environment.json`, `studies.sqlite3`, logs, `results.json`, and measured-run workload/config artifacts. Repeating `experiment run` resumes studies; `report` rebuilds exports without benchmarking. |
| Prepared workload manifest | The shared lab adapter loads/tokenizes the workload once per adapter instance and reuses exact requests across trials. Measured runs retain request manifests for auditing. |
| Dataset download cache | The download helper reuses a valid JSON file; its default location is `/tmp/<URL filename>`. Temporary storage may be cleared by the host. |
| Generated shared-prefix cache | Generated prompts are stored under `~/.cache/sglang/benchmark/`, keyed by generation settings and tokenizer class. Loading is bypassed when partial system-prompt randomization is enabled. |
| In-process vocabulary cache | Tokenizer vocabulary lists are memoized for random prompt generation, avoiding repeated vocabulary conversion. |

Use a new output/run directory when changing the experiment contract. CLI resume
checks the saved signature; lab commands require the frozen configuration to match.
One active worker per CLI study is supported. Resolve abandoned lab `RUNNING`
trials only after verifying that their old worker has stopped. The shared-prefix
cache key is not a complete model/tokenizer revision identity, so isolate or
regenerate that cache when changing tokenizer revisions.

These mechanisms do not provide persistent inference-engine compilation caches
or reuse a running server across trials. Framework KV/prefix-cache tuning affects
serving behavior and is separate from storing optimizer history and prepared data.

## Startup monitoring and execution efficiency

Real benchmark attempts preserve `server.log` and `execution.json` under
`<result-stem>-attempts/<attempt-id>/`; calls without an output file use
`benchmark-attempts/`. Logs survive muted console output, startup or benchmark
failure, pruning, interruption, and cleanup failure. Successful rows link execution
metadata; exceptional attempts retain sidecars instead of incomplete metric rows.

Phase timings separate launch-to-ready, client preparation, warmup, request
execution, cleanup, and lab rest. Runner and allocation wall times expose overhead
outside the benchmark metric window. Log-derived loading, compilation, autotuning,
and graph-capture observations retain source evidence; missing observations are
unknown, and overlapping or per-rank durations must not be added as wall time.
Read the attempt sidecar for finalized rest/allocation accounting.

Implemented changes reduce avoidable work and improve diagnosis:

- **Workload and vocabulary reuse:** prepare the shared lab workload once and
  cache vocabulary conversion; dataset and shared-prefix caches avoid supported
  repeated download/generation work.
- **Duplicate suppression:** avoid rerunning completed configurations in a study.
  GP guards also reject exact previously failed or pruned configurations before
  server launch, retaining an audit trail and stopping after repeated blocked
  proposals. See [GP proposal rejection](docs/gp-pruning.md).
- **Live CLI pruning:** optional scalar, unconstrained pruning uses completed-request
  checkpoints and cancels/drains outstanding work before cleanup. The default is
  `NopPruner`; multi-objective and constrained CLI studies reject active pruning.
  The extracted experiment lab separately records real-GPU pruning arms as skipped.
- **Startup and lifecycle checks:** readiness polling detects an exited server;
  TRT-LLM validates listener ownership and port release to prevent a subsequent
  trial from benchmarking a leftover process.

Monitoring makes startup costs visible; it does not itself shorten engine loading.
Readiness still uses retry counts and five-second delays by default. Persistent
engine cache roots, faster loaders, shorter polling, and identical-server session
reuse remain proposed work in the [startup optimization plan](docs/startup-optimization-plan.md).
No blanket startup speedup is claimed. See the
[execution timing guide](docs/optuna_experiment.md#execution-timing-and-logs) for
phase definitions and artifact details.

## Original BentoML workflow and the TensorRT-LLM extension

The original workflow remains available: enumerate a configuration grid, launch
and benchmark each setting, apply SLOs, and visualize results. Our native
TensorRT-LLM adapter extends this workflow using `trtllm-serve` with Hugging Face
checkpoints and native YAML configuration, sharing benchmark metrics with vLLM
and SGLang. GPU, model, and precision support depend on the installed runtime.

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

## Remote execution and deployment tools

- **Nebius GPU jobs:** [controller setup, request schema, submission, and results](docs/nebius-remote.md). Supports grid and Optuna searches with one job per framework. Worker Dockerfiles are in [docker/](docker/); build and push images before submission. Install controller dependencies with `pip install -e '.[remote]'`. Configure the Nebius CLI and S3 credentials, load `.env`, and start with a dry-run request.
## NVIDIA Dynamo AI Configurator

The repository also includes a [Dynamo integration](docs/trtllm-dynamo.md).
`llm-optimizer dynamo-check` validates configuration grids and generates deployment
manifests; `llm-optimizer dynamo-optimize` performs GPU-free offline AI Configurator
sizing and produces ranked deployment artifacts. These estimates complement measured
engine tuning; they are not Bayesian benchmark results. Install the optional
integration with `pip install -e '.[ai-configurator]'` on Python 3.11–3.13.
Applying generated manifests is a separate Kubernetes operation.

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
