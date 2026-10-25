# Optuna experiment commands

The experiment commands expose the workflow from
`docs/optuna_inference_optimization_lab.ipynb`. That notebook remains unchanged
as the behavior reference. Open `docs/optuna_experiment_cli.ipynb` to edit run
settings, launch shell commands, and explore saved tables and plots.

Install the package with `pip install -e ".[optuna]"`, or use
`pip install -e ".[notebook]"` for the launcher notebook's plotting dependencies.
The notebook extra pins Optuna 4.9.0, the reference version. Expanded grid search
uses that version's private grid sampler fields. W&B is optional:
`pip install -e ".[wandb]"`.

```bash
llm-optimizer experiment run --config docs/optuna_experiment.json --run-dir experiment_results/my-run
llm-optimizer experiment validate --config experiment_results/my-run/config.json --run-dir experiment_results/my-run
llm-optimizer experiment report --config experiment_results/my-run/config.json --run-dir experiment_results/my-run
```

`run` searches configured tasks and exports their trials. Repeating it resumes
the same named studies in SQLite, counting existing attempts toward the trial
budget. The initial queued candidate consumes a trial only once executed, so even
a one-trial run performs a measurement. The reference notebook subtracts that
candidate before execution; the extraction corrects this budget accounting. Interrupted studies containing RUNNING trials require checking
that the old worker has stopped and resolving those trials explicitly before
resuming. No abandoned trial is silently changed.

`validate` independently measures shortlisted candidates using reserved request
and simulation noise seeds and the configured validation request budget.
It selects matching scalar winners or a Pareto front from replicate means,
excludes incomplete candidates, and applies SLO eligibility when requested.
`report` rebuilds the export from storage and saved validation without launching
benchmarks. Validation and report require a previously initialized run directory.

## Configuration

### Execution timing and logs

Real benchmark attempts now save `server.log` and `execution.json` under
`<result-stem>-attempts/<attempt-id>/`, beside their result file. Calls without
an output file use `benchmark-attempts/`. Logs are retained even when server
console output is muted, and on startup failure, pruning, benchmark failure,
cleanup failure or interruption. An attempt killed before finalization remains
`running`; this means unfinished/unknown, not a successful measurement.

Successful result rows include optional `metadata.execution`. Exceptional
attempts use sidecars rather than incomplete candidate rows. Historical results
without execution metadata remain supported. The shared lab also includes
execution metadata in measured `Measurement` records and attaches it to
exceptions. Optuna trial attributes link to this evidence where available.

`timings_s` separates `launch_to_ready`, `client_preparation`, `warmup`,
`benchmark` (request execution and cancellation/drain), `cleanup`, and `rest`.
`runner_wall` covers the runner before intertrial rest; `orchestration` is its
remaining wall time outside the instrumented phases. The lab additionally
records `allocation_wall` and its GPU-seconds proxy, including its rest. Lab
workload preparation before `measure()` is reported separately as
`workload_preparation_seconds`; it is one preparation per adapter instance,
not a new preparation on every trial (deduplicate by `workload_preparation_id`).
Existing wall/GPU proxies retain their
definitions. The historical benchmark metric window is unchanged and reported
as `metric_benchmark_duration_s`; it can include post-request hooks beyond the
new request-execution timing.

Missing timings are unknown, not zero. `phase_status` and `incomplete_phases`
identify partial intervals. Explicit durations found in engine logs appear in
`startup_observations` with source lines and parser identity. Parsing is best
effort; repeated ranks and overlapping loading/compilation/autotuning/capture
observations are not additive wall phases. Never derive their proportions from
total trial duration. Full logs remain the evidence when an engine's wording
does not match the parser.

Intertrial rest is finalized in the sidecar after the successful metric row is
written, so read the linked `attempt_artifact` for final rest/allocation timing.
Readiness polling, engine ownership, warmup, precision, graph capture,
autotuning and rest policy have not changed. No persistent-cache, loader or
server-reuse optimization is enabled by this instrumentation.

`docs/optuna_inference_optimization_lab_v2.ipynb` uses a thin subclass of the
shared `experiment.lab.RepositoryBenchmark`, preserving the notebook's own
workload and native configuration serialization. Its source-patching diagnostic
cells have been retired. The original pruning notebook remains unchanged.

### Experiment controls

`docs/optuna_experiment.json` contains the reference models, workload, search
budgets, and SLOs. Its execution defaults are simulation and disabled W&B so the
example can run on a CPU. Set these controls before launching a new directory:

- `phases`, `phase_models`, `frameworks`, `real_frameworks`, `concurrencies`:
  task selection. Concurrency 1 uses the single-request regime; any larger
  positive integer uses the numerical search space keyed by `128`.
- `base_workload`, `phase_requests`, `validation_requests`, `validation_repeats`:
  fixed workload and search/validation request budgets. Use `"inf"` for unlimited
  request arrival rate in JSON.
- `search_trials`, `demo_trials`, `quick_demo`, `seeds`: trial budgets and sampler
  seeds. Quick demo uses the demo budget and the first seed.
- `sampler_names`, `pareto_samplers`, `sampler_settings`: sampler arms, startup
  budgets, TPE candidate count, and NSGA-II population size. GP is clearly
  recorded as skipped if Torch is unavailable.
- `objectives`: objectives by concurrency regime. Regime `1` supports `e2e`,
  `ttft`, and `pareto_latency`; regime `128` supports `e2e`, `throughput`, and
  `pareto_throughput`. The Pareto pairs are E2E/TTFT latency and
  throughput/E2E latency, respectively.
- `ttft_limits_ms`, `e2e_limits_ms`, `failure_rate_limit`, `constrained`,
  `validation_require_slo`, `gpu_budget`: feasibility and per-study allocation
  time budgets. GPU seconds retain the reference's allocation wall-time proxy;
  they are not GPU utilization measurements.
- `simulation_seed`, `validation_seed`, `validation_noise_seed`,
  `calibration_sampler_seed`, `calibration_noise_seed`: independent noise,
  request, and pruning calibration seeds.
- `pruning`, `pruning_samples`, `pruning_trials`: simulated scalar checkpoint
  calibration and pruning arms. Real GPU pruning is recorded in `skipped_arms`
  because the repository adapter has no continuous checkpoints or clean
  cancellation interface.
- `run_real_gpu`, `model_snapshots`, `real_preflight`, `gpu_count`,
  `rest_seconds`, `fixed_server`: measured adapter controls. Real runs require
  existing pinned local snapshots and verified framework, CUDA, driver, and
  model revisions plus native config verification. Phase B also requires
  numerical quality verification and verified expanded choices. Replace the
  example preflight values with actual measurements before enabling this mode.
- `wandb_enabled`, `wandb_project`, `wandb_entity`, `wandb_mode`, `wandb_group`:
  optional telemetry. Authenticate separately with `wandb login` for online
  runs. Telemetry failures retain the reference workflow's isolation from
  optimization and cleanup behavior.

Unknown settings and invalid values are rejected. Omitted top-level controls
use package defaults; nested controls must contain their required keys.
A run directory's frozen configuration must match subsequent commands. Use a
new directory to change settings.

## Artifacts

Each run directory contains:

- `config.json`: the resolved frozen configuration.
- `environment.json`: Python, platform, installed package versions, and source
  revision information captured when the run is initialized.
- `studies.sqlite3`: persistent Optuna trials and study contracts.
- `experiment.log`: command and Optuna progress logs.
- `results.json`: controls, skipped arms, calibration, contracts, summaries,
  trial parameters and distributions, measurements, accounting, intermediate
  checkpoint values, validation replicates, and recommendations.

Measured runs also preserve request manifests, benchmark JSONL records, and
engine config artifacts under the same directory. Nonfinite request rate
values are serialized as strings. JSON exports are written through a temporary
file and replaced after successful serialization. SQLite remains the source
for rebuilding trial reports.

The existing `llm-optimizer benchmark --search-strategy optuna` interface retains
its own controls and behavior.

### Draft models

Python experiments share model setup with the v2 notebook through
`Lab.configure_hf_models()`. With a real GPU lab and a Phase B preflight configured:

```python
# vLLM: the notebook's legacy toggle selects native DSpark.
lab.configure_hf_models(
    "google/gemma-4-12B-it", framework="vllm", enable_dflash=True,
    draft_model_id="deepseek-ai/dspark_gemma4_12b_block7",
    num_speculative_tokens=7,
)

# SGLang: the notebook's DSPARK block-7 example, with an 8-token verify window.
lab.configure_hf_models(
    "Qwen/Qwen3-4B", framework="sglang", speculative_algorithm="DSPARK",
    draft_model_id="deepseek-ai/dspark_qwen3_4b_block7", block_size=7,
)

# TensorRT-LLM: requires the installed 1.3.0rc26 runtime and a compatible draft.
lab.configure_hf_models(
    "meta-llama/Llama-3.1-8B-Instruct", framework="trtllm",
    speculative_algorithm="Eagle3",
    draft_model_id="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B", max_draft_len=3,
)
```

Run these examples separately for the selected backend. SGLang DFlash remains
available through `enable_dflash=True` (or `speculative_algorithm="DFLASH"`).
TensorRT also accepts explicit `DFlash` and `DSpark` algorithms. Setup pins model
revisions, writes `hf-<framework>-models.json`, and retains existing verification
declarations; GPU loading, quality, and performance still require validation.
Call again without draft options to restore target-only Phase B settings.

Both search and validation merge `real_preflight[phase][framework].fixed_server`
over the backend-wide `fixed_server[framework]` defaults. The CLI accepts the same
resolved settings in JSON, alongside pinned local `model_snapshots`, target/draft
revisions, and `speculative_decoding` metadata in `real_preflight`. It does not
automatically download drafts or invoke the Python model-setup helper. Keep the
resolved draft configuration in a new run's JSON before initializing that run;
existing run configurations remain frozen.

## Historical GPU stability checks

Open [optuna_trial_stability.ipynb](optuna_trial_stability.ipynb) to analyze saved
Phase A and Phase B GPU trials separately. It reads measured JSONL records and
joins SQLite trial identity without counting the same measurement twice.
Exact-repeat comparisons hold native engine settings, request fingerprints,
workload, model, and recorded environment fixed. TRT-LLM YAML contents are
resolved from local artifacts, including logs moved from another machine.

The notebook reports repeat counts, min/max ranges, sample coefficient of
variation, and deviations from each configuration's median. Nearby numeric
settings have a separate table with the changed parameters shown; those
differences include tuning effects and do not establish measurement noise.
Failed/invalid measurements and incomplete environment metadata have explicit
audit tables. No GPU benchmarks are launched.

The settings cell controls run directories, minimum repeats, alert thresholds,
and numeric tolerances. By default it discovers saved measured runs under
`experiment_results/`. Tables, figures, and comparison settings are exported
to `experiment_results/phase_stability_analysis/`. Matching placeholder metadata
or a GPU model name cannot establish an unchanged physical GPU or runtime.
