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
