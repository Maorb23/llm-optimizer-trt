# Phase 1 differences from the v2 lab

Reference: [optuna_inference_optimization_lab_v2.ipynb](optuna_inference_optimization_lab_v2.ipynb).
Current launcher: [phase1_experiments.ipynb](phase1_experiments.ipynb).

## Restored controls

- Scalar sampler selection: `gp`, `tpe`, and `tpe_multivariate`, all selected by default.
  A group can override this with `sampler_names=["gp"]`. Missing PyTorch produces
  an explicit GP skip warning and `skipped-arms.json`; no sampler is substituted.
- Twenty trial attempts per sampler, seed, framework, concurrency, objective,
  and pruner. Seeds remain `[11, 29, 47]`; three samplers therefore mean 180
  attempts for one framework/concurrency/objective/pruner, not 20 total.
- Search requests: concurrency 1 -> 32, 8 -> 62, 16 -> 125, 32 -> 250,
  128 -> 1000. Both phases use this map. Validation retains its independent
  1000-request budget and five repeats.
- The v2 pruning observation floor of 400/1000 is scaled by request count:
  12, 24, 50, 100, 400. This affects checkpoint-based pruning only.
- `apply_chat_template=True`, explicit sampler startup/settings, W&B controls
  and notebook login. Existing shared GP failure-policy logic remains in use.
- Every group reads the current controls, avoiding a stale cached runner.

## Restored optional experiments

- [x] Independent Phase B Pareto E2E/TTFT (c=1) and throughput/E2E
  (loaded concurrencies) cells, using `PARETO_SAMPLERS` and Matplotlib Pareto plots.
- [x] Editable TTFT/E2E/failure SLOs; explicit constrained scalar and Pareto cells.
  GP, TPE, multivariate TPE and NSGA-II support constraints; random/grid/QMC
  constraint combinations fail before execution. Scalar sampler defaults remain unchanged.
- [x] Nop, median, percentile, successive halving (`halving`), Hyperband and
  calibrated threshold pruning. Pruning rejects Pareto and constrained searches.
- [x] Task-specific calibration uses the search workload, request manifest and
  scaled checkpoint schedule. Reports include rank correlations, top-decile
  false-prune rate and estimated time savings from a first-checkpoint median
  replay. These are pilot diagnostics, not measured savings for every pruner.
  Reports identify synthetic versus measured data. Threshold comparisons reuse
  one persisted calibration per task in the frozen campaign, across sampler/seed
  arms and resumes; pilot samples are additional to the equal search budgets.
- [x] Explicit ordinary/speculative HF setup via `HF_SETUP` and the opt-in
  `setup_pinned_models()` call. Uses the repository helper, immutable resolved
  target/draft revisions and installed runtime capability checks. SGLang supports
  DFLASH/DSPARK; vLLM's legacy `enable_dflash=True` selects DSpark; TRT supports
  Eagle3/DFlash/DSpark through its existing version-gated helper. No new native
  framework flags are introduced.
- [x] Resolved profiles and speculative settings enter campaign hashes, frozen
  JSON, study contracts, exports and W&B config. Changing HF setup after setup
  raises an explicit error until profiles are cleared and setup reruns. Expanded-choice edits after setup also
  require fresh setup, preventing stale capability restrictions.

## Remaining differences to choose explicitly

| v2 capability | Phase 1 status |
|---|---|
| Untuned Transformers and suggested vLLM baselines | No launcher cells. Phase A is an optimization search, not an untuned baseline. |
| Separate phase models | Phase 1 currently uses the same Qwen3.5-4B target for A/B; v2 defines 0.8B/4B phase targets. |
| Grid, Random, Sobol and NSGA-II comparison cells | Not selected for the default scalar queue. The shared samplers remain available; the requested default is GP/TPE/multivariate TPE. |
| Validation recommendations, full JSON exports, coverage and correctness/reporting cells | SQLite retains study history; Phase 1 has per-study JSON, scalar plots and a manual validation cell, not the full v2 report pipeline. Validation is not automatically part of the batch queue. |
| Quick-demo seed/trial switches | Not exposed: all selected seeds run with the configured 20-attempt budget. `--smoke` is a separate CPU-only wiring check. |

## Operational limits

Only the batch entry point enforces the persistent campaign deadline and group
watchdog; direct notebook experiment cells do not. Timeouts cover whole groups,
whose work increases with the number of selected samplers/frameworks. Deadline
expiry currently exits successfully even if tasks remain. W&B study invocations
create separate tracking runs on restart. GPU/backend/precision verification is
still required; CPU checks do not establish compatibility or quality. Automatic
backup, alerting and sustained-load validation are not implemented by this launcher.


## CPU validation

The Phase 1, experiment workflow, HF setup, speculative serialization and live
pruning CPU suites cover propagation, identities, equal budgets, resume/locking,
callback calibration and notebook/script alignment. HF/runtime calls and W&B
are mocked. In the local test environment, CLI-import tests need a process-local
`readline.read_history_file` stub because its existing history file raises
`OSError`; this does not change application code.

The separate legacy check
`test_v2_search_functions_and_controls_remain_identical` currently fails on the
AST comparison between the unchanged prune and v2 notebooks. That legacy
notebook mismatch is outside the Phase 1 changes. GPU validation remains pending.
