# Startup optimization implementation plan

Deliver in three stages: measurement/readiness, persistent caches/local loading, then optional identical-configuration reuse. This document proposes implementation; it does not change runtime behavior.

## Current boundaries

- `src/llm_optimizer/cli.py::run_single_configuration()` owns launch, benchmarking, result serialization, and cleanup. Results are currently written before its `finally` cleanup; startup failures and pruning do not produce the same successful-result record.
- `server_utils.py::start_server()` owns readiness and launch-failure cleanup. Muting currently discards server output. `wait_for_server()` uses retry counts and five-second sleeps. `_TRTLifecycle` tracks workers and validates listener ownership/port release.
- `bench_client.py::run_benchmark()` prepares tokenizer/data; `benchmark()` owns warmup and measured requests. Instrument those actual boundaries rather than timing the whole call as benchmark duration.
- `experiment/lab.py::RepositoryBenchmark.measure()` wraps the runner, then sleeps. Its allocation wall proxy includes orchestration, startup, benchmark, cleanup, and successful-trial rest. Constructor workload preparation happens outside that timer.
- `docs/optuna_inference_optimization_lab_prune.ipynb` has a separate `RepositoryBenchmark` in code cell 13 (zero-based index at time of writing), plus setup, preflight, search, validation, and artifact cells. It must stop owning a second lifecycle implementation.

## Where each responsibility belongs

All paths below are relative to `src/llm_optimizer/` unless otherwise noted. New modules are proposed names.

| File | Responsibility |
| --- | --- |
| `startup_policy.py` (new) | Validated operational policy: readiness deadline/poll interval, persistent cache roots, local snapshot root, loader selection, reuse mode, rest policy. Defaults preserve fresh engines and ten-second rest. No Optuna distributions. |
| `trial_telemetry.py` (new) | Monotonic phase recorder, optional metadata schema, attempt/session identifiers, durable attempt artifacts, finalization on every exit path. No engine ownership. |
| `startup_logs.py` (new) | Version-aware parsing of saved framework logs for loading, compilation/autotuning, capture, and explicit cache-hit evidence. Retain source lines, rank, units, and parser version. |
| `server_utils.py` | Process launch, readiness, output capture/optional console tee, process tracking and verified cleanup. Retain TRT lifecycle protections. |
| `server_session.py` (new, stage 3) | Context-managed engine ownership, complete identity, acquire/release/reset/invalidate/close. Calls existing process utilities. |
| `cache_setup.py` (new) | Resolve persistent cache environment and pinned-backend capabilities; produce cache provenance. Never delete caches as trial cleanup. |
| `model_artifacts.py` (new) | Stage immutable target/draft snapshots locally; verify revisions/tensor provenance; manage topology-specific loader artifacts. |
| `bench_client.py` | Exact preparation/warmup/measured-request timings; bounded tokenizer/config and prepared-workload caches; workload identity and reset hooks. |
| `cli.py` | Compose policies, telemetry, session acquisition, benchmark and persistence; expose CLI options and pass policy through single/grid/Optuna callers. Preserve existing return contract. |
| `predefined/{vllm,sglang,trtllm}.py` | Register supported native loading/configuration arguments after checking pinned versions. Existing TRT argument classification remains metadata only. |
| `experiment/config.py` | Backward-compatible nested startup-policy defaults and validation; reject unsupported combinations early. |
| `experiment/hf_models.py` | Call shared staging for target and draft; keep model/profile setup and resolved revision metadata here. |
| `experiment/lab.py` | Adapt lab contracts to shared runner; own measurement-level rest and lab accounting; add optional telemetry to `Measurement` and trial attributes. |
| `experiment/workflow.py` | Initialize/close experiment resources, record environment, one-time preparation costs and final session shutdown; export attempt references and phase summaries. |
| `wandb_tracking.py`, `result_utils.py` | Export optional phase/cost fields; tolerate absent fields in historical results. No lifecycle decisions. |
| pruning notebook | Controls, preflight, shared adapter construction, plots and explanation. No process management, cache resolution, readiness loops, or separate reuse rules. |

Do not put this in `performance.py`: that module estimates model performance rather than managing measured engine execution. Keep startup policy separate from sampled server/client parameters and record it alongside the experiment contract. Loader changes remain explicit effective server arguments even when selected through an operational policy.

## Stage 1 — measure and preserve evidence

1. Introduce a shared phase recorder using `perf_counter()`/monotonic time. Persist elapsed seconds and UTC timestamps for correlation; do not subtract monotonic timestamps across processes or worker restarts.
2. Instrument launch immediately before `Popen`, health readiness after successful owned-listener validation, client preparation, warmup, measured requests, cleanup, and rest. Keep existing benchmark timing/metric definitions unchanged. Include miscellaneous orchestration and end-to-end allocation wall time so phase totals can be reconciled.
3. Treat log-derived loading/compile/autotune/capture durations as startup annotations, potentially overlapping and rank-specific. They are not extra additive wall phases. Missing observations are `null`/absent, never zero. Do not derive proportions from total trial wall time or sum rank durations as elapsed startup.
4. Allocate an attempt ID and artifact directory before launch. Include study/trial identity when available, plus a unique attempt ID for retries, validation and non-Optuna calls. Thread these identifiers from the lab; its current runner invocation does not supply a trial number.
5. Capture stdout/stderr directly to durable files or continuously drain a tee; never leave an unread pipe. `mute_server` controls console output only. Save partial logs on startup failure, pruning, benchmark failure, cleanup failure and interruption. Flush/close after process cleanup.
6. Replace retry-count timing with an explicit monotonic timeout and 0.25–0.5 second poll interval. Bound each HTTP timeout and sleep by remaining deadline; check process exit on every iteration. Preserve TRT prelaunch port checks and owned-listener verification. Provide compatibility for existing callers using `max_retries`/`delay` while moving internal callers to explicit timeout seconds.
7. Finalize telemetry after cleanup, not before it. Keep successful result rows backward compatible; use a separate attempt sidecar/event record for failures/pruned/interrupted runs rather than inserting metric-less rows into files whose consumers expect `results`. Save an initial record before launch and atomically update on completion. A hard kill may leave an unfinished record, which recovery marks interrupted/unknown.
8. Let the lab append rest duration and final allocation accounting to the same attempt artifact after the runner returns. Record policy-driven skipped rest as skipped. Preserve current successful-trial rest behavior initially. Include constructor/client preparation and setup in experiment accounting even when outside per-trial timing.

Suggested additive metadata:

```json
{
  "execution": {
    "schema_version": 1,
    "attempt_id": "…",
    "session_id": "…",
    "status": "complete",
    "server_reused": false,
    "timings_s": {
      "launch_to_ready": 42.1,
      "client_preparation": 0.8,
      "warmup": 2.0,
      "benchmark": 15.0,
      "cleanup": 1.2,
      "rest": 10.0
    },
    "startup_observations": [],
    "server_log": "attempts/…/server.log",
    "attempt_artifact": "attempts/…/execution.json"
  }
}
```

Status values distinguish complete, failed, pruned, interrupted and cleanup_failed. Preserve the original exception plus cleanup errors; a cleanup verification failure must block subsequent execution, including when pruning was the initiating event. Persist partial request counts and elapsed cost on exceptional exits. Existing `MEASURED_ALLOCATION_WALL_PROXY` fields retain their meaning; new fields add measured decomposition.

Stage gate: collect representative baseline runs and identify observed costs before implementing lifecycle reuse.

## Stage 2 — stable caches, local weights, smaller delays

1. Add an explicit cache root on node-local fast storage, outside run/output directories. Validate writability and record resolved paths. Preserve it across trials and worker restarts. Use stable framework cache locations and let native cache validation handle configuration compatibility; do not create a new directory per trial.
2. Record framework/backend, Torch, CUDA runtime and driver versions, GPU architecture, target/draft/tokenizer revisions, effective arguments and relevant environment alongside cache policy. Record whether cache reuse was observed, missed or unknown. Directory existence and a faster second startup are not proof of a hit.
3. Framework implementation:

| Framework | Cache and loading work |
| --- | --- |
| vLLM | Persist `VLLM_CACHE_ROOT` plus dependent caches used by the pinned stack. Verify actual disk compilation reuse across representative sampled configurations. Compare supported loaders; optionally generate tensor-parallel sharded checkpoints per compatible topology and preserve original tensor values/dtype. |
| SGLang | Discover/persist installed Triton and FlashInfer cache mechanisms; persist Inductor cache if the trial already enables compilation. Do not enable compilation as a startup shortcut. Compare supported fast loaders with the existing loader on identical tensors/dtype. |
| TRT-LLM | Inspect installed PyTorch backend cache/autotuning and loading support through native configuration. Keep AutoDeploy pipeline caching scoped to AutoDeploy. Do not add an automatic `trtllm-build` stage. |

Use the official [vLLM compilation](https://docs.vllm.ai/en/stable/design/torch_compile/) and [sharded-state](https://docs.vllm.ai/en/stable/examples/features/sharded_state/) references, [SGLang arguments](https://docs.sglang.io/docs/advanced_features/server_arguments) and [loading](https://docs.sglang.io/docs/advanced_features/model_loading) references, and [TRT TorchLlmArgs](https://nvidia.github.io/TensorRT-LLM/llm-api/reference/TorchLlmArgs.html) and [AutoDeploy cache design](https://nvidia.github.io/TensorRT-LLM/features/auto_deploy/pipeline_cache_design.html). These are discovery references; the pinned installation decides which flags and mechanisms are available.

4. Stage both target and draft snapshots to local NVMe before launching engines. Resolve immutable revisions, verify complete staging, use locking and atomic publication for shared workers, and record source-to-local mapping. Preserve OS page cache between trials. Shared helpers must work for CLI and lab; `hf_models.py` is only the lab entry point.
5. Keep original snapshots intact. Derived sharded artifacts have manifests covering source revisions, tensor identity/dtype, producing framework/version and topology. Explicitly reject incompatible topology. Validate outputs and steady-state metrics before adopting a loader.
6. Cache tokenizer/config objects in a bounded process-local cache keyed by resolved revision, tokenizer options, trust/remote-code settings and relevant library versions. Audit mutation before sharing objects. Cache prepared workloads by complete generation identity: tokenizer, dataset content/version, sampling seed, prompt counts/lengths, chat template, prefix/multimodal settings and all generation inputs. Keep exact request hashes; do not accidentally cache client concurrency or routing settings into a reusable manifest.
7. Evaluate optional idle/thermal rest only after phase timing exists. Default remains ten seconds. Define utilization, in-flight request, temperature, timeout and unavailable-telemetry behavior; validate equivalence against fixed rest. Account for this as intertrial overhead, not initialization savings.
8. Track staging, artifact generation and optional prewarming once per experiment, including actual GPU allocation duration and GPU count. Include failed preparation and final shutdown. Do not double-count startup observations or shared session time. Compilation reuse does not imply graph captures survive a fresh process; retain existing graph capture behavior.

Stage gate: cold-cache and naturally warm-cache runs show measured loading/cache benefits with unchanged tensors, precision, warmup, graph settings and autotuning. Include preparation in total experiment cost.

## Stage 3 — optional identical-configuration reuse

Introduce an optional caller-owned `ServerSessionManager` argument to `run_single_configuration()`. With no manager, use a temporary session and close it on every call, preserving the default. The experiment/workflow owns a supplied manager and closes it in `finally`; notebook users get the same explicit context-manager/close boundary. Start with one active session per GPU allocation, not a pool of resident engines.

Build identity after resolving all defaults, native arguments, cache policy and model paths. Include target/draft/tokenizer revisions, all server arguments, framework/backend version, GPU assignment and parallel topology, relevant inherited/overridden environment, executable/command, and hashes of referenced TRT YAML/config file contents and nested referenced configuration. Resolve opaque configuration conservatively: if identity cannot be established, restart. Client-only changes may reuse; any server identity change restarts. Do not use `classify_server_args()` to authorize live changes.

Before reusing: verify process health and listener ownership, drain requests, clear/reset benchmark-relevant request/prefix/KV state using a pinned-backend adapter, verify reset completion, then run equivalent warmup. If reset is unsupported or unverifiable, restart. A flush request returning success is insufficient without verified semantics for that backend/version. Record reset time, reuse decision and restart reason. Disable reuse per unsupported backend rather than guessing.

Invalidate sessions on benchmark failure, pruning or interruption initially; always retain TRT worker cleanup and port-release verification. Close all sessions on experiment exit and provide a notebook shutdown control for abandoned interactive work. Keep final validation able to force fresh engines, and compare fresh/reused repeats to detect bias.

Do not reorder Optuna trials to increase reuse. Preserve ask/evaluate/tell and existing checkpoint/pruning semantics. Any savings apply only when naturally consecutive evaluations have identical server identity. Account shared launch and final cleanup once at experiment level; individual reused attempts report no new launch rather than inheriting the first attempt's startup cost.

## Notebook migration

1. In setup controls, add a startup-policy object separate from `SEARCH_SPACES`, sampled configuration dictionaries and pruning settings. Display effective policy/preflight results before real GPU execution.
2. Replace the lifecycle implementation in cell 13 with a thin adapter around `experiment.lab.RepositoryBenchmark`. Use a public shared factory accepting the notebook context while retaining notebook-specific configuration serializers and study functions. Do not silently replace custom notebook search behavior with a different lab search.
3. Import shared measurement/error definitions where needed so pruning and cleanup exception handling agree. Preserve all request manifests, objective directions, budgets, seeds and study ordering.
4. Route target/draft setup through shared staging. Remove executable source-patching cells that rewrite `bench_client.py` (currently cells 21–23); keep diagnostics read-only.
5. Update artifact/export and plotting cells to join attempt telemetry by attempt ID. Show startup/preparation/warmup/benchmark/cleanup/rest separately; show unobserved engine internals as unknown. Keep historical wall-proxy plots readable.
6. Add explicit session closure and fresh-engine validation controls. Do not run the notebook's real GPU search, study-state mutation or optional baseline cells during automated validation.

## Validation and delivery

Extend existing `tests/test_trtllm_lifecycle.py`, `test_benchmark_client.py`, `test_optuna_run_metadata.py`, `test_live_pruning.py`, `test_experiment.py`, `test_optuna_notebook.py` and `test_notebook_hf_models.py`; add focused tests for telemetry, cache policy, staging and session identity/reset behavior.

- Use fake clocks/processes to verify deadlines, partial startup records, timing boundaries and readiness ownership checks. Ensure log capture works with mute enabled and large output.
- Cover success, startup failure, benchmark failure, pruning, keyboard interruption, cleanup failure and unfinished-attempt recovery. Verify no worker/listener survives and cleanup errors prevent another trial.
- Check old metadata still loads, JSONL consumers receive valid rows, exceptional attempts retain logs/accounting, and rest/final session cleanup appear in experiment totals.
- Verify cache paths survive run-directory changes and worker restart; workload cache misses on any identity change; TRT configuration-content changes invalidate session identity even at the same path.
- Prove shared notebook/CLI policy plumbing, unchanged manifests/configuration generation, and identical adaptive ask/evaluate/tell order. Unsupported reset must force restart.
- On real GPUs, establish repeat variability first, then use matched repeated comparisons of baseline, warm persistent caches/local loading, loader alternatives, and reuse. Keep request hashes and semantic trial settings fixed; declare loader/path differences explicitly with tensor-equivalence evidence. Assess output behavior, throughput, TTFT and E2E latency against predeclared tolerances derived from repeat variability.
- Accept an optimization only when total experiment wall time improves with all preparation/prewarming, failures, rest and shutdown included, GPU cost is reported, steady-state behavior remains equivalent, and lifecycle checks pass. Do not promise a speedup until these measurements exist.

Deliver one reviewable change set per stage. Stage 1 establishes trustworthy evidence; stage 2 reduces repeated disk/compilation/client work; stage 3 remains opt-in until backend reset semantics and fresh-engine comparisons pass.
