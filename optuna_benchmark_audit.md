# LLM Optimizer Benchmarking and Optuna Audit

Audit date: 2026-09-22
Repository: `/mnt/c/Users/maorb/work/bento_extend/llm-optimizer`
Optuna installed in the audit environment: **4.9.0** (`pyproject.toml` permits `>=4,<5`).

This is a read-only audit. No source, configuration, dependency, or benchmark file was modified.

## A. Executive summary

| Capability | Status | Evidence |
|---|---|---|
| Throughput optimization | **Supported** | `output_throughput` is the default objective; `optimization.py:20-31`, `cli.py:748-752`; scalar trials are asked/told in `cli.py:562-612`. |
| P95 TTFT measurement | **Supported, with measurement caveats** | Per-request streaming timestamps and `p95_ttft_ms` are implemented in `bench_client.py:1140-1199`; request handlers start their clocks before `session.post()` (`bench_client.py:140-169`, `215-258`, `387-437`). |
| P95 TTFT as an objective | **Supported** | `p95_ttft_ms` is in `LATENCY_OBJECTIVES` and is minimized (`optimization.py:26-31`, `46-67`). |
| Throughput optimization with a TTFT constraint | **Supported for single-objective studies** | CLI constraints, violation calculation, `constraints_func`, and feasible-best reporting exist (`cli.py:726-727`, `optimization.py:83-132`, `cli.py:449-455`, `646-677`). |
| Multi-objective optimization | **Missing** | `create_study(direction=direction)` is scalar; `study.tell(trial, benchmark_value)` and `trial.value` assume one value (`cli.py:503-510`, `612`, `638-645`). |
| Pareto visualization | **Partial** | Dashboard supports selecting X/Y fields and displays data points, but the study/result pipeline does not create Optuna Pareto trials or mark Pareto fronts (`visualization/visualize.py:109-192`, `260-305`; `cli.py:646-677`). |
| Configurable samplers | **Partial** | Class names/import paths and JSON kwargs are passed through (`cli.py:457-500`), but not every sampler is compatible with the current search space or constructor call. |
| Functional intermediate pruning | **Missing** | A pruner is instantiated and passed to `create_study`, but there is no `trial.report()` or `trial.should_prune()` (`cli.py:500-510`; repository-wide search found neither call). |
| Reproducible benchmark comparison | **Partial** | Seeds, search signatures, configs, GPU type/count, and durable study state are recorded, but workload generation, environment state, failures, and run-to-run variance are not fully controlled or measured. |

### Immediate conclusions

1. Existing scalar, throughput-only Optuna experiments can start after normal environment/GPU validation.
2. TTFT is a genuine client-observed streaming measurement per successful request, not a complete-duration proxy. It includes time from immediately before the HTTP POST through the first non-empty token event, including server queueing after dispatch and client/network overhead.
3. P95 and P99 TTFT can be selected directly as scalar objectives.
4. Output throughput subject to P95/P99 TTFT constraints is already supported through `--constraints`, with Optuna constraint values supplied to samplers that accept `constraints_func`.
5. A throughput-vs-TTFT dashboard can be generated from stored results, but native multi-objective Optuna optimization and explicit Pareto-front annotation are not implemented.
6. Pruner selection currently changes the study object only; it does not shorten any benchmark because trials have no intermediate reports.

## B. Current architecture

The actual path is:

```text
CLI options
  -> cli.benchmark()
  -> OptunaSearchSpace (categorical ArgSets + integer ranges)
  -> study.ask() / space.suggest(trial)
  -> construct_benchmark_settings()
  -> start managed inference server
  -> bench_client.run_benchmark()
  -> warmup + finite request batch + streaming request handlers
  -> calculate_metrics()
  -> extract_objective()
  -> constraint_violations() / trial user attributes
  -> study.tell(trial, scalar value)
  -> JSONL per-trial record + combined JSON + HTML dashboard
```

The CLI supports SGLang, vLLM, MAX, and TensorRT-LLM through `predefined/__init__.py:11-42`. Optuna trials are created in `cli.py:326-678`. Parameter suggestions are generated in `optimization.py:198-264`; finite argument lists become categorical index distributions and `int(low,high,step)` definitions become Optuna integer distributions.

Each trial starts a fresh managed server, waits for readiness, runs the benchmark, and terminates the server in `run_single_configuration()` (`cli.py:245-315`). The Optuna default benchmark settings are ShareGPT, 1,000 prompts, infinite request rate, and seed 1 (`cli.py:260-271`). Client arguments overwrite these defaults.

## C. Detailed findings

### 2. TTFT measurement

#### Measurement path

`RequestFuncOutput` stores one `ttft`, one complete `latency`, and a list of `itl` samples per request (`bench_client.py:79-94`). The supported benchmark backends map to streaming request handlers at `bench_client.py:577-585`.

For the OpenAI-compatible vLLM/SGLang-OAI path, the timer is set immediately before `session.post()` (`bench_client.py:213-220`). Chunks are parsed, `[DONE]` and empty text chunks are ignored, and the first chunk with non-empty `choices[0].text` sets `output.ttft` (`bench_client.py:223-249`). The TensorRT-LLM handler applies the same first non-empty `text_output` rule (`bench_client.py:140-169`). The native SGLang handler uses the first non-empty `data["text"]` event (`bench_client.py:387-433`).

Therefore:

- This is genuine streaming TTFT, not an estimate from complete request duration.
- It is recorded individually for each successful request.
- It is client-observed TTFT: it includes HTTP/client scheduling, request transmission, server queueing after the POST is issued, server prefill, and time until the first non-empty streamed token/chunk is received. It is not server-internal TTFT.
- The timer does **not** include time a task spends waiting for the client semaphore, because `st` is created inside the request handler after semaphore acquisition (`bench_client.py:1262-1274`, then handler lines above).
- Request-rate delay is outside the individual request timer but is included in the benchmark wall-clock denominator. With `inf`, requests are yielded without delay (`bench_client.py:1122-1137`).
- `max_concurrency` limits client-side in-flight handlers; it is not necessarily the server's own scheduling limit.

The first response event is not blindly treated as a token: OpenAI-compatible and native SGLang handlers require non-empty text, and the code explicitly comments about usage-only final events (`bench_client.py:235-243`, `318-326`, `409-420`). TensorRT-LLM appends `text_output` for each non-empty chunk, but does not independently verify a token field (`bench_client.py:146-159`).

#### Percentiles and metric names

`calculate_metrics()` collects successful-request TTFTs and calculates:

- `mean_ttft_ms`
- `median_ttft_ms`
- `std_ttft_ms`
- `p95_ttft_ms`
- `p99_ttft_ms`

at `bench_client.py:1153-1168` and `1194-1199`. There is no P90 TTFT calculation. P95/P99 are NumPy percentiles over successful requests only, in milliseconds.

The benchmark supports all four main framework backends through the mapped streaming handlers, but exact wire semantics depend on each backend's streaming protocol. A framework that returns a successful response with no recognized non-empty token can be marked successful with `ttft=0` in some handlers; this is a correctness risk for empty-output cases.

Failures and exceptions set `success=False` and preserve an error string in each request output (`bench_client.py:259-269`, `342-349`, `439-450`). `calculate_metrics()` excludes failed requests from latency and token aggregates (`bench_client.py:1155-1175`) but does not expose a failure count or failure rate in `BenchmarkMetrics` or the stored result. A request that times out at the six-hour aiohttp session limit is caught as a failed request, assuming the process reaches that exception path (`bench_client.py:54-64`). Cancellation handling is not separately modeled. If all requests fail, the code warns but still computes zero-like latency values (`bench_client.py:1176-1217`), which can make an invalid run look numerically usable unless objective validation rejects the resulting objective.

The implementation distinguishes TTFT from decoding time: ITL is measured between subsequent token/chunk timestamps; TPOT is computed from `(complete latency - TTFT) / (output_len - 1)` (`bench_client.py:1165`, `1200-1209`). TTFT is therefore first-token wait; TPOT is an average post-first-token duration per output token; ITL is a distribution of inter-arrival intervals, with SGLang dividing multi-token updates across the number of new tokens (`bench_client.py:423-430`).

### 3. Available performance metrics

| Metric name | Definition / denominator | Unit | Computation and storage | Optuna objective? |
|---|---|---:|---|---|
| `request_throughput` | Successful request count / benchmark duration | requests/s | `bench_client.py:1182-1193`, stored `1505-1513` | Yes; maximize |
| `input_throughput` | Input tokens from successful requests / duration | tokens/s | Same lines; stored | Yes; maximize |
| `output_throughput` | Requested/recorded generated output token count for successful requests / duration | tokens/s | Same lines; stored | Yes; maximize; default |
| `total_throughput` | Successful input + output tokens / duration | tokens/s | Same lines; stored | Yes; maximize |
| `output_throughput_retokenized` | Retokenized generated text tokens / duration | tokens/s | Calculated `1189-1193`, only returned in Python result; not included in serialized result dict `1505-1536` | No |
| `total_throughput_retokenized` | Retokenized input + output equivalent / duration | tokens/s | Calculated `1192-1193`; not serialized in standard result | No |
| `mean_ttft_ms`, `median_ttft_ms` | Mean/median successful-request first-token times | ms | `1194-1197`; stored `1519-1521` | Yes; minimize |
| `p95_ttft_ms`, `p99_ttft_ms` | 95th/99th percentile successful-request TTFT | ms | `1198-1199`; stored `1522-1523` | Yes; minimize |
| `mean_tpot_ms`, `median_tpot_ms` | Mean/median per-request post-TTFT duration divided by `output_len-1`, only requests with output length >1 | ms/token | `1165`, `1200-1204`; stored `1524-1528` | Yes; minimize |
| `p95_tpot_ms`, `p99_tpot_ms` | Percentiles of the per-request TPOT values | ms/token | `1203-1204`; stored | Yes; minimize |
| `mean_itl_ms`, `median_itl_ms` | Mean/median of all collected inter-token/chunk intervals | ms | `1205-1209`; stored `1529-1533` | Yes; minimize |
| `p95_itl_ms`, `p99_itl_ms` | Percentiles over flattened ITL samples | ms | `1208-1209`; stored | Yes; minimize |
| `max_itl_ms` | Maximum flattened ITL sample | ms | `1210`; present in `BenchmarkMetrics` but omitted from standard serialized result | No |
| `mean_e2e_latency_ms`, `median_e2e_latency_ms` | Mean/median complete client-observed successful request latency | ms | `1169`, `1211-1215`; stored `1514-1518` | Yes; minimize |
| `p95_e2e_latency_ms`, `p99_e2e_latency_ms` | Percentiles of complete successful request latencies | ms | `1214-1215`; stored | Yes; minimize |
| `std_*` latency fields | Population standard deviations of collected samples | metric unit | `1197`, `1202`, `1207`, `1213`; some stored | No; not in supported objective set |
| `duration` | Main benchmark wall-clock duration from just before request task creation until all tasks finish | seconds | `bench_client.py:1351-1423`, stored `1505` | No |
| `completed` | Number of successful requests | requests | `1182-1185`, stored `1506` | No |
| `concurrency` | `sum(successful e2e latencies) / duration`; an average concurrency proxy, not configured max concurrency | requests | `1216`, stored `1534` | No |
| `total_input_tokens`, `total_output_tokens` | Successful request token totals | tokens | stored `1507-1509` | No |
| failures / timeout rate | Not calculated as a named metric | — | Per-request `errors` exists only with `output_details=True` (`1541-1551`); no count/rate field | No |
| GPU utilization / peak allocated memory | Not collected by benchmark client | — | No GPU telemetry fields in `BenchmarkMetrics` or serialized result | No |

There are no P90 metrics for TTFT, ITL, TPOT, or E2E latency. The README/configuration may mention or expose fields, but the authoritative calculation is `calculate_metrics()` and the result dictionary above.

The denominator for throughput is the main benchmark duration, not the sum of successful request latencies and not the offered workload duration. Consequently, throughput is throughput of successful completions over the wall-clock run. Failed requests remain in the offered batch but are excluded from token totals and request throughput numerator. This is useful for serving capacity, but it can hide rejection/failure behavior unless `completed` and errors are inspected.

The current metrics are sufficient for a first throughput/latency comparison, but not sufficient for a production-grade interactive-inference study without adding failure rate, queue-wait separation, GPU telemetry, and repeated-run uncertainty.

### 4. Optuna objective selection

The CLI default is `output_throughput` at `cli.py:748-752`. For Optuna, `benchmark()` validates the name with `objective_direction()` (`cli.py:923-936`). `run_single_configuration()` calls `bench_client.run_benchmark()`, then `extract_objective(benchmark_result, objective)` (`cli.py:260-280`). `extract_objective()` accepts only finite numeric values and rejects missing/non-numeric values (`optimization.py:58-67`). The value is passed to `study.tell(trial, benchmark_value)` (`cli.py:607-612`).

Supported objective names are exactly the four throughput fields and all combinations of `mean`, `median`, `p95`, `p99` with `ttft`, `itl`, `tpot`, and `e2e_latency`, suffixed `_ms` (`optimization.py:20-31`). Throughput directions are maximize and latency directions are minimize (`optimization.py:46-55`). Thus both `p95_ttft_ms` and `p99_ttft_ms` are directly selectable.

The current study is explicitly scalar: `optuna.create_study(direction=direction)` takes one direction, the trial receives one scalar through `study.tell`, and best-trial reporting uses `trial.value` (`cli.py:503-510`, `612`, `633-645`). There is no multi-direction list, `directions=`, `FrozenTrial.values`, or Pareto-best handling. Auxiliary metrics are retained in each JSONL result under `results`, but not attached as Optuna objective values or reported intermediate values (`cli.py:283-305`, `bench_client.py:1505-1536`).

Invalid/missing/non-finite objectives cause the trial path to fail and mark the Optuna trial `FAIL` (`optimization.py:58-67`, `cli.py:617-623`). Startup errors, benchmark errors, and OOM-like process failures are caught at the trial boundary; the server is cleaned up by `finally` in `run_single_configuration()` (`cli.py:312-315`).

### 5. Constrained optimization

Constraints are configured through `--constraints` (`cli.py:726-727`) and parsed by `parse_slo_constraints()`. `constraint_violations()` maps `ttft:p95<200ms` to `p95_ttft_ms`, converts seconds to milliseconds, and returns Optuna-style non-positive values for satisfied constraints (`optimization.py:70-122`).

For every completed trial, violations and a Boolean `slo_feasible` attribute are set on the Optuna trial (`cli.py:607-610`). When constraints exist, the code supplies a `constraints_func` to the sampler (`cli.py:449-455`). It also stores constraints in each result and reports constrained and unconstrained best configurations (`cli.py:283-301`, `646-677`; `cli.py:89-150`).

Therefore `--optuna-objective output_throughput --constraints 'ttft:p95<=200ms'` is supported without changing the benchmark. It is not equivalent to discarding violating trials: constrained TPE receives violation values and can use feasibility information, while the reporting layer separately preserves the unconstrained best and the best feasible trial. Failed trials receive infinite constraint violations and `slo_feasible=False`, so they are distinguishable in the Optuna state/user attributes, although the combined JSON `test_results` contains only successfully persisted benchmark result records and does not have a separate failed-trial record for every failure.

Optuna 4.9.0 supports constrained samplers through `constraints_func` on `TPESampler` and several other samplers. The current code unconditionally injects that callable for any selected sampler unless the JSON kwargs already contains the key (`cli.py:482-499`). A selected sampler that does not accept `constraints_func` will fail at construction; this is not prevalidated.

The smallest required changes for a different constraint policy would be in `constraint_violations()`, the sampler construction, and result/trial failure persistence. No benchmark modification is required for the existing single-objective constraint path.

### 6. Samplers and search-space compatibility

The recently added options are confirmed at `cli.py:761-781`. The implementation looks up a class by name in `optuna.samplers`, or imports a dotted class path with `importlib` (`cli.py:493-499`), and constructs it with a dictionary parsed by `json.loads()` (`cli.py:482-490`). JSON converts primitive values, arrays, objects, booleans, and nulls, but cannot express Python callables such as `gamma`, `weights`, `constraints_func`, crossover strategies, or custom distance functions. Fully qualified external classes are importable only if their package is already installed and importable in the runtime; OptunaHub installation/discovery is not separately implemented.

The following behavior was checked against Optuna 4.9.0 constructor signatures:

| Sampler | Can the current generic constructor call instantiate it? | Compatibility with current space |
|---|---|---|
| `TPESampler` | Yes, with supported JSON primitive kwargs such as `multivariate`, `group`, `constant_liar`, `n_startup_trials`, and `n_ei_candidates`. | Good for categorical and stepped integer distributions. Constraint callable is injected when applicable. |
| `RandomSampler` | Yes, with `seed`. | Good for categorical and integer distributions. |
| `GPSampler` | Usually yes with primitive kwargs such as `seed`, `n_startup_trials`, and `deterministic_objective`; callable kwargs are unavailable via JSON. | May independently sample unsupported/less suitable distributions; must be tested on the exact space. |
| `CmaEsSampler` | Constructor can instantiate, but the current mixed categorical/integer space is not a good compatible space. | CMA-ES is continuous-oriented; categorical parameters are independently sampled or unsupported, and the configuration does not validate this before GPU execution. |
| `GridSampler` | **No** with the current call: its required `search_space` constructor argument is not supplied. | Even if constructed, it expects a mapping of parameter names to value sequences and conflicts conceptually with the current `study.ask()` categorical-index space unless adapted. |
| `NSGAIISampler` | Constructor can instantiate with primitive kwargs. | It is intended for multi-objective evolutionary optimization; the current study is single-objective, so it does not provide the intended Pareto behavior. |

The current search space is a product of finite categorical choices and stepped integer distributions (`optimization.py:198-231`). `suggest()` uses categorical indices for each multi-valued ArgSet and `suggest_int()` for each `IntRange` (`optimization.py:243-264`). Conditional parameters are not represented: every ArgSet is sampled unconditionally, and there is no Optuna conditional branch based on another trial parameter. Coupled parameters are supported only as a fixed composite choice such as `tp_size*dp_size`, not as dynamic conditional distributions.

Framework-specific validation is incomplete. Integer ranges require a known integer entry in the selected framework/client config (`optimization.py:157-195`), but server argument parsing is non-strict in `cli.py:977-982`, and native server CLI validity is ultimately delegated to the framework process. There are separate studies when different output/signature/study names are used; the framework is included in the signature (`cli.py:376-397`).

Resume support is durable and signature-checked: SQLite defaults beside the output, `--continue` loads the existing study and JSONL, and changes to model/framework/search space/objective/constraints/GPU/host/port are rejected (`cli.py:408-455`, `README.md:163-181`). The signature currently includes sampler/pruner names and raw kwargs text (`cli.py:376-397`), while result metadata stores only class names, not the full effective constructor kwargs (`cli.py:646-666`). This is insufficient for complete reproducibility if two configurations have equivalent class names but different effective/default kwargs, although the signature does prevent changes to the provided raw kwargs.

### 7. Pruning support

There is no call to `trial.report()` or `trial.should_prune()` in the repository. The benchmark collects final per-request data only after `asyncio.gather(*tasks)` completes (`bench_client.py:1351-1390`), then computes final metrics (`bench_client.py:1416-1424`). The CLI now constructs and passes a pruner to `create_study()` (`cli.py:500-510`), but no trial execution code consults it.

Selecting `MedianPruner`, `SuccessiveHalvingPruner`, or `HyperbandPruner` therefore does not prune an active benchmark. They remain unused because no intermediate values are reported. Duplicate configurations are labeled `PRUNED` (`cli.py:573-580`), but that is duplicate suppression, not pruner behavior.

No cumulative-throughput or partial-TTFT callback exists. The current request tasks could technically be instrumented after a configurable number of completed requests, but this would require coordinated asynchronous progress collection, safe cancellation of remaining tasks, server cleanup, partial-result semantics, and a defined comparable step/resource scale. TTFT percentile estimates from very short prefixes are especially unstable and are not guaranteed to predict the final 1,000-request percentile. Multi-objective intermediate reporting is also absent, and Optuna pruners are generally scalar-oriented; a multi-objective design needs an explicit policy for what intermediate value means.

Pruning should not be enabled automatically for the current benchmark. A future implementation would need benchmark-level checkpoints, `trial.report(step)`/`should_prune()` integration, cancellation and cleanup, and validation that early measurements correlate with full-run measurements.

### 8. Benchmark comparability and statistical reliability

#### Workload and seeds

Optuna trial settings default to ShareGPT, 1,000 prompts, infinite arrival rate, and seed 1 (`cli.py:260-271`). Dataset creation honors `num_prompts`, random lengths, ShareGPT fixed length, and other client settings (`bench_client.py:516-574`). The benchmark client seeds Python `random` and NumPy from `args_dict["seed"]` (`bench_client.py:1575-1578`). The seed is fixed in the Optuna trial defaults but is not included as an explicit client argument in the saved `client_args` unless supplied; the trial result metadata does not record the benchmark seed separately.

For `dataset_name=random`, fixed seed and fixed input/output lengths can make requests repeatable. For ShareGPT, the same source/sample behavior is expected from the fixed seed, but external dataset/cache state and tokenizer/model revisions still matter. Random datasets are regenerated per benchmark process unless a generated-prefix cache path is used. Requested output lengths and actual output lengths are distinguished in the metrics: `output_len` is updated from usage for the OpenAI path and stored in `total_output_tokens`; retokenized output is also computed. However, generated output lengths can vary by backend/EOS behavior, so throughput comparisons can be affected by different completion counts.

#### Timing and saturation

With `request_rate=inf`, all request tasks are created immediately (`bench_client.py:1351-1390`). The semaphore limits active requests, but the timer starts only after a task acquires the semaphore. Thus queueing in the client semaphore is not included in per-request TTFT/E2E latency, while server-side queueing after HTTP submission is included. The benchmark wall-clock duration starts before task creation and ends after all tasks complete (`bench_client.py:1351-1417`), so client scheduling and task creation can influence throughput duration.

Unlimited arrival plus finite `max_concurrency` is a saturated offered workload, but not a guarantee that the server is saturated: server request limits, HTTP failures, rejection behavior, or a low output workload can dominate. Failed requests are excluded from throughput numerator and no failure rate is reported. A configuration that rejects many requests can therefore appear less bad than it should if only output throughput is considered; always inspect `completed` and errors/details.

#### Warmup, rest, and interference

Each benchmark runs one warmup request by default, using the first prompt and at most 32 output tokens, then waits one second before starting the main timer (`bench_client.py:1276-1330`, `1349-1354`). Warmup is excluded from the recorded main duration, but cache state and server state are intentionally changed by warmup. Model loading and server startup occur before benchmark timing (`cli.py:252-272`). CUDA graph capture/compilation behavior is framework-dependent; the optimizer does not detect or verify that those events have completed beyond the readiness endpoint and warmup.

The default inter-trial rest is 10 seconds (`cli.py:658-660`), but it is not a thermal, cache, or memory-isolation guarantee. Only one active worker per study is supported by documented behavior; separate processes/jobs can still interfere on shared GPUs. GPU type/count and model tag are stored (`cli.py:288-294`), but utilization, clocks, power, temperature, peak memory, driver, CUDA, framework, and model revision are not captured in benchmark results.

The optimizer skips identical configuration IDs in a study and does not deliberately repeat completed configurations for variance estimation (`cli.py:548-580`). It cannot currently produce confidence intervals or select based on replicate distributions. This is a significant source of risk for close Optuna comparisons.

### 9. Result storage and visualization

Every successful benchmark result stores throughput and P95/P99 TTFT together in the `results` object (`bench_client.py:1505-1536`). JSONL records preserve the full `config` object, including server/client args, native command args, normalized config, and TensorRT-LLM engine/runtime split (`cli.py:283-305`, `59-70`). Optuna records additionally contain trial number, objective value, feasibility, and study signature.

The combined JSON contains metadata, best-throughput summaries, all `test_results`, and Optuna optimization metadata (`cli.py:184-217`, `646-677`). Failed benchmark trials are marked `FAIL` in the Optuna SQLite study but are not appended as normal result records when `run_single_configuration()` raises. SLO-infeasible successful trials remain in JSONL and are marked in trial attributes; they are not automatically visually distinguished as a separate category in the dashboard.

The dashboard loads one or more enhanced JSON files, normalizes records, passes all data to an HTML/Plotly template, and supports configurable field mappings (`visualization/visualize.py:109-192`, `250-305`; `visualization/visualization_config.json:1-37`). The default field configuration includes P95 TTFT and output throughput (`visualization_config.json:24-35`, `282-296`). The generated dashboard allows X/Y metric selection and hover data; it can generate a graph with X=`p95_ttft_ms` and Y=`output_throughput` from existing stored results without rerunning benchmarks.

However, the existing visualization is not an Optuna multi-objective Pareto implementation. The dashboard contains performance exploration/best-configuration logic, but the optimizer's stored Optuna study has one scalar objective and no `pareto_front`/multi-value trial representation. The full server/client config and most benchmark statistics are present for hover if mapped by the dashboard configuration; GPU memory utilization is not present because the benchmark never collects it. Adding a true Pareto highlight would require computing dominance over valid records, deciding how failed/infeasible records are treated, and adding the corresponding UI state.

### 10. Implementation and test coverage

Covered capabilities include:

- Search-space scoping, categorical suggestions, composite arguments, integer ranges, and duplicate range validation (`tests/test_optimization.py:30-166`).
- Objective direction and numeric extraction (`tests/test_optimization.py:169-175`).
- Constraint operators and missing constraint metrics (`tests/test_optimization.py:177-195`).
- Optuna dry-run, optional dependency error, SQLite resume/signature validation, and integer-range study persistence (`tests/test_optimization.py:198-380`).
- TensorRT argument classification, native spelling, GPU checks, startup cleanup, and visualization loading/normalization (`tests/test_trtllm.py:50-122`, `210-270`, `307-347`).

Not covered by the current tests:

- The newly added `--optuna-sampler`, `--optuna-sampler-kwargs`, `--optuna-pruner`, and `--optuna-pruner-kwargs` options.
- Dotted sampler/pruner imports, invalid JSON kwargs, constructor failures, sampler-specific parameters, or persisted full kwargs.
- `GPSampler`, `CmaEsSampler`, `GridSampler`, `RandomSampler`, or `NSGAIISampler` behavior.
- Any `trial.report()`/`trial.should_prune()` behavior; no pruning tests exist.
- TTFT request-handler parsing, first-token selection, empty output, timeout, cancellation, failure-rate behavior, or percentile calculations. No unit tests target `calculate_metrics()` or the streaming handlers in the inspected test files.
- P90 absence/behavior, GPU telemetry absence, repeated-run variance, or cross-framework metric comparability.
- Multi-objective studies, Pareto fronts, or Pareto visualization.
- Loading an Optuna JSON/JSONL result into the dashboard together with failure/constraint-state semantics.

Tests actually attempted during this audit:

1. `python - <<... import optuna ...` — passed; reported Optuna `4.9.0`.
2. `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests/test_optimization.py` — could not start because pytest could not create a temporary file: the environment reported no usable writable temporary directory (`/tmp`, `/var/tmp`, `/usr/tmp`, or repository path).
3. `PYTHONDONTWRITEBYTECODE=1 python -m llm_optimizer.cli --help | rg ...` — passed and displayed all four new sampler/pruner options.
4. AST parsing of `src/llm_optimizer/cli.py` — passed (`syntax ok`).

No GPU benchmark, model download, package installation, or expensive operation was run.

## D. Required changes

| Priority | Proposed change | Why needed | Affected files | Estimated complexity |
|---|---|---|---|---|
| P0 before baseline | Fix/verify the test environment's writable temp directory and run the existing Optuna test suite. | The audit could not obtain a passing pytest result; baseline infrastructure should be validated before GPU experiments. | Test/runtime environment; no source change required | Low |
| P0 before baseline | Define and record benchmark workload identity: dataset/source revision, model revision, tokenizer revision, seed, exact input/output lengths, framework/runtime versions, GPU/driver/CUDA metadata. | Prevents Optuna from optimizing environmental or workload differences. | `bench_client.py`, `cli.py`, result schema | Low–Medium |
| P0 before baseline | Add failure count/rate and validate successful-request coverage before accepting a throughput objective. | Current throughput is based on successful requests only and can hide rejection/failure behavior. | `bench_client.py`, `optimization.py`, `cli.py` | Low–Medium |
| P0 before baseline | Add tests for `calculate_metrics()` and at least one streaming handler per protocol. | TTFT and percentile correctness is central to P95/P99 experiments and currently lacks unit coverage. | `tests/`, `bench_client.py` only if defects are found | Medium |
| P1 before TTFT-constrained experiments | Decide whether TTFT should include client semaphore wait. If so, capture a separate task-created timestamp and report both queue-inclusive and dispatch-to-first-token TTFT. | Current TTFT excludes client-side semaphore queueing but includes server/network time; the interpretation must be explicit. | `bench_client.py`, result schema | Medium |
| P1 before tight comparisons | Add configurable repetitions or a replicate mode and summarize variance/confidence intervals. | One 1,000-request run per configuration can cause noisy sampler choices. | `cli.py`, `bench_client.py`, result/reporting code | Medium |
| P1 before multi-objective work | Implement scalar-vs-multi-objective study creation, multi-value trial reporting, Pareto best selection, and result metadata. | Current `direction`, `trial.value`, `study.tell`, and summaries are scalar-only. | `cli.py`, `optimization.py`, result schema, tests | Medium–High |
| P1 before Pareto visualization | Add explicit dominance/Pareto-front computation and valid/infeasible/failed styling; add GPU memory fields only if telemetry is collected. | Current dashboard can plot two fields but does not identify Optuna Pareto trials and lacks memory telemetry. | `visualization/visualize.py`, `visualization_config.json`, result schema | Medium |
| P2 | Validate sampler compatibility and add tests; special-case `GridSampler` search-space construction and reject/translate incompatible CMA-ES spaces. | Generic constructor passthrough currently makes some advertised sampler choices fail at runtime or behave poorly. | `cli.py`, `optimization.py`, `tests/` | Medium |
| P2 | Persist full effective sampler/pruner class paths and kwargs, including defaults where reproducibility matters. | Current metadata stores class names, while signatures store raw CLI strings rather than effective constructor configuration. | `cli.py`, result schema | Low–Medium |
| P3, conditional | Add intermediate benchmark checkpoints and safe cancellation only after a correlation study shows early metrics predict full-run metrics. | Current pruning is non-functional and early TTFT/throughput estimates may be misleading. | `bench_client.py`, `cli.py`, tests | High |

The existing single-objective constraint mechanism is already implemented and should not be duplicated. The new sampler/pruner CLI options are also implemented; the required work is compatibility validation and testing, not re-adding those options.

## E. Minimal implementation plan

1. Establish a reproducible baseline protocol: fixed dataset mode, fixed seed, exact input/output lengths, one framework/model/GPU environment at a time, no simultaneous GPU jobs, and explicit failure-rate acceptance criteria.
2. Run the supplied finite search spaces with the default scalar `output_throughput` objective, retaining `completed`, TTFT, E2E, and all configuration fields.
3. Run constrained scalar studies using `--constraints 'ttft:p95<=...'` or `ttft:p99<=...`; inspect feasible and unconstrained results separately.
4. Add metric/stream-parser tests and validate a small mocked benchmark before GPU runs.
5. For multi-objective work, introduce a separate study mode rather than overloading scalar result code: accept two objective names/directions, call `create_study(directions=[...])`, tell a value sequence, store `values`, compute Pareto dominance, and update best-trial summaries.
6. Extend the dashboard to consume those values or compute the Pareto front from existing records. Treat failed records as excluded and infeasible records as visible but non-Pareto unless the experiment explicitly requests another policy.
7. Investigate pruning only with a controlled pilot comparing early-prefix metrics to full-run metrics. If correlation is insufficient, retain full benchmarks and do not enable pruning.

## F. Final assessment questions

1. **Can we start existing throughput-only Optuna experiments immediately?** Yes, after confirming the environment and GPU are clean and after validating the test/runtime setup. The scalar Optuna path, finite search space, SQLite persistence, and default output-throughput objective exist.
2. **Do we already measure genuine per-request TTFT?** Yes. It is measured from just before the HTTP POST to the first recognized non-empty streamed token/chunk for each successful request. It is client-observed and excludes client semaphore wait before dispatch.
3. **Can we currently optimize P95 TTFT directly?** Yes, with `--optuna-objective p95_ttft_ms`; it is minimized.
4. **Can we maximize throughput subject to a P95 TTFT requirement?** Yes for a scalar, constraint-aware study, using `--optuna-objective output_throughput --constraints 'ttft:p95<=200ms'`. Validate failure coverage and sampler compatibility before trusting results.
5. **Can we generate the desired throughput–TTFT Pareto graph from existing results?** You can plot both stored fields from existing JSON/JSONL without rerunning benchmarks. The current dashboard does not natively compute/highlight a true Pareto front, and GPU memory utilization is unavailable.
6. **Which changes are genuinely necessary before experimentation, and which can wait?** Before baseline: workload/environment recording, failure-rate visibility, test execution, and basic TTFT/metric tests. Before constrained TTFT experiments: clarify queueing semantics and validate percentile behavior. Multi-objective study support, explicit Pareto highlighting, sampler compatibility work, repetitions, and pruning can wait until the scalar baseline is established; pruning should wait for evidence that intermediate measurements are predictive.
