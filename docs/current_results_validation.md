# Current-results report validation

The report follows section → base model → framework → concurrency → objective. Notebook version, phase, workload, hardware, sampler, seed, and search space are study metadata. ModelOpt artifacts such as `Qwen/Qwen3-14B@modelopt-fp8-323a067030074b29` share their base model heading, and exact IDs appear in rankings and run inventories.

Each model summary ranks up to five studies across frameworks for each concurrency and objective. Samplers are ordered by their best study result, with counts for all eligible runs and for runs with an eligible completed trial.

The November 17 collection contains scalar-objective studies. Each ranking selects the best eligible completed trial per study and orders those winners by objective direction. Output throughput and p95 E2E latency have individual summaries, plots, and rankings.

| Stage | Studies | Trials | Completed |
|---|---:|---:|---:|
| Raw | 423 | 8,137 | 6,410 |
| Deduplicated | 388 | 7,602 | 6,000 |
| Eligible | 371 | 7,417 | 5,868 |

All reference counts reconcile exactly. Classification partitions eligible data without changing these totals.

| Section | Studies | Trials | Completed |
|---|---:|---:|---:|
| Initial runs | 281 | 5,617 | 4,478 |
| Parameters probe | 90 | 1,800 | 1,390 |
| Unclassified | 0 | 0 | 0 |

Full names are recovered from collected W&B transaction logs by their recorded W&B IDs. Checksummed records are decoded structurally. The latest display-name record in each log supersedes its initial placeholder; prior names and source paths remain in provenance. Independent disagreements remain unclassified. Older JSON exports do not supply matching collection IDs, and no historical experiments are added.

Excluded studies comprise 11 runs below 300 seconds and six with unknown duration; six of the short runs also have unverified/synthetic measurement status. Exactly 300 seconds is retained. The 86 empty database/folder artifacts remain separately inventoried. Unverified VM identity and the original campaign are recorded in provenance.

Timing extraction joins 7,399 attempts to 7,399 trials: 5,601 initial and 1,798 probe attempts. Eighteen eligible trials have no recorded execution. Six additional execution artifacts lack a recorded trial identity or matching attempt ID and remain in the audit, outside aggregates. Repeated copies of an attempt count once; distinct joined attempts would each contribute to attempt statistics. Per-trial launch means first average attempts within each trial.

| Complete recorded phase | Observations |
|---|---:|
| Launch-to-ready | 6,416 |
| Client preparation | 6,416 |
| Warmup | 6,416 |
| Benchmark | 5,868 |
| Runner wall | 5,868 |
| Cleanup | 7,399 |

Timing means and sample variances combine complete observations of the same phase and definition within each objective group. Incomplete phases stay in closed detail tables. Weight-loading, compilation, graph-capture, and autotuning log observations remain separate from wall phases and are never added together.

Parameters probe has 1,478 recorded server-telemetry attempts: 1,390 complete benchmark windows and 88 incomplete windows. Initial runs have another 129 recorded telemetry attempts. Both sections include objective-level telemetry coverage and retain attempt details. Incomplete benchmark windows do not contribute to pressure means or peaks. Coverage tables retain units, sample counts, measured windows, availability, coverage fractions, aggregation semantics, and missing reasons. Gauge means weight recorded covered seconds; analytic client integrals retain their distinct definition.

Representative populated v3 trial records from SGLang, vLLM, and TensorRT-LLM are checked against their stored attempt identities, availability, counts, and averages. Recorded framework version flags distinguish schemas; an unconfirmed adapter label containing `v3` does not establish a v3 telemetry schema. The shared nine wall-timing fields already occur in v2. The additional common v3 fields are `server_telemetry`, `request_throughput`, and `configured_client_concurrency`; `startup_error` also appears in the observed TensorRT-LLM v3 difference.

TensorRT-LLM records running requests, queued requests, and KV-cache utilization as unsupported/not exposed. GPU utilization is unavailable throughout the recorded telemetry because no GPU index/device was configured. These values remain unavailable. Scrape errors and unavailable individual scrape samples are explicit in details.

Both fANOVA and PED-ANOVA computed 323 study/objective estimates and skipped 48 for insufficient completed candidates. Every study/objective/method has exactly one status. Parameter exclusions, warnings, and method results remain auditable; no parameter is silently excluded.

The notebook executed all 99 code cells top to bottom on November 18, 2026. All 17 focused tests and the collection assertions passed. The rendered notebook contains eight model summaries, 58 concurrency sections, 87 objective groups, 200 plots, and 88 closed detail toggles. Structural checks verified Markdown headings, pandas summary outputs, plot outputs, unique detail tables within each objective toggle, and a single audit toggle. The objective cells match the groups discovered in the November 17 collection.

SHA-256 checks matched all 20,623 inventoried source files before and after notebook execution. The inventory covers the collection tree, executed source notebooks, cached W&B name evidence, and the v2 reference notebook. All 134 SQLite databases were opened read-only and also passed the notebook's internal hash check.
