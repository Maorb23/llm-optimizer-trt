# Implementation plan: restructure the current-results notebook

## Target and reference

Modify `docs/current_results_by_vm_framework_model.ipynb`. Use the actual rendering in `docs/optuna_run_metadata_v2.ipynb` as the layout reference, with the additions below. Do not modify the reference notebook or experiment artifacts.

The reference was inspected, including `collapsible_table` (cell 3), `objective_result_table` (cell 38), `show_objective` (48), `show_concurrency` (50), `show_framework` (52), and the summary, per-trial, startup, and model rendering functions in cell 56. Its model sections repeat **Results Summary**, **Per-Trial behavior**, and **Overall results and plots**.

Important dependency finding: v2 imports `docs/optuna_metadata_analysis.py`, but that file is absent from this checkout. Do not assume v2 can execute unchanged. Recover that helper if available in repository history, or implement the required functions in a small helper for the current notebook. Preserve the reference's presentation without importing unavailable code.

## 1. Required report hierarchy

```text
Initial runs                         # verified full run name does not contain invest
  Model: exact identifier / variant
    Framework: SGLang | vLLM | TensorRT-LLM
      Concurrency: 1, 8, 16, 32, 128, ... (actual collected values)
        Results Summary
          Notebook version / phase / objective / compatible context
          Coverage, objective statistics, up to five best runs
          Startup and serving timing statistics
        Per-Trial behavior
          Objective by trial number
          Server initiation by trial number
        Overall results and plots
          Compatible-context objective distributions and progress
          fANOVA and PED-ANOVA
        [Show all runs — closed by default]
          Complete run list, results, parameters, individual trial details,
          startup observations, importance estimates/skip reasons, provenance

Parameters probe                     # verified full run name contains invest
  Model
    Framework
      Concurrency
        Same blocks and toggle behavior as above

Audit and provenance
  Raw/deduplicated coverage, duration exclusions, empty artifacts,
  unresolved run-name classification, missing metadata and source issues
```

The visible order is **section → model → framework → concurrency**. Do not put framework above model, or phase above concurrency. Keep notebook versions, phases, objectives, and compatibility contexts separate inside each concurrency subsection. Do not introduce a single automatically selected context that hides the other groups.

Use native HTML `<details><summary>Show all runs (N)…</summary>…</details>` without an `open` attribute. Full run listings, individual run/trial tables, detailed timing/importance tables, and provenance must not also be rendered outside their toggles. Keep summary tables and aggregate plots visible, matching v2's report style. Escape metadata and wrap long parameter/configuration values as v2 does.

## 2. Recover actual run names and classify the runs

The current local Optuna study name typically omits the W&B display-name prefix. It is insufficient to distinguish `invest` from initial runs.

- Enrich the existing collected study records by matching their stored W&B IDs to cached W&B snapshots/exports and saved executed-notebook evidence. Full names such as `fp8-clean-invest-B-vllm-c8-throughput-tpe-median-s29` contain both the probe marker and framework.
- Preserve `display_name`, `name_source`, W&B ID/URL, source collection, and any conflicting name evidence. Resolve conflicting exports by documented authoritative/freshness rules, not filesystem iteration order.
- Apply the rule to the verified full name using a case-insensitive `invest` substring check. A known name without `invest` is **Initial runs**; a known name containing it is **Parameters probe**.
- A missing full name is **unclassified**, not automatically an initial run. Show its study name, ID, source, and missing evidence in the audit. Use an explicit, auditable override map if necessary. Do not classify from v2/v3 folder names, model suffixes, or a notebook's current `RUN_NAME` control alone.
- Cached exports outside the requested collection may enrich names or matching telemetry only for existing collected IDs. Do not silently add their unrelated historical runs to this report. Any live refresh must be optional, read-only, and recorded in provenance.
- Perform artifact reconciliation before classification. Copies of the same experiment must not appear in both sections because different downloads supply different metadata.

## 3. Preserve the existing analysis rules

Keep the current data roots: `runs/20261117-143416/` and `experiments/notebooks/20261117-143416/`. Retain configurable repository-relative paths, renamed and legacy folder support, read-only SQLite access, exact model/revision/quantization distinctions, metadata mismatch reporting, and explicit uncertain VM provenance.

Retain content/provenance-based deduplication and separate counts for folders, databases, studies, trials, and measured attempts. Keep genuine repeated evaluations. Empty databases remain inventory artifacts, not experiment runs with invented durations.

The **300-second study/run filter applies to every main summary, ranking, comparison, timing aggregate, and importance calculation**. Keep exactly 300 seconds. Unknown duration and shorter runs stay in the audit. Unlike v2's startup exception, do not pool their timing observations into the main timing summaries. Preserve the recorded duration source and existing fallback rules.

All compatibility keys must include the new report section as well as model variant, actual framework, concurrency, notebook version, phase, workload/request count, objective/directions/constraints, hardware/revision, and materially different search spaces. Preserve existing uncertain-VM isolation. Presentation nesting must not weaken comparison rules.

## 4. Reproduce v2's Results Summary and add top five

For every section/model/framework/concurrency/version/phase/objective/compatible-context group:

- Display completed, failed, pruned, unfinished and usable trial counts, run/seed/sampler coverage, and missing-result counts.
- Report objective observation count, contributing runs, mean, and sample variance (`ddof=1`), with units and optimization direction. Singleton variance is unavailable, not zero. Keep full histories separate from best-only/candidate-only data.
- Show **up to five distinct eligible experiment runs**, using each scalar study's best valid completed trial. Sort by its stored objective direction; use a deterministic identity tie-break. Include full run name, study/W&B identity, sampler, pruner, seed, winning trial number, objective value/unit, duration, and readable parameters.
- If fewer than five suitable studies exist, show the available count. Do not fill the table with multiple trials from one study or rank other workloads together.
- For Pareto objectives, preserve paired objective values and nondominated candidates. Do not invent a scalar ordering to force a top-five leaderboard. Show a clearly labeled frontier preview and put the full frontier inside the toggle.
- Keep all remaining eligible runs available in the closed detail table; do not discard them from statistics or plots.

Use the observed model identifiers in this collection rather than copying v2's hardcoded model list or its model aliases.

## 5. Reproduce per-trial behavior and startup timing

Port the intent of v2's `mean_by_trial_number`, startup coverage, timing statistics, and trial timing/parameter tables, adding section and concurrency filtering throughout.

- Extract local execution data from `trial_user_attributes.execution` and `trial_user_attributes.measurement.execution`; the inspected collection contains the latter with `timings_s`, `phase_status`, `attempt_id`, and `startup_observations`.
- Matching JSONL attempt artifacts or ID-matched full W&B histories may supplement missing telemetry. Join using recorded attempt/study/trial identities; never infer trial correspondence from JSONL row order. Deduplicate copies without collapsing real checkpoint/retry attempts.
- Include launch-to-ready, client preparation, warmup, benchmark, runner wall, and other explicitly recorded timing phases. Report seconds and variance in seconds squared. Launch-to-ready includes server initiation and model loading.
- Keep explicit weight-loading, compilation, graph-capture, and autotuning log observations separate from wall-clock phases. They can overlap; do not sum them to estimate startup time. Repeated observations remain identifiable.
- Exclude partial/incomplete timing phases from means while retaining their status and reason in the detailed tables. Missing timings remain unavailable; do not derive them from optimization duration.
- Plot mean objective and mean launch-to-ready by original trial number within compatible groups, with observation and contributing-run counts at each point. Keep seeds/samplers identifiable and do not forward-fill failed or missing trials.
- Clearly distinguish a trial-level statistic from an attempt-level statistic. Specify how multiple attempts for one trial contribute to any displayed mean.

## 6. Retain both importance methods and contextual plots

Keep fANOVA (`FanovaImportanceEvaluator`) and PED-ANOVA (`PedAnovaImportanceEvaluator`) side by side. Do not regress to v2's PED-only implementation.

Compute per suitable study/objective within its eligible section/context, using in-memory reconstructed Optuna studies. Retain explicit handling of conditional/changing distributions, constant parameters or targets, unsupported distributions, insufficient completed candidates, and computation failures. Missing importance is not zero. Any parameter exclusions from the reference, such as `cuda_graph_on`, must be explicit and audited rather than silently applied.

Keep objective history/incumbent curves, compatible distribution plots, sampler comparisons and seed variation, nested under the correct section/model/framework/concurrency. Do not copy v2's pooling across phases, concurrencies, or incompatible frameworks into the new aggregates.

## 7. Separate calculation from rendering

Load and normalize sources, reconcile identities/names, apply eligibility, and compute objective/timing/importance tables once. Then render through a small hierarchy:

```python
show_section(section)
show_model(section, model)
show_framework(section, model, framework)
show_concurrency(section, model, framework, concurrency)
```

Pass the complete group selection to every summary/detail/plot function; avoid renderers that re-read an unfiltered global frame. Reuse v2's formatting patterns and existing repository utilities where they are available. A small helper module is preferable to duplicating large rendering cells. An HTML export is optional; if retained, it must use the same hierarchy, filters, and closed toggles as the notebook.

## 8. Validation and completion criteria

1. Add focused checks for name classification (including shared SGLang/vLLM prefixes, missing/conflicting names), hierarchy ordering, concurrency isolation, top-five distinct-study ranking/direction, Pareto handling, timing status/deduplication, and the 299/300/unknown duration boundary.
2. Inspect rendered HTML: all detailed runs/trials are inside closed toggles; both requested sections exist; each contains model → framework → concurrency; summaries and timing/per-trial blocks remain visible. Check empty groups and missing telemetry explicitly.
3. Execute the modified notebook from top to bottom. Confirm both importance methods still run or report explicit skips. Compare source database hashes before/after.
4. Reconcile coverage against the previous execution, as reference values rather than hardcoded assertions: 134 source folders/databases, 423 raw studies, 8,137 raw trials; 124 distinct artifacts, 388 distinct studies, 7,602 trials; 371 eligible studies, 7,417 trials and 5,868 completed trials. The previous inventory had 86 empty databases and 17 duration-excluded studies (11 short, six unknown). Classification should partition eligible coverage into initial/probe/unclassified without changing total counts by itself.
5. Explain any changes from newly reconciled data. Report run counts for each section, unresolved name assignments, exclusions, timing coverage, and importance coverage. Identify any execution/dependency blockers accurately.

Deliver the executed `docs/current_results_by_vm_framework_model.ipynb`, any required small helper/tests, and a concise change/validation report. Do not launch GPU experiments, rename W&B runs, or modify source results.
