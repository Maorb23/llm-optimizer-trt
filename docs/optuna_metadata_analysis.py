"""Offline adapters and statistics for optuna_run_metadata.ipynb."""

import html
import json
import math
import re

import pandas as pd

MODEL_ORDER = [
    "cpm5_2",
    "qwen3.5-4b",
    "qwen3-4b",
    "qwen3-4b dspark",
    "qwen3-8b",
    "qwen3-8b dspark",
    "gemma4-12b",
    "gemma4-12b dspark",
]
SUFFIX_MODELS = {
    "q3_4": "qwen3-4b",
    "q3_8": "qwen3-8b",
    "dsparkq3_4": "qwen3-4b dspark",
    "dsparkq3_8": "qwen3-8b dspark",
    "cmp_2": "cpm5_2",
    "cpm_2": "cpm5_2",
    "gemma4_12": "gemma4-12b",
    "dsparkg4_12": "gemma4-12b dspark",
    "q3.5_4": "qwen3.5-4b",
    "q35_4": "qwen3.5-4b",
    "dflashq3.5_4": "qwen3.5-4b dflash",
}


def suffix_model_group(name):
    for suffix in sorted(SUFFIX_MODELS, key=len, reverse=True):
        if re.search(r"(?:^|-)" + re.escape(suffix) + r"(?:-|$)", name.lower()):
            return SUFFIX_MODELS[suffix]
    return None


def model_group(name, contract):
    suffix_group = suffix_model_group(name)
    env = contract.get("environment") or {}
    model = str(
        env.get("hf_target_model_id")
        or contract.get("target_model")
        or contract.get("model")
        or env.get("model")
        or ""
    ).lower()
    draft = str(env.get("draft_model_id") or "").lower()
    if not model:
        return suffix_group or "unclassified"
    if "minicpm5" in model and re.search(r"(?:-|_)2b", model):
        base = "cpm5_2"
    elif "qwen3.5" in model:
        match = re.search(r"qwen3\.5[-_]([\d.]+)b", model)
        base = f"qwen3.5-{match.group(1)}b" if match else "unclassified"
    elif "qwen3" in model:
        match = re.search(r"qwen3[-_]([\d.]+)b", model)
        base = f"qwen3-{match.group(1)}b" if match else "unclassified"
    elif "gemma" in model and "12" in model:
        base = "gemma4-12b"
    else:
        return "unclassified"
    if "dspark" in draft or (suffix_group == base + " dspark" and not draft):
        return base + " dspark"
    if "dflash" in draft or (suffix_group == base + " dflash" and not draft):
        return base + " dflash"
    return base


def model_suffix_mismatch(name, contract):
    suffix_group = suffix_model_group(name)
    group = model_group(name, contract)
    if suffix_group and group != suffix_group:
        return f"Model/suffix mismatch: metadata={group}, name={suffix_group}"
    return None


def flatten(mapping, prefix=""):
    result = {}
    for key, value in (mapping or {}).items():
        path = prefix + key
        if isinstance(value, dict):
            result.update(flatten(value, path + "."))
        else:
            result[path] = value
    return result


def normalize_trial(row, metrics):
    """Accept both exported trials and raw W&B scan_history rows."""
    raw = flatten(row)
    measurement = dict(row.get("metrics") or row.get("measurement") or {})
    measurement.update(
        {
            k.removeprefix("benchmark/"): v
            for k, v in raw.items()
            if k.startswith("benchmark/") and v is not None
        }
    )
    params = dict(row.get("params") or {})
    params.update(
        {
            k.removeprefix("parameters/"): v
            for k, v in raw.items()
            if k.startswith("parameters/") and v is not None
        }
    )
    values = row.get("values")
    if values is None:
        objectives = row.get("objectives") or {}
        values = [
            objectives.get(
                m,
                raw.get(
                    "objective/" + m,
                    raw.get("trial/objective/" + str(i), measurement.get(m)),
                ),
            )
            for i, m in enumerate(metrics)
        ]
        if len(metrics) == 1 and values == [None]:
            values = [raw.get("trial/objective")]
    meta = row.get("trial_metadata") or {}
    number = row.get("number", raw.get("trial/number"))
    return {
        "number": number,
        "state": row.get("state", raw.get("trial/state")),
        "params": params,
        "values": values,
        "measurement": measurement,
        "execution": row.get("execution") or {},
        "config": row.get("effective_config", row.get("config")),
        "slo_feasible": row.get(
            "slo_feasible", meta.get("slo_feasible", raw.get("trial/slo_feasible"))
        ),
        "constraint_values": row.get(
            "constraint_values",
            meta.get("constraint_values", raw.get("trial/constraint_values")),
        ),
    }


def load_export(path, project):
    payload = json.loads(path.read_text())
    if payload.get("project") != project:
        raise ValueError("Unexpected W&B project")
    rows = []
    for row in payload["runs"]:
        config = row.get("config", row.get("params", {}))
        trials = row.get("trials", row.get("trial_history", []))
        # A numbered trial can have multiple logging rows. Keep its last state.
        by_number = {}
        for trial in trials:
            trial = normalize_trial(trial, config.get("metrics", []))
            if trial["number"] is not None:
                by_number[trial["number"]] = trial
        rows.append(
            {
                "id": row["id"],
                "name": row.get("name", row["id"]),
                "url": row.get("url", f"https://wandb.ai/{project}/runs/{row['id']}"),
                "state": row.get("state", row.get("run_state")),
                "group": row.get("group"),
                "config": config,
                "summary": row.get("summary", row.get("summary_metrics", {})),
                "history": list(by_number.values()),
            }
        )
    return rows


def numeric(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) and value >= 0 else None
    except (TypeError, ValueError):
        return None


def objective_statistics(frame):
    """Trial-weighted statistics; never mix summary-only with full histories."""
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for row in frame.to_dict("records"):
        if row["target"] not in {"throughput", "e2e", "pareto_throughput"}:
            continue
        for metric, value in zip(row["metrics"], row["values"]):
            if numeric(value) is not None:
                rows.append(
                    {
                        **{
                            k: row[k]
                            for k in [
                                "model_group",
                                "framework",
                                "phase",
                                "target",
                                "concurrency",
                                "coverage",
                                "run_identity",
                            ]
                        },
                        "metric": metric,
                        "value": float(value),
                    }
                )
    if not rows:
        return pd.DataFrame()
    long = pd.DataFrame(rows)
    long["coverage"] = long["coverage"].replace(
        {
            "all local COMPLETE trials": "full trial history",
            "all W&B COMPLETE trials": "full trial history",
        }
    )
    keys = ["model_group", "phase", "target", "concurrency", "coverage", "metric"]
    parts = []
    for contextual in [True, False]:
        for overall in [False, True]:
            data = long.assign(framework="overall") if overall else long
            if not contextual:
                data = data.assign(phase="all", concurrency="all")
            summary = (
                data.groupby(keys + ["framework"], dropna=False, sort=False)
                .agg(
                    n=("value", "size"),
                    runs=("run_identity", "nunique"),
                    mean=("value", "mean"),
                    variance=("value", "var"),
                )
                .reset_index()
            )
            summary["scope"] = "phase/concurrency" if contextual else "model"
            parts.append(summary)
    return pd.concat(parts, ignore_index=True)


def startup_rows(run, trials, source):
    """Preserve engine log observations separately from measured wall phases."""
    result = []
    for trial in trials:
        execution = flatten(trial.get("execution") or {})
        measured = flatten(trial.get("measurement") or {})
        execution.update(
            {
                k.removeprefix("execution."): v
                for k, v in measured.items()
                if k.startswith("execution.")
            }
        )
        common = {
            "model_group": run["model_group"],
            "framework": run["framework"],
            "phase": run["phase"],
            "study_name": run["study_name"],
            "wandb_id": run.get("wandb_id"),
            "wandb_url": run.get("wandb_url"),
            "trial": trial.get("number"),
            "state": trial.get("state"),
            "params": trial.get("params") or {},
            "source": source,
        }
        for phase in [
            "launch_to_ready",
            "client_preparation",
            "warmup",
            "benchmark",
            "runner_wall",
        ]:
            value = numeric(execution.get("timings_s." + phase))
            status = execution.get("phase_status." + phase, "unknown")
            if value is not None:
                result.append(
                    dict(
                        common,
                        timing=phase,
                        seconds=value,
                        phase_status=status,
                        observation=None,
                        timing_source="recorded wall clock",
                    )
                )
        for index, observation in enumerate(
            execution.get("startup_observations") or []
        ):
            value = numeric(observation.get("seconds"))
            if value is not None:
                result.append(
                    dict(
                        common,
                        timing=observation.get("phase", "unknown"),
                        seconds=value,
                        phase_status="log observation",
                        observation=index,
                        timing_source=observation.get("source", "engine log"),
                    )
                )
    return result


def startup_statistics(frame):
    if frame.empty:
        return pd.DataFrame()
    # Partial timeouts are retained in the raw table, excluded from duration means.
    frame = frame[frame["phase_status"].isin(["complete", "log observation"])]
    keys = ["model_group", "phase", "timing", "phase_status"]
    parts = []
    for contextual in [True, False]:
        for overall in [False, True]:
            data = frame.assign(framework="overall") if overall else frame
            if not contextual:
                data = data.assign(phase="all")
            summary = (
                data.groupby(keys + ["framework"], dropna=False, sort=False)
                .agg(
                    n=("seconds", "size"),
                    mean_seconds=("seconds", "mean"),
                    variance_seconds2=("seconds", "var"),
                )
                .reset_index()
            )
            summary["scope"] = "phase" if contextual else "model"
            parts.append(summary)
    return pd.concat(parts, ignore_index=True)


def parameters_html(params):
    return (
        "<br>".join(
            f"<b>{html.escape(str(k))}</b>: "
            f"{html.escape(json.dumps(v, ensure_ascii=False, default=str))}"
            for k, v in sorted(params.items())
        )
        or "—"
    )


def startup_trial_table(frame):
    """One parameter row per trial, retaining repeated log observations."""
    if frame.empty:
        return pd.DataFrame()
    rows = []
    keys = ["model_group", "framework", "phase", "study_name", "wandb_id", "trial"]
    for identity, group in frame.groupby(keys, dropna=False, sort=False):
        row = dict(zip(keys, identity))
        row.update(
            params=group.iloc[0]["params"],
            state=group.iloc[0]["state"],
            source=group.iloc[0]["source"],
            wandb_url=group.iloc[0]["wandb_url"],
        )
        for timing, observations in group.groupby("timing", sort=False):
            values = observations["seconds"].tolist()
            row[timing + " (s)"] = values[0] if len(values) == 1 else values
            row[timing + " status"] = ", ".join(
                sorted(set(observations["phase_status"]))
            )
        rows.append(row)
    return pd.DataFrame(rows)


def table_html(frame, parameter_columns=("params",)):
    """Escape all data before enabling HTML for explicit parameter line breaks."""
    safe = frame.copy().astype(object)
    for column in safe:
        if column in parameter_columns:
            safe[column] = safe[column].map(
                lambda p: parameters_html(p) if isinstance(p, dict) else "—"
            )
        else:
            safe[column] = safe[column].map(
                lambda v: (
                    "—"
                    if v is None or (isinstance(v, float) and math.isnan(v))
                    else html.escape(str(v))
                )
            )
    return (
        '<div style="overflow-x:auto;max-height:650px;overflow-y:auto">'
        + safe.to_html(index=False, escape=False)
        + "</div>"
    )


def write_report_html(
    path,
    models,
    inventory,
    objectives,
    timing_summary,
    timing_details,
    timing_coverage,
    importances,
    importance_coverage,
    excluded,
    issues,
    exported_at=None,
):
    """Save the notebook's tables as a standalone report with readable parameters."""
    parts = [
        '<!doctype html><html><head><meta charset="utf-8">',
        "<title>Optuna run metadata</title><style>",
        "body{font:15px system-ui;margin:2rem;color:#17202a}",
        "table{border-collapse:collapse;font-size:13px}",
        "td,th{padding:8px;border:1px solid #ddd;text-align:left;vertical-align:top}",
        "th{position:sticky;top:0;background:#eff3f7}",
        "td{min-width:90px;max-width:450px;overflow-wrap:anywhere}",
        "a{color:#175ec0}h2{margin-top:3rem}",
        "nav{display:flex;gap:16px;flex-wrap:wrap}</style></head><body>",
        "<h1>Optuna run metadata by model</h1>",
        f"<p>W&amp;B cache exported at {html.escape(str(exported_at))}.</p>",
        "<p>Objective statistics use measured runs lasting at least 300 seconds. "
        "Timing coverage also includes shorter measured runs. Variance is sample "
        "variance (ddof=1); singleton variance is unavailable. Objective units "
        "are tokens/s and milliseconds. Timings use seconds and variance seconds².</p>",
        "<p>One objective observation is one eligible completed trial. Full trial "
        "histories and best/Pareto-only summaries remain separate. Model totals "
        "pool phases and concurrencies; contextual rows retain them. Differing "
        "workloads and search budgets can affect descriptive means. Both metrics "
        "of throughput/E2E Pareto studies are shown.</p>",
        "<p>Constrained studies include only SLO-feasible objective results. "
        "Timing tables retain all recorded trial states.</p>",
        "<p>Initiation is launch-to-ready, including model loading. Engine log "
        "loading and compilation observations can overlap; they are never added "
        "to wall time. Partial timings appear in trial rows and are excluded "
        "from means. Repeated log durations appear as lists. Missing telemetry "
        "is unavailable. Optuna PED-ANOVA importance is computed per run and "
        "objective from complete trials; conditional parameters absent from "
        "some trials do not enter the shared search space.</p>",
        "<nav>"
        + "".join(
            f'<a href="#m{i}">{html.escape(model)}</a>'
            for i, model in enumerate(models)
        )
        + "</nav>",
    ]

    def table(frame):
        return table_html(frame) if not frame.empty else "<p>No data available.</p>"

    def select(frame, model):
        return frame[frame["model_group"] == model] if not frame.empty else frame

    for i, model in enumerate(models):
        parts.append(f'<section id="m{i}"><h2>{html.escape(model)}</h2>')
        for title, frame in [
            ("Runs", inventory),
            ("Objective mean and variance", objectives),
            ("Server timing mean and variance", timing_summary),
            ("Initiation metadata coverage per run", timing_coverage),
        ]:
            parts.append(f"<h3>{title}</h3>" + table(select(frame, model)))
        parts.append(
            "<h3>Trial timings and parameters</h3>"
            + table(startup_trial_table(select(timing_details, model)))
        )
        for framework in ["vllm", "trtllm", "sglang"]:
            parts.append(f"<h3>{framework} · Optuna feature importance</h3>")
            for frame in [importances, importance_coverage]:
                selected = select(frame, model)
                if not selected.empty:
                    selected = selected[selected["framework"] == framework]
                parts.append(table(selected))
        parts.append("</section>")
    parts.append("<h2>Excluded objective runs</h2>" + table(excluded))
    parts.append("<h2>Source issues</h2>" + table(issues))
    parts.append("</body></html>")
    path.write_text("\n".join(parts), encoding="utf-8")


def history_importance(run, trials):
    """Reconstruct the lab's numeric grids and categorical search dimensions."""
    import optuna

    contract = run["contract"]
    metrics = contract.get("metrics", [])
    usable = [
        t
        for t in trials
        if t.get("state") == "COMPLETE"
        and len(t.get("values") or []) == len(metrics)
        and all(numeric(v) is not None for v in t["values"])
    ]
    if len(usable) < 2:
        return [], "fewer than two complete trials"
    menu = contract.get("sampling_menu") or {}
    numeric_grids = dict(contract.get("phase_a_space") or {})
    # Expanded lab dimensions sampled numerically outside phase_a_space.
    for param in ["graph_padding", "schedule_conservativeness", "max_prefill_tokens"]:
        if param in menu:
            numeric_grids[param] = menu[param]
    common = set.intersection(*(set(t["params"]) for t in usable)) - {"cuda_graph_on"}
    choices = {}
    for param in sorted(common):
        values = list(menu.get(param) or [])
        for trial in usable:
            value = trial["params"][param]
            if not any(type(value) is type(v) and value == v for v in values):
                values.append(value)
        if all(v is None or isinstance(v, (str, bool, int, float)) for v in values):
            choices[param] = values
    distributions = {}
    for param, values in choices.items():
        grid = numeric_grids.get(param)
        if grid and len(grid) > 1:
            step = grid[1] - grid[0]
            regular = step > 0 and all(
                math.isclose(v, grid[0] + i * step, abs_tol=1e-8)
                for i, v in enumerate(grid)
            )
            if regular and all(isinstance(v, int) for v in grid):
                distributions[param] = optuna.distributions.IntDistribution(
                    grid[0], grid[-1], step=step
                )
            elif regular:
                distributions[param] = optuna.distributions.FloatDistribution(
                    grid[0], grid[-1], step=round(step, 8)
                )
        distributions.setdefault(
            param, optuna.distributions.CategoricalDistribution(values)
        )
    study = optuna.create_study(
        directions=[str(direction).lower() for direction in contract["directions"]]
    )
    for trial in usable:
        params = {k: trial["params"][k] for k in choices}
        study.add_trial(
            optuna.trial.create_trial(
                params=params,
                distributions=distributions,
                values=trial["values"],
            )
        )
    output = []
    for index, metric in enumerate(metrics):
        if len({t["values"][index] for t in usable}) < 2:
            continue
        sign = -1 if contract["directions"][index].upper() == "MAXIMIZE" else 1
        scores = optuna.importance.get_param_importances(
            study,
            evaluator=optuna.importance.PedAnovaImportanceEvaluator(),
            target=lambda t, i=index, s=sign: s * t.values[i],
        )
        for rank, (param, score) in enumerate(scores.items(), 1):
            output.append(
                {
                    "model_group": run["model_group"],
                    "framework": run["framework"],
                    "target": contract.get("target"),
                    "sampler": contract.get("sampler"),
                    "run": run["study_name"],
                    "wandb_id": run.get("wandb_id"),
                    "metric": metric,
                    "rank": rank,
                    "parameter": param,
                    "importance": score,
                    "evaluator": "Optuna PED-ANOVA",
                    "source": "W&B trial history",
                }
            )
    return output, None if output else "no varying objective/shared parameters"
