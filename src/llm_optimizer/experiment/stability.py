"""Read-only comparison of recorded GPU measurements from lab runs.

JSONL is the measurement source; SQLite adds trial identity and failed-start
coverage. The two sources are joined, never counted as separate observations.
"""

import hashlib
import json
import math
import sqlite3
from collections import defaultdict
from contextlib import closing
from itertools import combinations
from pathlib import Path
from statistics import mean, median, stdev

METRICS = ("output_throughput", "p95_ttft_ms", "p95_e2e_latency_ms")
DEFAULT_TOLERANCES = {
    "max_num_seqs": 8,
    "max_running_requests": 8,
    "max_batch_size": 8,
    "max_num_batched_tokens": 2048,
    "chunked_prefill_size": 2048,
    "max_num_tokens": 2048,
    "max_prefill_tokens": 2048,
    "gpu_memory_utilization": 0.05,
    "mem_fraction_static": 0.05,
    "free_gpu_memory_fraction": 0.05,
    "schedule_conservativeness": 0.25,
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def normalize_config(value):
    """Decode JSON-valued native flags without assuming backend defaults."""
    if isinstance(value, dict):
        return {k: normalize_config(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_config(v) for v in value]
    if isinstance(value, str) and value.lstrip().startswith(("{", "[")):
        try:
            return normalize_config(json.loads(value))
        except json.JSONDecodeError:
            pass
    return value


def flatten(value, prefix=""):
    """Keep arrays intact: graph capture sizes are execution settings."""
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict) and item:
            result.update(flatten(item, name))
        else:
            result[name] = item
    return result


def _measurement_key(metrics):
    keys = (*METRICS, "completed", "total_requests", "failed")
    try:
        values = [float(metrics[k]) for k in keys]
    except (KeyError, TypeError, ValueError):
        return None
    return tuple(values) if all(math.isfinite(v) for v in values) else None


def _read_trials(run_dir):
    trials, index, issues = [], defaultdict(list), []
    database = run_dir / "studies.sqlite3"
    if not database.exists():
        issues.append(
            {
                "source": str(database),
                "issue": "No SQLite trial index; measurements will be unmatched",
            }
        )
        return trials, index, issues
    # Read-only SQLite access also prevents creation of an empty database.
    with closing(
        sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    ) as connection:
        studies = connection.execute(
            "SELECT study_id, study_name FROM studies"
        ).fetchall()
        for study_id, study_name in studies:
            attributes = {
                k: json.loads(v)
                for k, v in connection.execute(
                    "SELECT key, value_json FROM study_user_attributes "
                    "WHERE study_id=?",
                    (study_id,),
                )
            }
            contract = attributes.get("contract", {})
            if contract.get("phase") not in ("A", "B"):
                continue
            for trial_id, number, state, start, end in connection.execute(
                "SELECT trial_id, number, state, datetime_start, datetime_complete "
                "FROM trials WHERE study_id=? ORDER BY number",
                (study_id,),
            ):
                attrs = {
                    k: json.loads(v)
                    for k, v in connection.execute(
                        "SELECT key, value_json FROM trial_user_attributes "
                        "WHERE trial_id=?",
                        (trial_id,),
                    )
                }
                trial = {
                    "run_dir": str(run_dir),
                    "study_name": study_name,
                    "trial_number": number,
                    "trial_id": trial_id,
                    "phase": contract["phase"],
                    "framework": contract.get("framework"),
                    "concurrency": contract.get("concurrency"),
                    "target": contract.get("target"),
                    "sampler": contract.get("sampler"),
                    "trial_state": state,
                    "datetime_start": start,
                    "datetime_complete": end,
                    "failure_reason": attrs.get("failure_reason"),
                    "config": attrs.get("config"),
                    "measurement": attrs.get("measurement"),
                }
                trials.append(trial)
                key = _measurement_key(attrs.get("measurement") or {})
                if key is not None:
                    index[(study_name, key)].append(trial)
    return trials, index, issues


def _native_config(server_args, run_dir, source):
    config = normalize_config(server_args)
    warnings = []
    # TRT-LLM writes settings to a YAML flag. Resolve relocated artifacts by
    # filename within this run, then compare their contents rather than paths.
    for name in ("config", "extra_llm_api_options"):
        value = config.get(name)
        if not isinstance(value, str) or not value.endswith((".yaml", ".yml", ".json")):
            continue
        local = run_dir / Path(value).name
        if not local.is_file():
            local = Path(value)
        if not local.is_file():
            warnings.append(f"Cannot inspect native config file {value}")
            continue
        text = local.read_text()
        if local.suffix == ".json":
            payload = json.loads(text)
        else:
            try:
                import yaml
            except ImportError as exc:
                raise ImportError(
                    "Install llm-optimizer[notebook] to inspect TRT-LLM YAML settings"
                ) from exc
            payload = yaml.safe_load(text)
        if not isinstance(payload, dict):
            raise ValueError(
                f"Expected a mapping in native config file {local} ({source})"
            )
        config[name] = normalize_config(payload)
    return config, warnings


def _context(row):
    metadata = row["metadata"]
    workload = dict(row["client_args"])
    # Historical InfinityToNullEncoder wrote unlimited request rate as null.
    if workload.get("request_rate") is None:
        workload["request_rate"] = "inf"
    environment_fields = (
        "gpu_type",
        "gpu_count",
        "cuda_visible_devices",
        "model",
        "model_revision",
        "tokenizer",
        "tokenizer_revision",
        "framework_version",
        "cuda_version",
        "driver_version",
        "package_versions",
        "python_version",
        "platform",
    )
    identity = {
        "phase": row["phase"],
        "framework": row["framework"],
        "model": metadata.get("target_model", metadata.get("model_tag")),
        "workload": workload,
        "requests_sha256": metadata.get("requests_sha256"),
        "workload_sha256": metadata.get("workload_sha256"),
        "environment": {k: metadata.get(k) for k in environment_fields},
    }
    warnings = []
    for name in (
        "gpu_type",
        "gpu_count",
        "model_revision",
        "framework_version",
        "cuda_version",
        "driver_version",
    ):
        value = metadata.get(name)
        if value is None or "REPLACE" in str(value):
            warnings.append(f"{name} missing or placeholder")
    if not metadata.get("requests_sha256"):
        warnings.append("Exact request fingerprint missing")
    if not any(metadata.get(k) for k in ("gpu_uuid", "gpu_uuids", "gpu_serial")):
        warnings.append("Physical GPU identity not recorded")
    else:
        identity["physical_gpu"] = {
            k: metadata.get(k) for k in ("gpu_uuid", "gpu_uuids", "gpu_serial")
        }
    return identity, warnings


def load_measurements(run_dirs):
    """Return records, trial-state coverage, and source issues from saved runs."""
    records, trials, issues = [], [], []
    seen_paths = set()
    for directory in dict.fromkeys(Path(p).resolve() for p in run_dirs):
        if not directory.is_dir():
            raise ValueError(f"Run directory does not exist: {directory}")
        trial_rows, index, db_issues = _read_trials(directory)
        trials.extend(trial_rows)
        issues.extend(db_issues)
        files = sorted(directory.glob("*-measured.jsonl"))
        if not files:
            issues.append(
                {"source": str(directory), "issue": "No measured JSONL files found"}
            )
        for source in files:
            if source.resolve() in seen_paths:
                continue
            seen_paths.add(source.resolve())
            for line_number, line in enumerate(source.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                    config = raw["config"]
                    metadata = raw.get("metadata", {})
                    framework = config.get("framework")
                    phase = metadata.get("phase")
                    if phase not in ("A", "B") or framework not in (
                        "vllm",
                        "sglang",
                        "trtllm",
                    ):
                        issues.append(
                            {
                                "source": str(source),
                                "line": line_number,
                                "issue": "Unsupported or missing phase/framework",
                            }
                        )
                        continue
                    metrics = raw["results"]
                    client_args = config["client_args"]
                    native, config_warnings = _native_config(
                        config["server_args"], directory, source
                    )
                    concurrency = client_args.get("max_concurrency")
                    suffix = f"-{phase}-{framework}-c{concurrency}-measured.jsonl"
                    study_name = (
                        source.name[: -len(suffix)]
                        if source.name.endswith(suffix)
                        else source.stem
                    )
                    matches = index.get((study_name, _measurement_key(metrics)), [])
                    trial = matches[0] if len(matches) == 1 else {}
                    record = {
                        "observation_id": f"{source}:{line_number}",
                        "source_file": str(source),
                        "source_line": line_number,
                        "run_dir": str(directory),
                        "study_name": study_name,
                        "phase": phase,
                        "framework": framework,
                        "concurrency": concurrency,
                        "trial_number": trial.get("trial_number"),
                        "trial_state": trial.get(
                            "trial_state", "AMBIGUOUS" if matches else "UNMATCHED"
                        ),
                        "target": trial.get("target"),
                        "sampler": trial.get("sampler"),
                        "datetime_start": trial.get("datetime_start"),
                        "datetime_complete": trial.get("datetime_complete"),
                        "server_config": native,
                        "client_args": client_args,
                        "metadata": metadata,
                        "metrics": metrics,
                        "command": raw.get("cmd"),
                        "raw_record_hash": digest(raw),
                    }
                    context, context_warnings = _context(record)
                    record["context"] = context
                    record["context_id"] = digest(context)
                    record["config_id"] = digest(native)
                    record["exact_group_id"] = digest(
                        {"context": context, "config": native}
                    )
                    record["identity_warnings"] = config_warnings + context_warnings
                    record["native_config_resolved"] = not config_warnings
                    key = _measurement_key(metrics)
                    expected = client_args.get("num_prompts")
                    valid = (
                        key is not None
                        and metrics.get("benchmark_valid") is True
                        and metrics["failed"] == 0
                        and metrics["completed"] > 0
                        and metrics["completed"]
                        == metrics["total_requests"]
                        == expected
                        and all(metrics[m] >= 0 for m in METRICS)
                    )
                    record["measurement_valid"] = valid
                    record["exclusion_reason"] = (
                        None
                        if valid
                        else "Invalid metrics, incomplete requests, "
                        "or changed request budget"
                    )
                    records.append(record)
                except (KeyError, TypeError, ValueError, OSError) as exc:
                    issues.append(
                        {
                            "source": str(source),
                            "line": line_number,
                            "issue": f"{type(exc).__name__}: {exc}",
                        }
                    )
    return {"records": records, "trials": trials, "issues": issues}


def eligible_records(records, include_unmatched=False, exclude_duplicate_records=True):
    """Select successful measured trials; preserve exclusions for inspection."""
    selected, excluded, seen = [], [], set()
    for row in records:
        reason = row["exclusion_reason"]
        if not reason and row["trial_state"] != "COMPLETE":
            if not (include_unmatched and row["trial_state"] == "UNMATCHED"):
                reason = f"Trial state {row['trial_state']}"
        if not reason and not row["native_config_resolved"]:
            reason = "Native config file unavailable; settings cannot be compared"
        if not reason and exclude_duplicate_records and row["raw_record_hash"] in seen:
            reason = "Byte-equivalent JSON record already counted"
        if reason:
            excluded.append({**row, "exclusion_reason": reason})
        else:
            selected.append(row)
            seen.add(row["raw_record_hash"])
    return selected, excluded


def _statistics(values):
    center = median(values)
    average = mean(values)
    low, high = min(values), max(values)
    std = stdev(values) if len(values) > 1 else None
    deviations = [100 * (v - center) / center for v in values] if center else []
    return {
        "n": len(values),
        "min": low,
        "max": high,
        "median": center,
        "mean": average,
        "std": std,
        "cv_pct": 100 * std / abs(average) if std is not None and average else None,
        "spread_pct": 100 * (high - low) / abs(center) if center else None,
        "max_min_ratio": high / low if low > 0 else None,
        "max_abs_deviation_pct": max(map(abs, deviations)) if deviations else None,
    }


def exact_repeat_summary(records, min_repeats=2, alert_pct=10):
    """Summarize range and sample CV for identical executed configurations."""
    if (
        isinstance(min_repeats, bool)
        or not isinstance(min_repeats, int)
        or min_repeats < 2
    ):
        raise ValueError("min_repeats must be an integer of at least two")
    if not math.isfinite(alert_pct) or alert_pct < 0:
        raise ValueError("alert_pct must be finite and nonnegative")
    groups = defaultdict(list)
    for row in records:
        groups[row["exact_group_id"]].append(row)
    coverage, summaries, deviations = [], [], []
    for group_id, rows in groups.items():
        first = rows[0]
        common = {
            k: first[k]
            for k in ("phase", "framework", "concurrency", "context_id", "config_id")
        }
        common["exact_group_id"] = group_id
        coverage.append(
            {
                **common,
                "n": len(rows),
                "studies": sorted({r["study_name"] for r in rows}),
                "server_config": first["server_config"],
                "identity_warnings": first["identity_warnings"],
            }
        )
        if len(rows) < min_repeats:
            continue
        for metric in METRICS:
            stats = _statistics([r["metrics"][metric] for r in rows])
            worst = stats["max_abs_deviation_pct"]
            summaries.append(
                {
                    **common,
                    "metric": metric,
                    **stats,
                    "alert": worst is not None and worst > alert_pct,
                    "studies": sorted({r["study_name"] for r in rows}),
                    "server_config": first["server_config"],
                    "identity_warnings": first["identity_warnings"],
                }
            )
            for row in rows:
                value = row["metrics"][metric]
                pct = (
                    100 * (value - stats["median"]) / stats["median"]
                    if stats["median"]
                    else None
                )
                deviations.append(
                    {
                        **common,
                        "metric": metric,
                        "observation_id": row["observation_id"],
                        "study_name": row["study_name"],
                        "trial_number": row["trial_number"],
                        "value": value,
                        "group_median": stats["median"],
                        "deviation_pct": pct,
                        "abs_deviation_pct": abs(pct) if pct is not None else None,
                        "alert": pct is not None and abs(pct) > alert_pct,
                    }
                )
    return {"groups": coverage, "metrics": summaries, "deviations": deviations}


def parameter_differences(left, right, tolerances=None, max_changes=1):
    """Allow only bounded numeric changes; graph/categorical settings stay exact."""
    tolerances = DEFAULT_TOLERANCES if tolerances is None else tolerances
    if (
        isinstance(max_changes, bool)
        or not isinstance(max_changes, int)
        or max_changes < 1
    ):
        raise ValueError("max_changes must be a positive integer")
    if any(
        isinstance(v, bool)
        or not isinstance(v, (int, float))
        or not math.isfinite(v)
        or v <= 0
        for v in tolerances.values()
    ):
        raise ValueError("Numeric tolerances must be finite and positive")
    a, b = flatten(left), flatten(right)
    if a.keys() != b.keys():
        return None
    changes = []
    for name in sorted(a):
        x, y = a[name], b[name]
        if canonical(x) == canonical(y):
            continue
        tolerance = tolerances.get(name, tolerances.get(name.rsplit(".", 1)[-1]))
        if (
            tolerance is None
            or isinstance(x, bool)
            or isinstance(y, bool)
            or not isinstance(x, (int, float))
            or not isinstance(y, (int, float))
        ):
            return None
        difference = abs(x - y)
        if difference > tolerance + 1e-12:
            return None
        changes.append(
            {
                "parameter": name,
                "left": x,
                "right": y,
                "absolute_difference": difference,
                "tolerance": tolerance,
            }
        )
    return changes if 0 < len(changes) <= max_changes else None


def similar_parameter_pairs(records, tolerances=None, max_changes=1):
    """Compare medians of numeric neighbors; these are not repeat noise estimates."""
    contexts = defaultdict(lambda: defaultdict(list))
    for row in records:
        contexts[row["context_id"]][row["exact_group_id"]].append(row)
    pairs = []
    for groups in contexts.values():
        for left, right in combinations(groups.values(), 2):
            changes = parameter_differences(
                left[0]["server_config"],
                right[0]["server_config"],
                tolerances,
                max_changes,
            )
            if changes is None:
                continue
            common = {
                k: left[0][k]
                for k in ("phase", "framework", "concurrency", "context_id")
            }
            for metric in METRICS:
                x = _statistics([r["metrics"][metric] for r in left])
                y = _statistics([r["metrics"][metric] for r in right])
                denominator = (abs(x["median"]) + abs(y["median"])) / 2
                pairs.append(
                    {
                        **common,
                        "metric": metric,
                        "left_group_id": left[0]["exact_group_id"],
                        "right_group_id": right[0]["exact_group_id"],
                        "left_n": len(left),
                        "right_n": len(right),
                        "left_median": x["median"],
                        "right_median": y["median"],
                        "left_min": x["min"],
                        "left_max": x["max"],
                        "right_min": y["min"],
                        "right_max": y["max"],
                        "median_gap_pct": 100
                        * abs(x["median"] - y["median"])
                        / denominator
                        if denominator
                        else None,
                        "parameter_changes": changes,
                        "left_studies": sorted({r["study_name"] for r in left}),
                        "right_studies": sorted({r["study_name"] for r in right}),
                    }
                )
    return pairs
