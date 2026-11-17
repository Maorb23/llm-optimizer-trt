#!/usr/bin/env python3
"""Compare completed W&B studies without launching servers or changing runs.

Run in the remote notebook environment with W&B authenticated:
  python tools/investigate_sglang_runs_v3.py
Optional: --runs ENTITY/PROJECT/ID ... --out /tmp/sglang-run-comparison
Exports exact downloaded configs, summaries and trial history for review.
No API keys or environment credentials are printed or exported.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_RUNS = [
    "maorblumberg-tel-aviv-university/inference-optimization/ijtqgc9e",
    "maorblumberg-tel-aviv-university/inference-optimization/97ru6g0k",
    "maorblumberg-tel-aviv-university/inference-optimization/6o0eengf",
]
E2E = "p95_e2e_latency_ms"


def clean(value):
    if isinstance(value, dict):
        return {
            str(k): clean(v)
            for k, v in value.items()
            if not any(
                x in str(k).lower()
                for x in ("api_key", "password", "secret", "access_token")
            )
        }
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def canonical(value):
    return json.dumps(clean(value), sort_keys=True, separators=(",", ":"))


def finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def flatten(value, prefix=""):
    result = {}
    if isinstance(value, dict) and value:
        for key, item in value.items():
            result.update(flatten(item, f"{prefix}/{key}" if prefix else str(key)))
    else:
        result[prefix] = value
    return result


def history_trials(history):
    """Merge repeated rows for the same trial, never forward-fill between trials."""
    trials = {}
    for row in history:
        number = row.get("trial/number")
        if not finite(number):
            continue
        number = int(number)
        trials.setdefault(number, {}).update(row)
    for number, row in trials.items():
        row["trial/number"] = number
    return [trials[number] for number in sorted(trials)]


def metric(row, name):
    # The objective and benchmark carry the same final metric for scalar E2E studies.
    for key in (f"benchmark/{name}", f"objective/{name}"):
        value = row.get(key)
        if finite(value):
            return float(value)
    benchmark = row.get("benchmark")
    if isinstance(benchmark, dict) and finite(benchmark.get(name)):
        return float(benchmark[name])
    return None


def params(row):
    return {
        key[len("parameters/") :]: value
        for key, value in row.items()
        if key.startswith("parameters/")
    }


def native_config(row):
    value = row.get("trial/config")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    if isinstance(value, dict) and value:
        return value
    # Handle a flattened W&B representation as well.
    values = {
        key[len("trial/config/") :]: value
        for key, value in row.items()
        if key.startswith("trial/config/")
    }
    return values or None


def eligible(run):
    rows = [
        row
        for row in run["trials"]
        if row.get("trial/state") == "COMPLETE" and metric(row, E2E) is not None
    ]
    if run["config"].get("constrained"):
        rows = [row for row in rows if row.get("trial/slo_feasible") is True]
    return rows


def format_value(value, limit=150):
    text = canonical(value)
    return text if len(text) <= limit else text[:limit] + "... [full value in export]"


def print_table(rows, columns):
    if not rows:
        print("  No data.")
        return
    text = [
        [
            str(row.get(key))
            if not isinstance(row.get(key), float)
            else f"{row[key]:.3f}"
            for key, _ in columns
        ]
        for row in rows
    ]
    widths = [
        max(len(label), *(len(row[i]) for row in text))
        for i, (_, label) in enumerate(columns)
    ]
    print(
        "  "
        + " | ".join(label.ljust(widths[i]) for i, (_, label) in enumerate(columns))
    )
    for row in text:
        print("  " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(row)))


def compare(runs, out, top=3):
    print("\nRUN SUMMARY — E2E values are per-trial p95 latency, milliseconds")
    summaries = []
    for run in runs:
        config = run["config"]
        rows = eligible(run)
        best = min(rows, key=lambda row: metric(row, E2E)) if rows else None
        e2es = [metric(row, E2E) for row in rows]
        summaries.append(
            {
                "id": run["id"],
                "state": run["state"],
                "seed": config.get("sampler_settings", {}).get("seed"),
                "concurrency": config.get("concurrency"),
                "sampler": config.get("sampler"),
                "attempts": len(run["trials"]),
                "eligible": len(rows),
                "best_trial": best.get("trial/number") if best else None,
                "best_e2e": min(e2es) if e2es else None,
                "median_trial_e2e": statistics.median(e2es) if e2es else None,
                "last_logged_best": run["summary"].get(f"best/{E2E}"),
            }
        )
        print(f"\n{run['id']}: {run['name']} | {run['path']}")
        print(
            "  Trial states:",
            dict(Counter(row.get("trial/state", "UNKNOWN") for row in run["trials"])),
        )
        print(
            "  Target:",
            config.get("target"),
            "metrics:",
            config.get("metrics"),
            "directions:",
            config.get("directions"),
            "constrained:",
            config.get("constrained"),
        )
        print(
            "  Pruner:",
            format_value(config.get("pruner", config.get("pruner_settings"))),
        )
        if not rows:
            print("  WARNING: no eligible E2E values. Inspect exported history.")
        if run["state"] != "finished":
            print("  WARNING: run is not finished; its best value may still change.")
    print()
    print_table(summaries, [(key, key) for key in summaries[0]])

    print("\nCONTRACT DIFFERENCES — optimizer randomness versus other changes")
    flats = [flatten(run["config"]) for run in runs]
    ignored = {"study_name", "sampler_settings/seed", "_wandb"}
    differences = []
    for key in sorted(set().union(*(set(flat) for flat in flats))):
        if key in ignored or key.startswith("_wandb/"):
            continue
        values = [flat.get(key, "<MISSING>") for flat in flats]
        if len({canonical(v) for v in values}) > 1:
            differences.append(
                {"field": key, **{run["id"]: value for run, value in zip(runs, values)}}
            )
            print(f"  {key}")
            for run, value in zip(runs, values):
                print(f"    {run['id']}: {format_value(value)}")
    if not differences:
        print("  No logged contract differences besides study name and optimizer seed.")
    for field in (
        "environment/requests_sha256",
        "environment/framework_version",
        "environment/model_revision",
        "environment/gpu_type",
        "workload/seed",
    ):
        print(
            f"  CHECK {field}: "
            + " | ".join(
                f"{run['id']}={format_value(flat.get(field, '<MISSING>'))}"
                for run, flat in zip(runs, flats)
            )
        )
    print("  Matching metadata does not prove identical GPU conditions.")

    trial_exports = []
    winners = []
    print("\nTRIAL HISTORIES — no partial/pruned E2E is treated as a completed result")
    for run in runs:
        displayed = []
        best_so_far = None
        for row in run["trials"]:
            e2e = metric(row, E2E)
            is_eligible = row in eligible(run)
            if is_eligible:
                best_so_far = min(best_so_far, e2e) if best_so_far is not None else e2e
            record = {
                "run": run["id"],
                "trial": row["trial/number"],
                "state": row.get("trial/state"),
                "e2e_p95_ms": e2e,
                "eligible": is_eligible,
                "best_so_far_ms": best_so_far,
                "ttft_p95_ms": metric(row, "p95_ttft_ms"),
                "tokens_per_s": metric(row, "output_throughput"),
                "rps": metric(row, "request_throughput"),
                "failure_reason": row.get("trial/failure_reason"),
                "config_id": row.get("trial/config_id"),
                **{f"param/{k}": value for k, value in params(row).items()},
            }
            displayed.append(record)
            trial_exports.append(record)
        print(f"\n  {run['id']}")
        print_table(
            displayed,
            [
                (key, key)
                for key in (
                    "trial",
                    "state",
                    "e2e_p95_ms",
                    "best_so_far_ms",
                    "ttft_p95_ms",
                    "tokens_per_s",
                )
            ],
        )
        best_rows = sorted(eligible(run), key=lambda row: metric(row, E2E))[:top]
        for rank, row in enumerate(best_rows, 1):
            print(
                f"  Top {rank}: trial {row['trial/number']}, "
                f"E2E={metric(row, E2E):.3f} ms"
            )
            print("    Sampled:", canonical(params(row)))
            config = native_config(row)
            print("    Rendered:", canonical(config))
            if rank == 1:
                winners.append((run, row, config))
            telemetry = row.get("benchmark/server_telemetry")
            if not telemetry:
                execution = row.get("benchmark/execution") or {}
                if isinstance(execution, dict):
                    telemetry = execution.get("server_telemetry")
            print("    Server telemetry:", canonical(telemetry))

    print("\nBEST-CONFIGURATION DIFFERENCES")
    for key in sorted(
        set().union(*(set(flatten(config or {})) for _, _, config in winners))
    ):
        values = [
            flatten(config or {}).get(key, "<MISSING>") for _, _, config in winners
        ]
        if len({canonical(value) for value in values}) > 1:
            print(
                " ",
                key,
                " | ".join(
                    f"{run['id']}={format_value(value)}"
                    for (run, _, _), value in zip(winners, values)
                ),
            )
    if any(config is None for _, _, config in winners):
        print(
            "  WARNING: a best trial has no rendered config; comparison is incomplete."
        )

    print("\nIDENTICAL RENDERED CONFIGURATIONS ACROSS RUNS")
    groups = defaultdict(list)
    for run in runs:
        for row in eligible(run):
            config = native_config(row)
            if config:
                groups[canonical(config)].append(
                    (run["id"], row["trial/number"], metric(row, E2E))
                )
    matched = []
    for config, observations in groups.items():
        if len({run_id for run_id, _, _ in observations}) < 2:
            continue
        matched.append({"config": json.loads(config), "observations": observations})
        print(
            "  Config", hashlib.sha256(config.encode()).hexdigest()[:12], observations
        )
        values = [value for _, _, value in observations]
        if min(values) > 0:
            print(f"    Slowest/fastest ratio: {max(values) / min(values):.3f}")
    if not matched:
        print(
            "  None found. These histories do not isolate run-to-run timing variation."
        )
    else:
        print(
            "  Check workload/environment differences before comparing these timings."
        )

    out.mkdir(parents=True, exist_ok=True)
    for run in runs:
        (out / f"{run['id']}.json").write_text(json.dumps(clean(run), indent=2) + "\n")
    (out / "comparison.json").write_text(
        json.dumps(
            clean(
                {
                    "summaries": summaries,
                    "contract_differences": differences,
                    "matched_configurations": matched,
                }
            ),
            indent=2,
        )
        + "\n"
    )
    columns = list(dict.fromkeys(key for row in trial_exports for key in row))
    with (out / "trials.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in trial_exports:
            writer.writerow(
                {
                    key: canonical(value) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )
    print("\nExports:", out.resolve())
    print(
        "Interpretation: different optimizer seeds explore different configurations. "
        "Lower best E2E can reflect better configurations rather than timing noise. "
        "Repeat identical configs on the same workload to quantify timing variability."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    parser.add_argument("--out", type=Path, default=Path("run_investigation_sglang_v3"))
    parser.add_argument("--top", type=int, default=3)
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top must be positive")
    try:
        import wandb
    except ImportError:
        parser.exit(
            1,
            "W&B SDK missing. Run in your authenticated notebook environment.\n",
        )
    api = wandb.Api(timeout=120)
    runs = []
    for path in args.runs:
        print("Reading", path, flush=True)
        try:
            run = api.run(path)
            runs.append(
                {
                    "path": path,
                    "id": run.id,
                    "name": run.name,
                    "state": run.state,
                    "config": clean(dict(run.config)),
                    "summary": clean(dict(run.summary)),
                    "trials": clean(history_trials(run.scan_history(page_size=1000))),
                }
            )
        except Exception as exc:
            # SDK errors can include request metadata; keep credentials out of reports.
            parser.exit(
                1,
                f"Cannot read {path}: {type(exc).__name__}. "
                "Check W&B login, project access and network connectivity.\n",
            )
    compare(runs, args.out, args.top)


if __name__ == "__main__":
    main()
