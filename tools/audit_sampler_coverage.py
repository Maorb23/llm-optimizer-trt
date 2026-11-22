"""Read-only GP coverage audit, using collected SQLite studies or live W&B.

Missing means not observed in this source, not proof an experiment never ran.
Contexts deliberately omit sampler/pruner and pruning checkpoint schedules, so
TPE+Median and GP+Nop can be paired; model/revision/workload/search space stay fixed.
Unknown VM identity is explicit: matching hardware metadata does not prove a VM.
"""

import argparse
from collections import Counter, defaultdict
from contextlib import closing
import csv
import hashlib
import json
from pathlib import Path
import sqlite3


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def context(contract):
    fields = (
        "framework", "phase", "target_model", "concurrency", "workload", "target",
        "metrics", "directions", "constrained", "ttft_limit_ms", "e2e_limit_ms",
        "space_kind", "phase_a_space", "expanded_choices", "sampling_menu",
        "budget_trials", "budget_gpu_seconds",
    )
    environment = contract.get("environment", {})
    env_fields = (
        "gpu_type", "gpu_count", "framework_version", "cuda_version", "driver_version",
        "model_revision", "tokenizer_revision", "hf_target_model_id", "fixed_server",
        "notebook_compatibility", "verified_precision_combinations", "requests_sha256",
        "dataset_identity", "row_indices", "package_versions", "source_vm", "vm_name",
        "hostname", "instance_id", "sglang_v3", "vllm_v3", "trtllm_v3",
    )
    result = {key: contract.get(key) for key in fields}
    result["environment"] = {key: environment.get(key) for key in env_fields}
    # Artifact identity survives different temporary snapshot paths.
    artifact = environment.get("weight_artifact", {})
    result["weight_artifact_id"] = artifact.get("checkpoint", {}).get("artifact_id")
    result["weight_source_revision"] = artifact.get("source_revision")
    return result


def local_records(root):
    paths = sorted(set(root.rglob("*.sqlite3")) | set(root.rglob("*.db")))
    for path in paths:
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not {"studies", "study_user_attributes", "trials"} <= tables:
                continue
            for sid, name in db.execute("SELECT study_id,study_name FROM studies"):
                attrs = {key: json.loads(value) for key, value in db.execute(
                    "SELECT key,value_json FROM study_user_attributes WHERE study_id=?", (sid,))}
                contract = attrs.get("contract", {})
                if not contract:
                    continue
                trials = list(db.execute(
                    "SELECT state,datetime_start,datetime_complete FROM trials WHERE study_id=? ORDER BY number", (sid,)))
                states = Counter(row[0] for row in trials)
                locator = attrs.get("wandb", {})
                yield {"contract": contract, "name": name, "id": locator.get("id"),
                       "url": locator.get("url"), "source": str(path), "host": None,
                       "states": dict(states), "trial_count": len(trials),
                       "duration": attrs.get("optimization_wall_seconds"),
                       # Only exact copies collapse; differing snapshots are kept.
                       "snapshot": canonical([locator or str(path), contract, trials])}


def wandb_records(project):
    import wandb

    for run in wandb.Api(timeout=60).runs(project):
        contract = dict(run.config)
        if not contract.get("sampler"):
            continue
        summary = dict(run.summary)
        study = summary.get("study", {})
        states = dict(study.get("states", summary.get("study.states", {})) or {})
        for key, value in summary.items():
            if key.startswith("study.states."):
                states[key.removeprefix("study.states.")] = value
        yield {"contract": contract, "name": run.name, "id": run.id, "url": run.url,
               "source": run.url, "host": getattr(run, "host", None),
               "states": states, "trial_count": sum(states.values()) if states else None,
               "duration": study.get("optimization_wall_seconds", summary.get("study.optimization_wall_seconds")),
               "summary_available": bool(states),
               "snapshot": run.id}


def audit(records):
    groups = defaultdict(list)
    seen = set()
    for record in records:
        contract = record["contract"]
        if contract.get("environment", {}).get("measurement") != "MEASURED":
            continue
        if contract.get("constrained") or str(contract.get("target", "")).startswith("pareto"):
            continue
        if record["snapshot"] in seen:
            continue
        seen.add(record["snapshot"])
        groups[canonical(context(contract))].append(record)
    rows = []
    for definition, group in sorted(groups.items()):
        c = group[0]["contract"]
        baseline = [r for r in group if r["contract"].get("sampler") in {"tpe", "tpe_multivariate"}]
        seeds = sorted({r["contract"].get("sampler_settings", {}).get("seed") for r in baseline}, key=str)
        for seed in seeds:
            tpe = [r for r in baseline if r["contract"].get("sampler_settings", {}).get("seed") == seed]
            gp = [r for r in group if r["contract"].get("sampler") == "gp"
                  and r["contract"].get("pruner_class") == "NopPruner"
                  and r["contract"].get("sampler_settings", {}).get("seed") == seed]
            completed = max((r["states"].get("COMPLETE", 0) for r in gp), default=0)
            recorded = max((r["trial_count"] or 0 for r in gp), default=0)
            budget = c.get("budget_trials")
            status = ("GP not observed" if not gp else
                      "GP observed; trial summary unavailable" if all(r.get("summary_available") is False for r in gp) else
                      "GP observed; no completed trial" if not completed
                      else "GP observed; below attempted-trial budget" if isinstance(budget, int) and recorded < budget
                      else "GP observed")
            vm = {key: c.get("environment", {}).get(key) for key in
                  ("source_vm", "vm_name", "hostname", "instance_id") if c.get("environment", {}).get(key)}
            rows.append({"context_id": hashlib.sha256(definition.encode()).hexdigest()[:12],
                "model": c.get("target_model"), "framework": c.get("framework"),
                "phase": c.get("phase"), "objective": c.get("target"), "concurrency": c.get("concurrency"),
                "seed": seed, "status": status, "gp_completed": completed, "gp_trial_count": recorded,
                "budget_trials": budget, "vm_identity": canonical(vm) if vm else "unverified",
                "tpe_names": canonical([r["name"] for r in tpe]),
                "gp_names": canonical([r["name"] for r in gp]),
                "gp_durations_seconds": canonical([r["duration"] for r in gp]),
                "expected_gp_study": f"{c.get('phase')}-{c.get('framework')}-c{c.get('concurrency')}-{c.get('target')}-gp-nop-s{seed}",
                "tpe_urls": canonical([r["url"] for r in tpe]), "gp_urls": canonical([r["url"] for r in gp]),
                "tpe_hosts": canonical([r["host"] for r in tpe]), "gp_hosts": canonical([r["host"] for r in gp]),
                "tpe_sources": canonical([r["source"] for r in tpe]), "context_definition": definition})
    return rows


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root / "runs/20261117-143416")
    parser.add_argument("--wandb-project", help="Read live W&B instead of local SQLite (requires wandb and authentication)")
    parser.add_argument("--out", type=Path, default=root / "docs/current_sampler_coverage.csv")
    args = parser.parse_args()
    rows = audit(wandb_records(args.wandb_project) if args.wandb_project else local_records(args.root))
    if not rows:
        raise SystemExit("No matching measured TPE studies in the selected source")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Source:", args.wandb_project or args.root)
    print("Status counts:", dict(Counter(r["status"] for r in rows)))
    print("Missing GP seed/context pairs (matching recorded environment; VM may be unverified):")
    missing = Counter((r["framework"], r["model"]) for r in rows if r["status"] == "GP not observed")
    for (framework, model), count in sorted(missing.items()):
        print(f"  {framework}: {model}: {count}")
    print("Full names, expected GP names, context definitions, and W&B links:", args.out)


if __name__ == "__main__":
    main()
