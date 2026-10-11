"""Persistent command workflow around the extracted lab algorithms."""

import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path

from .config import ExperimentConfig
from .lab import Lab, optuna


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def environment_manifest():
    packages = {}
    for name in ("llm-optimizer", "optuna", "numpy", "scipy", "torch", "wandb"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "git_revision": revision,
        "implementation_sha256": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in ("lab.py", "config.py", "workflow.py", "defaults.json")
        },
    }


def create_lab(config, run_dir):
    c = copy.deepcopy(config.values)
    settings = {name.upper(): value for name, value in c.items()}
    for name in ("SEARCH_TRIALS", "DEMO_TRIALS"):
        settings[name] = {
            p: {int(k): v for k, v in regimes.items()}
            for p, regimes in settings[name].items()
        }
    for name in ("TTFT_LIMITS_MS", "E2E_LIMITS_MS"):
        settings[name] = {int(k): v for k, v in settings[name].items()}
    settings["BASE_WORKLOAD"]["request_rate"] = (
        float("inf")
        if c["base_workload"]["request_rate"] == "inf"
        else c["base_workload"]["request_rate"]
    )
    settings["REAL_PREFLIGHT"] = {
        (p, f): v
        for p, frameworks in c["real_preflight"].items()
        for f, v in frameworks.items()
    }
    settings.update(
        RUN_DIR=run_dir,
        ROOT=Path.cwd(),
        STORAGE=f"sqlite:///{run_dir / 'studies.sqlite3'}",
        ACTIVE_SEEDS=c["seeds"][:1] if c["quick_demo"] else c["seeds"],
        RUN_GP=importlib.util.find_spec("torch") is not None,
        STUDIES={},
        CONSTRAINED_STUDIES={},
        SKIPPED_ARMS=[],
        GROUP_RUNS=Counter(),
        calibration={},
        WANDB_GROUP=c["wandb_group"] or run_dir.name,
    )
    settings["ARM_AVAILABILITY"] = {
        name: {
            "enabled": name != "gp" or settings["RUN_GP"],
            "reason": None
            if name != "gp" or settings["RUN_GP"]
            else "Install torch to run GPSampler",
        }
        for name in set(c["sampler_names"] + c["pareto_samplers"])
    }
    settings["ENVIRONMENT"] = {
        "measurement": "SYNTHETIC",
        "optuna": optuna.__version__,
        "python": platform.python_version(),
        "numpy": settings_numpy_version(),
        "model": "simulation-only",
        "gpu_count": c["gpu_count"],
    }
    return Lab(settings)


def settings_numpy_version():
    import numpy

    return numpy.__version__


def export_results(lab, studies, validation, skipped):
    recommendations = lab.validation_recommendations(
        validation, lab.VALIDATION_REQUIRE_SLO
    )
    artifact = {
        "controls": {
            "models": lab.PHASE_MODELS,
            "concurrencies": lab.CONCURRENCIES,
            "phase_requests": lab.PHASE_REQUESTS,
            "validation_requests": lab.VALIDATION_REQUESTS,
            "quick_demo": lab.QUICK_DEMO,
        },
        "skipped_arms": skipped,
        "calibration": list(lab.calibration.values()),
        "studies": [
            {
                "name": s.study_name,
                "contract": s.user_attrs["contract"],
                "summary": lab.summarize(s),
                "trials": [
                    {
                        "number": t.number,
                        "state": t.state.name,
                        "values": t.values,
                        "params": t.params,
                        "distributions": {
                            k: json.loads(optuna.distributions.distribution_to_json(v))
                            for k, v in t.distributions.items()
                        },
                        "attrs": t.user_attrs,
                        "intermediate_values": t.intermediate_values,
                    }
                    for t in s.trials
                ],
            }
            for s in studies
        ],
        "validation": validation,
        "recommendations": recommendations,
    }
    write_json(lab.RUN_DIR / "results.json", lab.artifact_value(artifact))
    return artifact


def execute(command, config_path, run_dir):
    config = ExperimentConfig.load(config_path)
    run_dir = Path(run_dir).resolve()
    frozen = run_dir / "config.json"
    if frozen.exists():
        if json.loads(frozen.read_text()) != config.values:
            raise ValueError(
                "Run directory config differs from frozen config; use "
                "the frozen config or a new run directory"
            )
    elif command != "run":
        raise ValueError(
            "Run directory has no frozen config; launch experiment run first"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    if not frozen.exists():
        write_json(frozen, config.values)
        write_json(run_dir / "environment.json", environment_manifest())
    lab = create_lab(config, run_dir)
    handler = logging.FileHandler(run_dir / "experiment.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger = logging.getLogger("optuna")
    logger.addHandler(handler)
    previous = (
        json.loads((run_dir / "results.json").read_text())
        if (run_dir / "results.json").exists()
        else {}
    )
    validation = previous.get("validation", [])
    for cal in previous.get("calibration", []):
        key = (cal["phase"], cal["concurrency"], cal["framework"], cal["target"])
        if cal["workload"].get("request_rate") == "inf":
            cal["workload"]["request_rate"] = float("inf")
        lab.calibration[key] = cal
    skipped = previous.get("skipped_arms", [])
    try:
        logger.info("experiment %s config=%s", command, frozen)
        if command == "run":
            try:
                run_search(lab, config.values)
            finally:
                studies = load_studies(lab)
                skipped = lab.SKIPPED_ARMS
                export_results(lab, studies, validation, skipped)
        elif command == "validate":
            studies = load_studies(lab)
            validation = lab.validate_candidates(lab.shortlist(studies))
            export_results(lab, studies, validation, skipped)
        elif command == "report":
            studies = load_studies(lab)
            export_results(lab, studies, validation, skipped)
        else:
            raise ValueError(f"Unknown experiment command: {command}")
    finally:
        logger.removeHandler(handler)
        handler.close()
    return run_dir / "results.json"


def load_studies(lab):
    if not (lab.RUN_DIR / "studies.sqlite3").exists():
        return []
    return [
        optuna.load_study(study_name=s.study_name, storage=lab.STORAGE)
        for s in optuna.study.get_all_study_summaries(storage=lab.STORAGE)
        if "contract" in s.user_attrs
    ]


def run_search(lab, c):
    frameworks = lab.REAL_FRAMEWORKS if lab.RUN_REAL_GPU else lab.FRAMEWORKS
    for phase in c["phases"]:
        for concurrency in lab.CONCURRENCIES:
            for target in c["objectives"][str(lab.regime_key(concurrency))]:
                samplers = (
                    lab.PARETO_SAMPLERS
                    if target.startswith("pareto")
                    else lab.SAMPLER_NAMES
                )
                for framework in frameworks:
                    choices = (
                        lab.REAL_PREFLIGHT[(phase, framework)].get("expanded_choices")
                        if lab.RUN_REAL_GPU and phase == "B"
                        else None
                    )
                    for arm in samplers:
                        if not lab.ARM_AVAILABILITY[arm]["enabled"]:
                            lab.SKIPPED_ARMS.append(
                                {
                                    "phase": phase,
                                    "framework": framework,
                                    "concurrency": concurrency,
                                    "target": target,
                                    "sampler": arm,
                                    "reason": lab.ARM_AVAILABILITY[arm]["reason"],
                                }
                            )
                            continue
                        for seed in (
                            lab.ACTIVE_SEEDS[:1] if arm == "grid" else lab.ACTIVE_SEEDS
                        ):
                            label = (
                                f"{phase}-{framework}-c{concurrency}-{target}-{arm}"
                                f"-s{seed}"
                            )
                            benchmark = lab.benchmark_for(
                                phase, framework, concurrency, label
                            )
                            lab.run_study(
                                label,
                                framework,
                                arm,
                                concurrency=concurrency,
                                phase=phase,
                                seed=seed,
                                target=target,
                                space_kind="baseline" if phase == "A" else "expanded",
                                choices=choices,
                                benchmark=benchmark,
                                constrained=c["constrained"],
                                gpu_budget=c["gpu_budget"],
                            )
    if c["pruning"]:
        if lab.RUN_REAL_GPU:
            lab.SKIPPED_ARMS.append(
                {
                    "arm": "pruning",
                    "reason": "Real pruning skipped; no continuous checkpoint "
                    "adapter or clean cancellation.",
                }
            )
        else:
            for phase in c["phases"]:
                for concurrency in lab.CONCURRENCIES:
                    for framework in frameworks:
                        for target in c["objectives"][str(lab.regime_key(concurrency))]:
                            if target.startswith("pareto"):
                                continue
                            cal = lab.calibrate_pruning(
                                phase,
                                concurrency,
                                framework,
                                target,
                                samples=c["pruning_samples"],
                            )
                            lab.calibration[(phase, concurrency, framework, target)] = (
                                cal
                            )
                            for name, (pruner_type, settings) in lab.pruner_menu(
                                cal["direction"], cal["threshold"]
                            ).items():
                                lab.run_study(
                                    f"pruning-{phase}-{framework}-c{concurrency}-{target}-{name}",
                                    framework,
                                    concurrency=concurrency,
                                    phase=phase,
                                    target=target,
                                    space_kind="baseline"
                                    if phase == "A"
                                    else "expanded",
                                    pruner=pruner_type(**settings),
                                    pruner_settings=settings,
                                    trials=c["pruning_trials"],
                                    seed=lab.ACTIVE_SEEDS[0],
                                )
