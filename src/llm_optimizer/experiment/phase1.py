"""Small, resumable runner shared by the phase-one notebook and batch script."""

import copy
import fcntl
import hashlib
import itertools
import json
from contextlib import contextmanager
from pathlib import Path

import optuna

from .config import DEFAULTS, ExperimentConfig
from .lab import Lab
from .workflow import create_lab, environment_manifest, write_json


class ExplicitLab(Lab):
    """Expose the numeric and conditional domains otherwise fixed in Lab."""

    def expanded_config(self, trial, framework, concurrency, config, choices=None):
        domains = self.extra_spaces[framework]
        lab = self

        class DomainTrial:
            def suggest_float(self, name, *args, **kwargs):
                if name in domains:
                    return lab.suggest_numeric(trial, name, domains[name])
                return trial.suggest_float(name, *args, **kwargs)

            def suggest_int(self, name, *args, **kwargs):
                if name in domains:
                    return lab.suggest_numeric(trial, name, domains[name])
                return trial.suggest_int(name, *args, **kwargs)

            def suggest_categorical(self, name, values):
                if name == "kv_cache_dtype" and lab.precision_choices:
                    return "auto"  # sampled independently below as kv_cache_precision
                return trial.suggest_categorical(name, domains.get(name, values))

        config = super().expanded_config(
            DomainTrial(), framework, concurrency, config, choices
        )
        if self.precision_choices:
            precision = self.precision_choices[framework]
            weights = trial.suggest_categorical("weight_precision", precision["weight_precision"])
            cache = trial.suggest_categorical("kv_cache_precision", precision["kv_cache_precision"])
            config.update(self.precision_config(framework, weights, cache))
            if framework == "trtllm":
                config["kv_cache_config"]["dtype"] = "fp8" if cache == "fp8" else "auto"
        return config

    def render_engine_config(self, framework, concurrency, p):
        config = super().render_engine_config(framework, concurrency, p)
        if self.precision_choices:
            config.update(self.precision_config(framework, "bf16", "bf16"))
        return config

    def precision_config(self, framework, weights, cache):
        config = {"dtype": "bfloat16"}
        if framework == "trtllm":
            config["quant_config"] = {
                "quant_algo": "FP8" if weights == "fp8" else None,
                "kv_cache_quant_algo": "FP8" if cache == "fp8" else None,
            }
            if weights == "fp8":
                config["calib_config"] = copy.deepcopy(self.trt_fp8_calibration)
        else:
            if weights == "fp8":
                config["quantization"] = "fp8"
            config["kv_cache_dtype"] = (
                ("fp8_e4m3" if framework == "sglang" else "fp8")
                if cache == "fp8" else "auto"
            )
            if framework == "vllm" and cache == "fp8":
                config["calculate_kv_scales"] = True
        return config

    def sampling_menu(self, framework, concurrency, space_kind="baseline", choices=None):
        menu = super().sampling_menu(framework, concurrency, space_kind, choices)
        if space_kind == "expanded":
            for name in ("graph_capture_profile", "graph_padding", "graph_padding_off"):
                menu.pop(name, None)
            menu.update(self.extra_spaces[framework])
            if self.precision_choices:
                menu.pop("kv_cache_dtype", None)
                menu.update(self.precision_choices[framework])
        return menu


class Phase1Runner:
    def __init__(self, settings, base_spaces, expanded_choices, extra_spaces, run_root,
                 precision_choices=None, trt_fp8_calibration=None):
        values = copy.deepcopy(DEFAULTS)
        values.update(copy.deepcopy(settings))
        ExperimentConfig._validate(values)
        precision_choices = copy.deepcopy(precision_choices or {})
        trt_fp8_calibration = copy.deepcopy(trt_fp8_calibration or {})
        for framework in values["frameworks"]:
            if not precision_choices:
                break
            precision = precision_choices.get(framework, {})
            if set(precision) != {"weight_precision", "kv_cache_precision"}:
                raise ValueError(f"{framework}: define weight_precision and kv_cache_precision")
            for domain in precision.values():
                if not domain or len(set(domain)) != len(domain) or not set(domain) <= {"bf16", "fp8"}:
                    raise ValueError("Precision domains must be nonempty subsets of bf16/fp8")
            for phase in values["phases"]:
                fixed = {
                    **values["fixed_server"].get(framework, {}),
                    **values["real_preflight"].get(phase, {}).get(framework, {}).get("fixed_server", {}),
                }
                controlled = {"dtype", "quantization", "kv_cache_dtype", "quant_config",
                              "calib_config", "calculate_kv_scales"}
                if controlled.intersection(fixed) or "dtype" in (fixed.get("kv_cache_config") or {}):
                    raise ValueError("Precision settings conflict with fixed_server; use PRECISION_CHOICES")
            if values["run_real_gpu"]:
                for phase in values["phases"]:
                    if phase != "B":
                        continue
                    verified = values["real_preflight"][phase][framework].get(
                        "verified_precision_combinations", []
                    )
                    requested = {f"{w}/{k}" for w, k in itertools.product(
                        precision["weight_precision"], precision["kv_cache_precision"]
                    )}
                    if not requested <= set(verified):
                        raise ValueError(f"{framework}: verify precision combinations {sorted(requested - set(verified))}")
                for phase in values["phases"]:
                    model_config = Path(values["model_snapshots"][phase]) / "config.json"
                    if not model_config.is_file():
                        raise ValueError("Precision search requires a pinned BF16 source config.json")
                    if json.loads(model_config.read_text()).get("quantization_config"):
                        raise ValueError("Precision search requires unquantized source weights, not an FP8 checkpoint")
        self.payload = {
            "settings": values,
            "base_spaces": base_spaces,
            "expanded_choices": expanded_choices,
            "extra_spaces": extra_spaces,
            "precision_choices": precision_choices,
            "trt_fp8_calibration": trt_fp8_calibration,
        }
        # Hash includes workload, environment/preflight and every search dimension.
        encoded = json.dumps(self.payload, sort_keys=True)
        digest = hashlib.sha256(encoded.encode()).hexdigest()[:16]
        self.run_root = Path(run_root).resolve()
        self.run_dir = self.run_root / (values["wandb_group"] + "-" + digest)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with self.lock():
            frozen = self.run_dir / "config.json"
            if frozen.exists() and json.loads(frozen.read_text()) != json.loads(encoded):
                raise ValueError("Frozen campaign configuration differs")
            if not frozen.exists():
                write_json(frozen, self.payload)
                write_json(self.run_dir / "environment.json", environment_manifest())
        original = create_lab(ExperimentConfig(values), self.run_dir)
        self.lab = ExplicitLab(original.__dict__)
        self.lab.FLOAT_PARAMS = self.lab.FLOAT_PARAMS | {"schedule_conservativeness"}
        self.lab.SPACES = copy.deepcopy(base_spaces)
        self.lab.extra_spaces = copy.deepcopy(extra_spaces)
        self.lab.precision_choices = precision_choices
        self.lab.trt_fp8_calibration = trt_fp8_calibration
        self.lab.EXPANDED_CHOICES = copy.deepcopy(expanded_choices)
        for framework in values["frameworks"]:
            self.lab.expanded_choices(framework)
        self.lab.STORAGE = optuna.storages.RDBStorage(
            url=f"sqlite:///{self.run_dir / 'studies.sqlite3'}",
            engine_kwargs={"connect_args": {"timeout": 60}},
        )

    @contextmanager
    def lock(self):
        # All workers sharing this root serialize access to the VM's GPU.
        self.run_root.mkdir(parents=True, exist_ok=True)
        with (self.run_root / "gpu.lock").open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Another experiment owns this VM's GPU") from exc
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def run_objective_group(self, phase, target, concurrencies, pruner_name="nop"):
        if pruner_name not in {"nop", "median"}:
            raise ValueError("Use nop or median")
        if pruner_name != "nop" and target.startswith("pareto"):
            raise ValueError("Pareto objectives require nop pruning")
        results = {}
        with self.lock():
            for concurrency in concurrencies:
                if target not in self.lab.objectives_for(concurrency):
                    raise ValueError("Objective incompatible with concurrency")
                for framework in self.lab.REAL_FRAMEWORKS:
                    for sampler in self.lab.SAMPLER_NAMES:
                        for seed in self.lab.ACTIVE_SEEDS:
                            name = (
                                f"{phase}-{framework}-c{concurrency}-{target}-"
                                f"{sampler}-{pruner_name}-s{seed}"
                            )
                            summaries = optuna.get_all_study_summaries(self.lab.STORAGE)
                            if any(s.study_name == name for s in summaries):
                                old = optuna.load_study(
                                    study_name=name, storage=self.lab.STORAGE
                                )
                                # Safe only after exclusive ownership; batch supervisor
                                # kills the previous process group before restarting.
                                for trial in old.trials:
                                    if trial.state == optuna.trial.TrialState.RUNNING:
                                        old.tell(trial.number, state=optuna.trial.TrialState.FAIL)
                            choices = self.lab.REAL_PREFLIGHT.get(
                                (phase, framework), {}
                            ).get("expanded_choices", self.lab.EXPANDED_CHOICES[framework])
                            pruning = {"n_startup_trials": 10, "n_warmup_steps": 32,
                                       "n_min_trials": 3} if pruner_name == "median" else {}
                            pruner = (optuna.pruners.MedianPruner(**pruning)
                                      if pruning else optuna.pruners.NopPruner())
                            study = self.lab.run_study(
                                name, framework, sampler, concurrency=concurrency,
                                phase=phase, seed=seed, target=target,
                                space_kind="expanded" if phase == "B" else "baseline",
                                choices=choices if phase == "B" else None,
                                pruner=pruner, pruner_settings=pruning,
                                benchmark=self.lab.benchmark_for(
                                    phase, framework, concurrency, name
                                ),
                            )
                            if not any(t.state == optuna.trial.TrialState.COMPLETE
                                       for t in study.trials):
                                raise RuntimeError(
                                    f"{name}: no completed measurements; inspect trial failures"
                                )
                            results[name] = study
                            write_json(self.run_dir / (name + ".json"), {
                                "study": name, "contract": study.user_attrs["contract"],
                                "trials": [{"number": t.number, "state": t.state.name,
                                            "values": t.values, "params": t.params,
                                            "attrs": t.user_attrs} for t in study.trials],
                            })
        return results


def plot_group(studies):
    """Plot completed scalar trials without silently including failures/pruning."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    for name, study in studies.items():
        trials = [t for t in study.trials
                  if t.state == optuna.trial.TrialState.COMPLETE]
        if trials and len(study.directions) == 1:
            ax.plot([t.number for t in trials], [t.value for t in trials], label=name)
    ax.set(xlabel="Trial", ylabel="Objective value")
    if ax.lines:
        ax.legend(fontsize=7)
    fig.tight_layout()
    return fig
