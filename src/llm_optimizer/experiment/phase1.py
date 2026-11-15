"""Shared persistent runner for docs/phase1_experiments.ipynb and batch execution."""

import copy
import fcntl
import hashlib
import itertools
import json
import warnings
from contextlib import contextmanager
from pathlib import Path

import optuna

from .config import DEFAULTS, ExperimentConfig
from .lab import Lab
from .workflow import create_lab, environment_manifest, write_json


class ExplicitLab(Lab):
    """Expose the extended numeric, conditional and precision domains."""

    def benchmark_for(self, phase, framework, concurrency, run_label, **kwargs):
        benchmark = super().benchmark_for(phase, framework, concurrency, run_label, **kwargs)
        if not self.RUN_REAL_GPU:
            benchmark.environment = {
                **copy.deepcopy(self.REAL_PREFLIGHT.get((phase, framework), {})),
                **self.ENVIRONMENT,
            }
        return benchmark

    def workload_for(self, concurrency, phase="A", num_prompts=None, seed=None):
        if num_prompts is None and self.requests_by_concurrency:
            num_prompts = self.requests_by_concurrency[concurrency]
        return super().workload_for(concurrency, phase, num_prompts, seed)

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
                    return "auto"  # one independent precision axis below
                return trial.suggest_categorical(name, domains.get(name, values))

        config = super().expanded_config(
            DomainTrial(), framework, concurrency, config, choices
        )
        if self.precision_choices:
            precision = self.precision_choices[framework]
            weights = trial.suggest_categorical(
                "weight_precision", precision["weight_precision"]
            )
            cache = trial.suggest_categorical(
                "kv_cache_precision", precision["kv_cache_precision"]
            )
            config.update(self.precision_config(framework, weights, cache))
            if framework == "trtllm":
                config["kv_cache_config"]["dtype"] = (
                    "fp8" if cache == "fp8" else "auto"
                )
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


def verify_precision(values, precision_choices):
    for framework in values["frameworks"]:
        if not precision_choices:
            return
        precision = precision_choices.get(framework, {})
        if set(precision) != {"weight_precision", "kv_cache_precision"}:
            raise ValueError("Define weight_precision and kv_cache_precision")
        for domain in precision.values():
            if (not domain or len(set(domain)) != len(domain)
                    or not set(domain) <= {"bf16", "fp8"}):
                raise ValueError("Precision domains must be nonempty subsets of bf16/fp8")
        for phase in values["phases"]:
            preflight = values["real_preflight"].get(phase, {}).get(framework, {})
            fixed = {
                **values["fixed_server"].get(framework, {}),
                **preflight.get("fixed_server", {}),
            }
            controlled = {"dtype", "quantization", "kv_cache_dtype", "quant_config",
                          "calib_config", "calculate_kv_scales"}
            if (controlled.intersection(fixed)
                    or "dtype" in (fixed.get("kv_cache_config") or {})):
                raise ValueError("Precision settings conflict with fixed_server")
            if values["run_real_gpu"]:
                if phase == "B":
                    verified = preflight.get("verified_precision_combinations", [])
                    requested = {f"{w}/{k}" for w, k in itertools.product(
                        precision["weight_precision"], precision["kv_cache_precision"]
                    )}
                    if not requested <= set(verified):
                        raise ValueError(
                            f"{framework}: verify precision combinations "
                            f"{sorted(requested - set(verified))}"
                        )
                path = Path(values["model_snapshots"][phase]) / "config.json"
                if not path.is_file():
                    raise ValueError("Precision search requires a pinned BF16 config.json")
                if json.loads(path.read_text()).get("quantization_config"):
                    raise ValueError("Precision search requires unquantized source weights")


class Phase1Runner:
    def __init__(self, settings, base_spaces, expanded_choices, extra_spaces, run_root,
                 precision_choices=None, trt_fp8_calibration=None,
                 requests_by_concurrency=None, pruning_min_completed_requests=None):
        values = copy.deepcopy(DEFAULTS)
        values.update(copy.deepcopy(settings))
        ExperimentConfig._validate(values)
        precision_choices = copy.deepcopy(precision_choices or {})
        trt_fp8_calibration = copy.deepcopy(trt_fp8_calibration or {})
        verify_precision(values, precision_choices)
        request_map = {int(c): n for c, n in (requests_by_concurrency or {}).items()}
        minimum_map = {int(c): n for c, n in (pruning_min_completed_requests or {}).items()}
        for concurrency in values["concurrencies"]:
            if request_map and concurrency not in request_map:
                raise ValueError(f"Missing request count for concurrency {concurrency}")
            count = request_map.get(concurrency, values["phase_requests"]["B"])
            minimum = minimum_map.get(concurrency, 1)
            if (isinstance(count, bool) or not isinstance(count, int) or count < 1
                    or isinstance(minimum, bool) or not isinstance(minimum, int)
                    or not 1 <= minimum <= count):
                raise ValueError("Request counts and pruning minimums must be positive integers; minimum <= count")
        self.payload = {
            "settings": values, "base_spaces": base_spaces,
            "expanded_choices": expanded_choices, "extra_spaces": extra_spaces,
            "precision_choices": precision_choices,
            "trt_fp8_calibration": trt_fp8_calibration,
            "requests_by_concurrency": request_map,
            "pruning_min_completed_requests": minimum_map,
        }
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
        self.lab.requests_by_concurrency = request_map
        self.lab.pruning_min_completed_requests = minimum_map
        self.lab.EXPANDED_CHOICES = copy.deepcopy(expanded_choices)
        for framework in values["frameworks"]:
            self.lab.expanded_choices(framework)
        self.lab.STORAGE = optuna.storages.RDBStorage(
            url=f"sqlite:///{self.run_dir / 'studies.sqlite3'}",
            engine_kwargs={"connect_args": {"timeout": 60}},
        )

    @contextmanager
    def lock(self):
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

    def _calibrate(self, phase, target, concurrency, framework, samples, reuse=False):
        path = self.run_dir / f"calibration-{phase}-{framework}-c{concurrency}-{target}.json"
        if reuse and path.exists():
            report = json.loads(path.read_text())
            if report.get("requested_samples") == samples:
                return report
        report = self.lab.calibrate_pruning(
            phase, concurrency, framework, target, samples=samples,
            min_completed_requests=self.lab.pruning_min_completed_requests.get(concurrency, 1),
        )
        report = self.lab.artifact_value(report)
        write_json(path, report)
        return report

    def calibrate(self, phase, target, concurrency, framework, samples=32):
        with self.lock():
            return self._calibrate(phase, target, concurrency, framework, samples)

    def run_objective_group(self, phase, target, concurrencies, pruner_name="nop",
                            sampler_names=None, constrained=False):
        if pruner_name not in {"nop", "median", "percentile", "halving", "hyperband", "threshold"}:
            raise ValueError("Unknown pruner")
        if constrained and pruner_name != "nop":
            raise ValueError("Constrained searches require nop pruning")
        if pruner_name != "nop" and target.startswith("pareto"):
            raise ValueError("Pareto objectives require nop pruning")
        samplers = list(sampler_names if sampler_names is not None else (
            self.lab.PARETO_SAMPLERS if target.startswith("pareto") else self.lab.SAMPLER_NAMES
        ))
        if not samplers or len(set(samplers)) != len(samplers):
            raise ValueError("Select a nonempty list of distinct samplers")
        supported = {"gp", "tpe", "tpe_multivariate", "nsga2", "random", "qmc", "grid"}
        if not set(samplers) <= supported:
            raise ValueError(f"Unknown samplers: {sorted(set(samplers) - supported)}")
        if constrained and not set(samplers) <= {"gp", "tpe", "tpe_multivariate", "nsga2"}:
            raise ValueError("Select a constraint-aware sampler")
        results = {}
        with self.lock():
            for concurrency in concurrencies:
                if target not in self.lab.objectives_for(concurrency):
                    raise ValueError("Objective incompatible with concurrency")
                for framework in self.lab.REAL_FRAMEWORKS:
                    for sampler in samplers:
                        if sampler == "gp" and not self.lab.RUN_GP:
                            skipped = {"phase": phase, "framework": framework,
                                       "concurrency": concurrency, "target": target,
                                       "sampler": sampler, "reason": "Install engine-compatible torch for GPSampler"}
                            self.lab.SKIPPED_ARMS.append(skipped)
                            warnings.warn(skipped["reason"], stacklevel=2)
                            write_json(self.run_dir / "skipped-arms.json", self.lab.SKIPPED_ARMS)
                            continue
                        for seed in self.lab.ACTIVE_SEEDS:
                            name = (f"{phase}-{framework}-c{concurrency}-{target}-"
                                    f"{sampler}-{pruner_name}-s{seed}"
                                    + ("-constrained" if constrained else ""))
                            summaries = optuna.get_all_study_summaries(self.lab.STORAGE)
                            if any(s.study_name == name for s in summaries):
                                old = optuna.load_study(
                                    study_name=name, storage=self.lab.STORAGE
                                )
                                # Exclusive ownership is required; the batch supervisor
                                # terminates prior engine descendants before restart.
                                for trial in old.trials:
                                    if trial.state == optuna.trial.TrialState.RUNNING:
                                        old.tell(trial.number,
                                                 state=optuna.trial.TrialState.FAIL)
                            choices = self.lab.REAL_PREFLIGHT.get(
                                (phase, framework), {}
                            ).get("expanded_choices", self.lab.EXPANDED_CHOICES[framework])
                            minimum = self.lab.pruning_min_completed_requests.get(concurrency, 1)
                            calibration = None
                            if pruner_name == "threshold":
                                calibration = self._calibrate(
                                    phase, target, concurrency, framework,
                                    samples=self.payload["settings"]["pruning_samples"], reuse=True,
                                )
                                write_json(self.run_dir / (name + "-calibration.json"), calibration)
                            menu = self.lab.pruner_menu(
                                self.lab.objective_spec(target, concurrency)[2][0],
                                threshold=calibration["threshold"] if calibration else None,
                                max_requests=self.lab.workload_for(concurrency, phase)["num_prompts"],
                                min_completed_requests=minimum,
                            )
                            pruner_type, pruning = menu[pruner_name]
                            pruner = pruner_type(**pruning)
                            study = self.lab.run_study(
                                name, framework, sampler, concurrency=concurrency,
                                phase=phase, seed=seed, target=target, constrained=constrained,
                                space_kind="expanded" if phase == "B" else "baseline",
                                choices=choices if phase == "B" else None,
                                pruner=pruner, pruner_settings=pruning,
                                min_completed_requests=minimum,
                                benchmark=self.lab.benchmark_for(
                                    phase, framework, concurrency, name
                                ),
                            )
                            if not any(t.state == optuna.trial.TrialState.COMPLETE
                                       for t in study.trials):
                                raise RuntimeError(f"{name}: no completed measurements")
                            results[name] = study
                            write_json(self.run_dir / (name + ".json"), {
                                "study": name, "contract": study.user_attrs["contract"],
                                "trials": [{"number": t.number, "state": t.state.name,
                                            "values": t.values, "params": t.params,
                                            "attrs": t.user_attrs} for t in study.trials],
                            })
        return results


def plot_group(studies):
    import matplotlib.pyplot as plt

    if any(len(s.directions) > 1 for s in studies.values()):
        return {name: optuna.visualization.matplotlib.plot_pareto_front(study).figure
                for name, study in studies.items() if len(study.directions) > 1}
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
