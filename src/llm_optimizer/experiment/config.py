"""Validated, serializable controls for the notebook experiment."""

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    values: dict

    @classmethod
    def load(cls, path):
        values = json.loads(Path(path).read_text())
        if not isinstance(values, dict):
            raise ValueError("Experiment config must be a JSON object")
        required = set(DEFAULTS)
        unknown = set(values) - required
        if unknown:
            raise ValueError(f"Unknown experiment settings: {sorted(unknown)}")
        config = copy.deepcopy(DEFAULTS)
        config.update(values)
        try:
            cls._validate(config)
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"Invalid or missing experiment setting: {exc}") from exc
        return cls(config)

    @staticmethod
    def _validate(c):
        def positive(value, name):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

        for name in (
            "concurrencies",
            "seeds",
            "frameworks",
            "real_frameworks",
            "phases",
            "sampler_names",
            "pareto_samplers",
        ):
            if (
                not isinstance(c[name], list)
                or not c[name]
                or len(set(c[name])) != len(c[name])
            ):
                raise ValueError(f"{name} must be a nonempty list of distinct values")
        for name in (
            "quick_demo",
            "run_real_gpu",
            "wandb_enabled",
            "validation_require_slo",
            "constrained",
            "pruning",
        ):
            if not isinstance(c[name], bool):
                raise ValueError(f"{name} must be boolean")
        for concurrency in c["concurrencies"]:
            positive(concurrency, "concurrency")
        for seed in c["seeds"]:
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError("seeds must be nonnegative integers")
        frameworks = {"vllm", "sglang", "trtllm"}
        for name in ("frameworks", "real_frameworks"):
            if not set(c[name]) <= frameworks:
                raise ValueError(f"Unsupported {name}")
        samplers = {"grid", "random", "qmc", "tpe", "tpe_multivariate", "gp", "nsga2"}
        for name in ("sampler_names", "pareto_samplers"):
            if not set(c[name]) <= samplers:
                raise ValueError(f"Unsupported {name}")
        if c["wandb_mode"] not in ("online", "offline", "disabled"):
            raise ValueError("Invalid wandb_mode")
        for name in (
            "validation_repeats",
            "pruning_samples",
            "pruning_trials",
            "gpu_count",
        ):
            positive(c[name], name)
        for name in (
            "simulation_seed",
            "validation_seed",
            "validation_noise_seed",
            "calibration_sampler_seed",
            "calibration_noise_seed",
        ):
            if isinstance(c[name], bool) or not isinstance(c[name], int) or c[name] < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for phase in c["phases"]:
            if phase not in ("A", "B") or not c["phase_models"].get(phase):
                raise ValueError("Unknown phase or missing model")
            for name in ("phase_requests", "validation_requests"):
                positive(c[name][phase], name)
            for name in ("search_trials", "demo_trials"):
                for regime in ("1", "128"):
                    positive(c[name][phase][regime], name)
        valid_targets = {
            "1": {"e2e", "ttft", "pareto_latency"},
            "128": {"e2e", "throughput", "pareto_throughput"},
        }
        if set(c["objectives"]) != {"1", "128"}:
            raise ValueError("objectives must define both concurrency regimes")
        for regime, targets in c["objectives"].items():
            if (
                regime not in valid_targets
                or not isinstance(targets, list)
                or not targets
                or len(set(targets)) != len(targets)
                or not set(targets) <= valid_targets[regime]
            ):
                raise ValueError("Invalid objectives for concurrency regime")
        for name in ("ttft_limits_ms", "e2e_limits_ms"):
            for regime in ("1", "128"):
                value = c[name][regime]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value <= 0
                ):
                    raise ValueError(f"{name} must be finite and positive")
        if (
            not isinstance(c["failure_rate_limit"], (int, float))
            or not 0 <= c["failure_rate_limit"] <= 1
        ):
            raise ValueError("failure_rate_limit must be between zero and one")
        for name in ("random_input_len", "random_output_len"):
            positive(c["base_workload"][name], name)
        rate = c["base_workload"]["request_rate"]
        if rate != "inf" and (
            not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0
        ):
            raise ValueError('request_rate must be positive or "inf"')
        if set(c["sampler_settings"]) != set(DEFAULTS["sampler_settings"]):
            raise ValueError("Invalid sampler_settings keys")
        for value in c["sampler_settings"].values():
            positive(value, "sampler_settings")
        if c["gpu_budget"] is not None and (
            isinstance(c["gpu_budget"], bool)
            or not isinstance(c["gpu_budget"], (int, float))
            or not math.isfinite(c["gpu_budget"])
            or c["gpu_budget"] <= 0
        ):
            raise ValueError("gpu_budget must be positive")
        if (
            isinstance(c["rest_seconds"], bool)
            or not isinstance(c["rest_seconds"], (int, float))
            or not math.isfinite(c["rest_seconds"])
            or c["rest_seconds"] < 0
        ):
            raise ValueError("rest_seconds must be nonnegative")
        if c["run_real_gpu"]:
            for phase in c["phases"]:
                if not Path(c["model_snapshots"][phase]).is_dir():
                    raise ValueError(
                        "Real GPU mode requires an existing pinned model snapshot"
                    )
                for framework in c["real_frameworks"]:
                    preflight = c["real_preflight"].get(phase, {}).get(framework, {})
                    for field in (
                        "gpu_type",
                        "framework_version",
                        "cuda_version",
                        "driver_version",
                        "model_revision",
                    ):
                        value = preflight.get(field)
                        if not value or "REPLACE" in str(value):
                            raise ValueError(
                                f"Real GPU preflight requires verified {field}"
                            )
                    if preflight.get("native_config_verified") is not True:
                        raise ValueError("Native config verification is required")
                    if phase == "B" and (
                        preflight.get("quality_verified") is not True
                        or not preflight.get("expanded_choices")
                    ):
                        raise ValueError(
                            "Phase B requires quality and expanded-choice preflight"
                        )


# Package default data is installed with the package, independent of the checkout.
DEFAULTS = json.loads(Path(__file__).with_name("defaults.json").read_text())
