"""Opt-in vLLM 0.33 v3 policy; legacy notebooks and registries stay unchanged."""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import math
import shlex
import subprocess
import sys
from functools import partial
from pathlib import Path

from llm_optimizer.args import Arg, ArgConfig, ArgScope
from llm_optimizer.predefined.vllm import SERVER_CONFIGS as LEGACY_SERVER_CONFIGS
from .phase1 import ExplicitLab, Phase1Runner
from .telemetry_vllm_v3 import DEFAULT_TELEMETRY, unavailable_telemetry

V3_SERVER_CONFIGS = dict(LEGACY_SERVER_CONFIGS)
for _name, _type in {
    "attention_backend": str, "performance_mode": str,
    "attention_config": str, "no_async_scheduling": bool,
    "disable_prefix_caching": bool, "prefill_schedule_interval": int,
    "watermark": float,
}.items():
    V3_SERVER_CONFIGS[_name] = ArgConfig[_type](name=_name)

DEFAULT_OPTIONAL_PARAMS = {
    "long_prefill_token_threshold": {"enabled": False, "values": [0, 128, 256]},
    "prefill_schedule_interval": {"enabled": False, "values": [1, 2]},
    "compilation_mode": {"enabled": False, "values": [0, 3]},
    "flash_attn_max_num_splits_for_cuda_graph": {"enabled": False, "values": [8, 16, 32]},
    "block_size": {"enabled": False, "values": [16, 32]},
    # One value gives a fixed override in B; A remains the original baseline.
    "max_model_len": {"enabled": False, "values": [1024]},
    "watermark": {"enabled": False, "values": [0.0, 0.05, 0.10]},
}


def validate_options(optional, telemetry):
    if set(optional) != set(DEFAULT_OPTIONAL_PARAMS):
        raise ValueError("optional_params must account for every v3 parameter")
    for name, option in optional.items():
        if set(option) != {"enabled", "values"} or type(option["enabled"]) is not bool:
            raise ValueError(f"Invalid option: {name}")
        values = option["values"]
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            raise ValueError(f"Nonempty distinct values required: {name}")
        if name == "watermark":
            valid = all(type(v) in (int, float) and math.isfinite(v) and 0 <= v < 1 for v in values)
        elif name == "compilation_mode":
            valid = all(type(v) is int and v in (0, 3) for v in values)
        elif name == "block_size":
            valid = all(type(v) is int and v in (16, 32) for v in values)
        else:
            minimum = 0 if name == "long_prefill_token_threshold" else 1
            valid = all(type(v) is int and v >= minimum for v in values)
        if not valid:
            raise ValueError(f"Unsupported optional domain: {name}")
    if set(telemetry) != set(DEFAULT_TELEMETRY) or type(telemetry["enabled"]) is not bool:
        raise ValueError("Invalid telemetry options")
    for name in ("interval_s", "timeout_s"):
        v = telemetry[name]
        if type(v) not in (int, float) or not math.isfinite(v) or v <= 0:
            raise ValueError(f"Invalid {name}")
    if telemetry["timeout_s"] > telemetry["interval_s"]:
        raise ValueError("Scrape timeout must not exceed interval")
    index = telemetry["gpu_index"]
    if index is not None and (type(index) is not int or index < 0):
        raise ValueError("gpu_index must be a physical NVML index")


class VLLMLabV3(ExplicitLab):
    def precision_config(self, framework, weights, cache):
        config = super().precision_config(framework, weights, cache)
        config.pop("calculate_kv_scales", None)
        if weights == "fp8":
            config["quantization"] = "fp8_per_tensor"
        return config

    def expanded_config(self, trial, framework, concurrency, config, choices=None):
        config = super().expanded_config(trial, framework, concurrency, config, choices)
        for name, option in self.optional_params.items():
            if not option["enabled"]:
                continue
            if concurrency == 1 and name in {"long_prefill_token_threshold", "prefill_schedule_interval", "watermark"}:
                continue
            if name == "flash_attn_max_num_splits_for_cuda_graph" and not (
                config.get("attention_backend") == "FLASH_ATTN"
                and config["compilation_config"]["cudagraph_mode"] != "NONE"
            ):
                continue
            value = trial.suggest_categorical(name, option["values"])
            if name == "compilation_mode":
                config["compilation_config"]["mode"] = value
                # Piecewise graphs require vLLM compilation; preserve decode graphs.
                if value == 0:
                    mode = config["compilation_config"]["cudagraph_mode"]
                    config["compilation_config"]["cudagraph_mode"] = {
                        "PIECEWISE": "NONE", "FULL_AND_PIECEWISE": "FULL_DECODE_ONLY"
                    }.get(mode, mode)
            elif name == "flash_attn_max_num_splits_for_cuda_graph":
                config.setdefault("attention_config", {})[name] = value
            else:
                config[name] = value
        if "max_model_len" in config and config["max_model_len"] < (
            self.BASE_WORKLOAD["random_input_len"] + self.BASE_WORKLOAD["random_output_len"]
        ):
            raise ValueError("max_model_len must cover the complete workload")
        return config

    def sampling_menu(self, framework, concurrency, space_kind="baseline", choices=None):
        menu = super().sampling_menu(framework, concurrency, space_kind, choices)
        if space_kind == "expanded":
            selected = self.expanded_choices(framework, choices)
            for name, option in self.optional_params.items():
                if not option["enabled"]:
                    continue
                if concurrency == 1 and name in {"long_prefill_token_threshold", "prefill_schedule_interval", "watermark"}:
                    continue
                if name == "flash_attn_max_num_splits_for_cuda_graph" and (
                    "FLASH_ATTN" not in selected["attention_backend"] or selected["graph_mode"] == ["NONE"]
                ):
                    continue
                menu[name] = option["values"]
        return menu

    def serialize_engine_config(self, framework, concurrency, config):
        native = super().serialize_engine_config(framework, concurrency, config)
        if "no_enable_prefix_caching" in native:
            native["disable_prefix_caching"] = native.pop("no_enable_prefix_caching")
        return native

    def record_measurement(self, trial, measurement, concurrency):
        super().record_measurement(trial, measurement, concurrency)
        execution = measurement.execution or {}
        trial.set_user_attr("server_telemetry", execution.get("server_telemetry") or unavailable_telemetry(
            "simulation: server metrics not measured" if not self.RUN_REAL_GPU else "no server telemetry returned"))
        trial.set_user_attr("configured_client_concurrency", concurrency)
        trial.set_user_attr("request_throughput", execution.get("request_throughput"))

    def benchmark_for(self, phase, framework, concurrency, run_label, **kwargs):
        if self.RUN_REAL_GPU:
            verify_runtime()
        return super().benchmark_for(phase, framework, concurrency, run_label, **kwargs)


def logging_measurement(trial, contract):
    attrs = trial.user_attrs
    execution = (
        attrs.get("execution") or attrs.get("measurement", {}).get("execution") or {}
    )
    telemetry = (
        attrs.get("server_telemetry")
        or execution.get("server_telemetry")
        or unavailable_telemetry("no measured server telemetry for this attempt")
    )
    values = dict(attrs.get("measurement", {}))
    values.update(
        request_throughput=execution.get(
            "request_throughput", attrs.get("request_throughput")
        ),
        configured_client_concurrency=execution.get(
            "configured_client_concurrency", contract["concurrency"]
        ),
        server_telemetry=telemetry,
    )
    for name, metric in telemetry.get("metrics", {}).items():
        for statistic in (
            "time_weighted_avg",
            "peak",
            "coverage",
            "sample_count",
            "reason",
            "unit",
            "available",
        ):
            values[f"server/{name}/{statistic}"] = metric.get(statistic)
    for name in ("sampling_interval_s", "sample_count", "measured_window_s"):
        values[f"server/{name}"] = telemetry.get(name)
    for name, value in telemetry.get("client_requests_in_flight", {}).items():
        values[f"client_requests_in_flight/{name}"] = value
    return values


class V3Logger:
    def __init__(self, original, contract):
        self.original = original
        self.contract = contract

    def __getattr__(self, name):
        return getattr(self.original, name)

    def __call__(self, study, trial):
        try:
            values = logging_measurement(trial, self.contract)
        except Exception:
            return self.original(study, trial)

        class TrialView:
            user_attrs = {**trial.user_attrs, "measurement": values}

            def __getattr__(self, name):
                return getattr(trial, name)

        return self.original(study, TrialView())


def verify_runtime():
    version = importlib.metadata.version("vllm")
    if version != "0.33.0":
        raise RuntimeError(f"v3 targets exact vLLM 0.33.0, found {version}; review adapter before changing pin")
    return {"python": sys.executable, "vllm": version}


def native_command(lab, config, model, concurrency=8, port=8000):
    native = lab.serialize_engine_config("vllm", concurrency, config)
    arguments = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", str(model),
                 "--host", "127.0.0.1", "--port", str(port)]
    for name, value in native.items():
        if name not in V3_SERVER_CONFIGS:
            raise ValueError(f"Unknown vLLM v3 argument: {name}")
        arguments.extend(Arg[V3_SERVER_CONFIGS[name].arg_type](
            name=name, value=value, scope=ArgScope.SERVER).generate_cmd_args())
    return shlex.join(arguments)


def audit_command(command, help_text):
    import re
    advertised = set(re.findall(r"--[a-z][a-z0-9-]*", help_text))
    used = {token.split("=", 1)[0] for token in shlex.split(command) if token.startswith("--")}
    if used - advertised:
        raise ValueError(f"CLI does not advertise {sorted(used - advertised)}")


def runtime_help():
    verify_runtime()
    return subprocess.run([sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", "--help=all"],
                          check=True, capture_output=True, text=True, timeout=120).stdout


class VLLMRunnerV3(Phase1Runner):
    def __init__(
        self, settings, *args, optional_params=None, telemetry_options=None, **kwargs
    ):
        optional = copy.deepcopy(
            DEFAULT_OPTIONAL_PARAMS if optional_params is None else optional_params
        )
        telemetry = {**DEFAULT_TELEMETRY, **(telemetry_options or {})}
        validate_options(optional, telemetry)
        settings = copy.deepcopy(settings)
        if settings["frameworks"] != ["vllm"]:
            raise ValueError("VLLM v3 is a dedicated VLLM runner")
        if settings["run_real_gpu"]:
            verify_runtime()
        identity = {
            "adapter": "vllm-v3",
            "optional_params": optional,
            "telemetry": telemetry,
            "implementation_sha256": {
                name: hashlib.sha256(
                    Path(__file__).with_name(name).read_bytes()
                ).hexdigest()
                for name in (
                    "vllm_policy_v3.py",
                    "runtime_vllm_v3.py",
                    "telemetry_vllm_v3.py",
                )
            },
        }
        for frameworks in settings["real_preflight"].values():
            frameworks["vllm"]["vllm_v3"] = identity
        super().__init__(settings, *args, **kwargs)
        self.lab.__class__ = VLLMLabV3
        self.lab.optional_params = optional
        self.lab.telemetry_options = telemetry
        from .runtime_vllm_v3 import RepositoryBenchmarkV3

        self.lab.RepositoryBenchmark = partial(RepositoryBenchmarkV3, _context=self.lab)
        original_logger = self.lab.NotebookWandbLogger
        self.lab.NotebookWandbLogger = lambda study, contract: V3Logger(
            original_logger(study, contract), contract
        )
