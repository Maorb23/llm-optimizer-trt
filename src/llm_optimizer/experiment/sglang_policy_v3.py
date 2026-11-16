"""Opt-in SGLang v3 search policy; no changes to legacy registries or lab classes."""

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
from llm_optimizer.predefined.sglang import SERVER_CONFIGS as LEGACY_SERVER_CONFIGS

from .phase1 import ExplicitLab, Phase1Runner
from .telemetry_sglang_v3 import DEFAULT_TELEMETRY, unavailable_telemetry

V3_SERVER_CONFIGS = dict(LEGACY_SERVER_CONFIGS)
for _name in (
    "linear_attn_decode_backend",
    "linear_attn_prefill_backend",
    "cuda_graph_config",
):
    V3_SERVER_CONFIGS[_name] = ArgConfig[str](name=_name)
for _name in (
    "prefill_max_requests",
    "prefill_decode_interval",
    "prefill_decode_interval_min_bs",
):
    V3_SERVER_CONFIGS[_name] = ArgConfig[int](name=_name)

DEFAULT_OPTIONAL_PARAMS = {
    "linear_attn_decode_backend": {
        "enabled": False,
        "values": ["triton", "flashinfer"],
    },
    "linear_attn_prefill_backend": {
        "enabled": False,
        "values": ["triton", "flashinfer"],
    },
    "decode_capture_profile": {"enabled": False, "values": ["sparse", "dense"]},
    # Stable Optuna categorical domain; values are clamped to effective capacity.
    "prefill_max_requests": {"enabled": False, "values": [4, 16, "capacity"]},
    "triton_attention_num_kv_splits": {"enabled": False, "values": [4, 8, 16]},
    "prefill_decode_interval": {"enabled": False, "values": [0, 2, 4]},
    "prefill_decode_interval_min_bs": {"enabled": False, "values": ["auto"]},
}


def validate_options(optional, telemetry):
    if set(optional) != set(DEFAULT_OPTIONAL_PARAMS):
        raise ValueError("optional_params must account for every v3 optional parameter")
    for name, option in optional.items():
        if set(option) != {"enabled", "values"} or type(option["enabled"]) is not bool:
            raise ValueError(f"Invalid optional parameter definition: {name}")
        values = option["values"]
        if (
            not isinstance(values, list)
            or not values
            or len(set(values)) != len(values)
        ):
            raise ValueError(f"Define a nonempty distinct domain for {name}")
        if name in ("linear_attn_decode_backend", "linear_attn_prefill_backend"):
            valid = set(values) <= {"triton", "flashinfer"}
        elif name == "decode_capture_profile":
            valid = set(values) <= {"sparse", "dense"}
        else:
            specials = (
                {"capacity"}
                if name == "prefill_max_requests"
                else ({"auto"} if name == "prefill_decode_interval_min_bs" else set())
            )
            minimum = 0 if name.startswith("prefill_decode_interval") else 1
            valid = all(
                v in specials or (type(v) is int and v >= minimum) for v in values
            )
        if not valid:
            raise ValueError(f"Unsupported optional domain: {name}")
    if (
        optional["prefill_decode_interval_min_bs"]["enabled"]
        and not optional["prefill_decode_interval"]["enabled"]
    ):
        raise ValueError("Interval minimum batch requires interval search")
    if (
        set(telemetry) != set(DEFAULT_TELEMETRY)
        or type(telemetry["enabled"]) is not bool
    ):
        raise ValueError("Invalid telemetry options")
    for name in ("interval_s", "timeout_s"):
        value = telemetry[name]
        if (
            not isinstance(value, (float, int))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be finite and positive")
    if telemetry["timeout_s"] > telemetry["interval_s"]:
        raise ValueError("Scrape timeout must not exceed sampling interval")
    index = telemetry["gpu_index"]
    if index is not None and (type(index) is not int or index < 0):
        raise ValueError(
            "gpu_index must be an explicit nonnegative NVML physical index"
        )


def cache_policies(choices):
    return [
        f"{'off' if disabled else 'on'}__{policy}"
        for disabled in choices["disable_radix_cache"]
        for policy in choices["schedule_policy"]
        if not (disabled and policy == "lpm")
    ]


class SGLangLabV3(ExplicitLab):
    def expanded_config(self, trial, framework, concurrency, config, choices=None):
        selected = self.expanded_choices(framework, choices)
        policies = cache_policies(selected)
        if not policies:
            raise ValueError("No nonduplicate cache/scheduling combinations")
        disabled, policy = trial.suggest_categorical("cache_policy", policies).split(
            "__"
        )

        class PolicyTrial:
            def __getattr__(self, name):
                return getattr(trial, name)

            def suggest_categorical(self, name, values):
                if name == "disable_radix_cache":
                    return disabled == "off"
                if name == "schedule_policy":
                    return policy
                return trial.suggest_categorical(name, values)

        config = super().expanded_config(
            PolicyTrial(), framework, concurrency, config, choices
        )
        capacity = min(concurrency, config["max_running_requests"])

        def pick(name, synthetic=None):
            return trial.suggest_categorical(
                synthetic or name, self.optional_params[name]["values"]
            )

        for name in ("linear_attn_decode_backend", "linear_attn_prefill_backend"):
            if self.optional_params[name]["enabled"]:
                config[name] = pick(name)
        if (
            self.optional_params["decode_capture_profile"]["enabled"]
            and config["cuda_graph_backend_decode"] != "disabled"
        ):
            profile = pick("decode_capture_profile") if concurrency > 1 else "sparse"
            buckets = (
                list(range(1, capacity + 1))
                if profile == "dense"
                else sorted(
                    {
                        1,
                        capacity,
                        *[
                            2**i
                            for i in range(capacity.bit_length())
                            if 2**i <= capacity
                        ],
                    }
                )
            )
            config["cuda_graph_config"] = {
                "decode": {
                    "backend": config["cuda_graph_backend_decode"],
                    "bs": buckets,
                },
                "prefill": {"backend": config["cuda_graph_backend_prefill"]},
            }
        if concurrency > 1 and self.optional_params["prefill_max_requests"]["enabled"]:
            value = pick("prefill_max_requests", "prefill_request_cap_choice")
            config["prefill_max_requests"] = (
                capacity if value == "capacity" else min(value, capacity)
            )
        if (
            self.optional_params["triton_attention_num_kv_splits"]["enabled"]
            and config["attention_backend"] == "triton"
        ):
            config["triton_attention_num_kv_splits"] = pick(
                "triton_attention_num_kv_splits"
            )
        if self.optional_params["prefill_decode_interval"]["enabled"]:
            config["prefill_decode_interval"] = (
                pick("prefill_decode_interval") if concurrency > 1 else 0
            )
            threshold = self.optional_params["prefill_decode_interval_min_bs"]
            value = (
                pick("prefill_decode_interval_min_bs")
                if threshold["enabled"]
                and concurrency > 1
                and config["prefill_decode_interval"]
                else "auto"
            )
            config["prefill_decode_interval_min_bs"] = (
                min(8, concurrency) if value == "auto" else value
            )
        return config

    def sampling_menu(
        self, framework, concurrency, space_kind="baseline", choices=None
    ):
        menu = super().sampling_menu(framework, concurrency, space_kind, choices)
        if space_kind == "expanded":
            selected = self.expanded_choices(framework, choices)
            menu.pop("disable_radix_cache", None)
            menu.pop("schedule_policy", None)
            menu["cache_policy"] = cache_policies(selected)
            for name, option in self.optional_params.items():
                if not option["enabled"]:
                    continue
                if concurrency == 1 and name in {
                    "decode_capture_profile",
                    "prefill_max_requests",
                    "prefill_decode_interval",
                    "prefill_decode_interval_min_bs",
                }:
                    continue
                if name == "decode_capture_profile" and selected[
                    "cuda_graph_backend_decode"
                ] == ["disabled"]:
                    continue
                if (
                    name == "triton_attention_num_kv_splits"
                    and "triton" not in selected["attention_backend"]
                ):
                    continue
                menu[
                    "prefill_request_cap_choice"
                    if name == "prefill_max_requests"
                    else name
                ] = option["values"]
        return menu

    def serialize_engine_config(self, framework, concurrency, config):
        config = dict(config)
        if "cuda_graph_config" in config:
            config.pop("cuda_graph_backend_decode", None)
            config.pop("cuda_graph_backend_prefill", None)
        return super().serialize_engine_config(framework, concurrency, config)

    def record_measurement(self, trial, measurement, concurrency):
        super().record_measurement(trial, measurement, concurrency)
        execution = measurement.execution or {}
        trial.set_user_attr(
            "server_telemetry",
            execution.get("server_telemetry")
            or unavailable_telemetry(
                "simulation: server metrics not measured"
                if not self.RUN_REAL_GPU
                else "no server telemetry returned"
            ),
        )
        trial.set_user_attr("configured_client_concurrency", concurrency)
        if execution:
            trial.set_user_attr(
                "request_throughput", execution.get("request_throughput")
            )
            trial.set_user_attr("configured_client_concurrency", concurrency)

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
    version = importlib.metadata.version("sglang")
    if version != "0.5.23":
        raise RuntimeError(f"v3 requires SGLang 0.5.23, found {version}")
    return {"python": sys.executable, "sglang": version}


def native_command(lab, config, model, concurrency=8, port=30000):
    native = lab.serialize_engine_config("sglang", concurrency, config)
    arguments = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    for name, value in native.items():
        if name not in V3_SERVER_CONFIGS:
            raise ValueError(f"Unknown SGLang v3 argument: {name}")
        arguments.extend(
            Arg[V3_SERVER_CONFIGS[name].arg_type](
                name=name, value=value, scope=ArgScope.SERVER
            ).generate_cmd_args()
        )
    return shlex.join(arguments)


def audit_command(command, help_text):
    advertised = set(__import__("re").findall(r"--[a-z][a-z0-9-]*", help_text))
    used = {
        token.split("=", 1)[0]
        for token in shlex.split(command)
        if token.startswith("--")
    }
    if used - advertised:
        raise ValueError(f"SGLang CLI does not advertise {sorted(used - advertised)}")


def runtime_help():
    verify_runtime()
    return subprocess.run(
        [sys.executable, "-m", "sglang.launch_server", "--help"],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout


class SGLangRunnerV3(Phase1Runner):
    def __init__(
        self, settings, *args, optional_params=None, telemetry_options=None, **kwargs
    ):
        optional = copy.deepcopy(
            DEFAULT_OPTIONAL_PARAMS if optional_params is None else optional_params
        )
        telemetry = {**DEFAULT_TELEMETRY, **(telemetry_options or {})}
        validate_options(optional, telemetry)
        settings = copy.deepcopy(settings)
        if settings["frameworks"] != ["sglang"]:
            raise ValueError("SGLang v3 is a dedicated SGLang runner")
        if settings["run_real_gpu"]:
            verify_runtime()
        identity = {
            "adapter": "sglang-v3",
            "optional_params": optional,
            "telemetry": telemetry,
            "implementation_sha256": {
                name: hashlib.sha256(
                    Path(__file__).with_name(name).read_bytes()
                ).hexdigest()
                for name in (
                    "sglang_policy_v3.py",
                    "runtime_sglang_v3.py",
                    "telemetry_sglang_v3.py",
                )
            },
        }
        for frameworks in settings["real_preflight"].values():
            frameworks["sglang"]["sglang_v3"] = identity
        super().__init__(settings, *args, **kwargs)
        self.lab.__class__ = SGLangLabV3
        self.lab.optional_params = optional
        self.lab.telemetry_options = telemetry
        from .runtime_sglang_v3 import RepositoryBenchmarkV3

        self.lab.RepositoryBenchmark = partial(RepositoryBenchmarkV3, _context=self.lab)
        original_logger = self.lab.NotebookWandbLogger
        self.lab.NotebookWandbLogger = lambda study, contract: V3Logger(
            original_logger(study, contract), contract
        )
