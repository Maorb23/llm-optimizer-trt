"""Opt-in TRT-LLM 1.4.0rc3 PyTorch policy; checkpoint precision stays notebook-owned.

Source: NVIDIA/TensorRT-LLM commit 9e334fbaae504cc836cddea8a20cc1bdab543e55,
llmapi/llm_args.py, commands/serve.py and metrics/collector.py.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
import re
import shlex
import shutil
import subprocess
from functools import partial
from pathlib import Path

from llm_optimizer.predefined.trtllm import NATIVE_CONFIG_FIELDS, SERVER_CONFIGS

from .phase1 import ExplicitLab, Phase1Runner
from .telemetry_trtllm_v3 import DEFAULT_TELEMETRY, unavailable_telemetry

V3_SERVER_CONFIGS = dict(SERVER_CONFIGS)
DEFAULT_OPTIONAL_PARAMS = {
    "disable_overlap_scheduler": {"enabled": False, "values": [False, True]},
    "enable_early_first_token_response": {"enabled": False, "values": [False, True]},
    "enable_in_graph_sampling": {"enabled": False, "values": [False, True]},
    "prefill_cuda_graph_backend": {
        "enabled": False,
        "values": ["disabled", "piecewise", "breakable"],
    },
    "prefill_capture_profile": {"enabled": False, "values": ["workload", "sparse"]},
    "compile_only_piecewise_graphs": {"enabled": False, "values": [False, True]},
    "stream_interval": {"enabled": False, "values": [1, 2, 4, 8]},
    "kv_tokens_per_block": {"enabled": False, "values": [32, 64]},
    "num_postprocess_workers": {"enabled": False, "values": [0, 1, 2, 4]},
}
V3_NATIVE_FIELDS = set(NATIVE_CONFIG_FIELDS) | {
    "disable_overlap_scheduler",
    "enable_early_first_token_response",
    "enable_in_graph_sampling",
    "prefill_cuda_graph_backend",
    "prefill_capture_num_tokens",
    "torch_compile_config",
    "stream_interval",
    "enable_iter_perf_stats",
}


def validate_options(optional, telemetry):
    if set(optional) != set(DEFAULT_OPTIONAL_PARAMS):
        raise ValueError("optional_params must account for every v3 parameter")
    for name, option in optional.items():
        if set(option) != {"enabled", "values"} or type(option["enabled"]) is not bool:
            raise ValueError(f"Invalid option: {name}")
        values = option["values"]
        if not isinstance(values, list) or not values:
            raise ValueError(f"Nonempty values required: {name}")
        defaults = DEFAULT_OPTIONAL_PARAMS[name]["values"]
        if len({json.dumps(v) for v in values}) != len(values) or any(
            not any(type(v) is type(d) and v == d for d in defaults) for v in values
        ):
            raise ValueError(f"Unsupported optional domain: {name}")
    if (
        set(telemetry) != set(DEFAULT_TELEMETRY)
        or type(telemetry["enabled"]) is not bool
    ):
        raise ValueError("Invalid telemetry options")
    for name in ("interval_s", "timeout_s"):
        value = telemetry[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid {name}")
    if telemetry["timeout_s"] > telemetry["interval_s"]:
        raise ValueError("Scrape timeout must not exceed interval")
    index = telemetry["gpu_index"]
    if index is not None and (type(index) is not int or index < 0):
        raise ValueError("gpu_index must be a physical NVML index")


def merge_config(fixed, sampled):
    """Sampled leaves win, retaining independent nested fixed settings."""
    result = copy.deepcopy(fixed)
    for key, value in sampled.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def graph_sizes(capacity, dense=False):
    return (
        list(range(1, capacity + 1))
        if dense
        else sorted(
            {
                1,
                capacity,
                *[2**i for i in range(capacity.bit_length()) if 2**i <= capacity],
            }
        )
    )


class TRTLLMLabV3(ExplicitLab):
    def precision_config(self, framework, weights, cache):
        if framework != "trtllm" or weights not in {"bf16", "fp8"}:
            raise ValueError("TRT v3 requires BF16 or externally exported FP8 weights")
        return {"dtype": "bfloat16"}

    def render_engine_config(self, framework, concurrency, parameters):
        config = super().render_engine_config(framework, concurrency, parameters)
        if config.get("cuda_graph_config") is not None:
            config["cuda_graph_config"]["batch_sizes"] = graph_sizes(
                config["max_batch_size"]
            )
        return config

    def expanded_config(self, trial, framework, concurrency, config, choices=None):
        # Avoid ineffective graph-profile and padding dimensions at capacity one.
        capacity = config["max_batch_size"]

        class GraphTrial:
            def __getattr__(self, name):
                return getattr(trial, name)

            def suggest_categorical(self, name, values):
                if name == "graph_capture_profile" and capacity == 1:
                    return "powers"
                return trial.suggest_categorical(name, values)

            def suggest_int(self, name, *args, **kwargs):
                if name == "graph_padding" and capacity == 1:
                    return 1
                return trial.suggest_int(name, *args, **kwargs)

        config = super().expanded_config(
            GraphTrial(), framework, concurrency, config, choices
        )
        graph = config.get("cuda_graph_config")
        if graph is not None:
            dense = getattr(trial, "params", {}).get("graph_capture_profile") == "dense"
            graph["batch_sizes"] = graph_sizes(capacity, dense)
        for name, option in self.optional_params.items():
            if not option["enabled"]:
                continue
            if name == "enable_early_first_token_response" and config.get(
                "disable_overlap_scheduler", False
            ):
                continue
            if name == "enable_in_graph_sampling" and graph is None:
                continue
            backend = config.get("prefill_cuda_graph_backend", "disabled")
            if name == "prefill_capture_profile" and backend == "disabled":
                continue
            if name == "compile_only_piecewise_graphs" and backend != "piecewise":
                continue
            value = trial.suggest_categorical(name, option["values"])
            if name == "kv_tokens_per_block":
                config["kv_cache_config"]["tokens_per_block"] = value
            elif name == "prefill_capture_profile":
                length = min(
                    self.BASE_WORKLOAD["random_input_len"], config["max_num_tokens"]
                )
                config["prefill_capture_num_tokens"] = (
                    [length]
                    if value == "workload"
                    else sorted(
                        {
                            length,
                            *[
                                s
                                for s in [512, 1024, 2048, 4096, 8192]
                                if s <= config["max_num_tokens"]
                            ],
                        }
                    )
                )
            elif name == "compile_only_piecewise_graphs":
                config.setdefault("torch_compile_config", {})[name] = value
            else:
                config[name] = value
        return config

    def sampling_menu(
        self, framework, concurrency, space_kind="baseline", choices=None
    ):
        menu = super().sampling_menu(framework, concurrency, space_kind, choices)
        if space_kind == "expanded":
            if concurrency == 1:
                menu.pop("graph_capture_profile", None)
                menu.pop("graph_padding", None)
            for name, option in self.optional_params.items():
                if option["enabled"]:
                    menu[name] = copy.deepcopy(option["values"])
        return menu

    def serialize_engine_config(self, framework, concurrency, config):
        import yaml

        native = copy.deepcopy(config)
        forbidden = {
            "quantization",
            "kv_cache_dtype",
            "calculate_kv_scales",
            "quant_config",
            "calib_config",
        } & native.keys()
        if forbidden:
            raise ValueError(
                f"Forbidden online precision settings: {sorted(forbidden)}"
            )
        if "config" in native or "extra_llm_api_options" in native:
            raise ValueError("Merge fixed TRT YAML explicitly")
        native["backend"] = "pytorch"
        kv = native.setdefault("kv_cache_config", {})
        if kv.get("dtype", "auto") == "auto":
            kv["dtype"] = "bfloat16"
        if native.get("disable_overlap_scheduler") and native.get(
            "enable_early_first_token_response"
        ):
            raise ValueError("Early first token requires overlap scheduling")
        if (
            native.get("prefill_cuda_graph_backend") == "breakable"
            and "torch_compile_config" in native
        ):
            raise ValueError("Breakable graphs conflict with torch_compile_config")
        if self.telemetry_options["enabled"]:
            # Counts are rank-0-owned: restrict telemetry to this single-GPU campaign.
            for key in (
                "tp_size",
                "tensor_parallel_size",
                "pp_size",
                "pipeline_parallel_size",
                "cp_size",
                "context_parallel_size",
                "ep_size",
                "moe_expert_parallel_size",
            ):
                if native.get(key, 1) not in (None, 1):
                    raise ValueError("TRT v3 telemetry requires single-GPU serving")
            if native.get("num_serve_frontends", 1) != 1:
                raise ValueError("TRT v3 telemetry requires one HTTP frontend")
            native["num_serve_frontends"] = 1
            native["enable_iter_perf_stats"] = True
        if self.RUN_REAL_GPU:
            verify_runtime()
            from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

            TorchLlmArgs(**{**native, "model": self.MODEL_SNAPSHOTS["A"]})
        nested = {
            key: native.pop(key) for key in sorted(V3_NATIVE_FIELDS) if key in native
        }
        digest = hashlib.sha256(
            json.dumps(nested, sort_keys=True).encode()
        ).hexdigest()[:16]
        path = Path(self.RUN_DIR) / f"trt-config-{digest}.yaml"
        path.write_text(yaml.safe_dump(nested, sort_keys=True))
        native["config"] = str(path)
        for key, value in native.items():
            if key not in V3_SERVER_CONFIGS:
                raise ValueError(f"Unknown TRT CLI field: {key}")
            if value is None or value is False or isinstance(value, (dict, list)):
                raise ValueError(f"Unsupported CLI value: {key}={value}")
        return native

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
        trial.set_user_attr("request_throughput", execution.get("request_throughput"))


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
    installed = importlib.metadata.version("tensorrt_llm")
    if installed != "1.4.0rc3":
        raise RuntimeError(f"Expected TensorRT-LLM 1.4.0rc3; found {installed}")


def resolve_cli():
    """Require the kernel and CLI to load the same pinned TRT package."""
    verify_runtime()
    import tensorrt_llm

    cli = shutil.which("trtllm-serve")
    if cli is None:
        raise RuntimeError("trtllm-serve is absent from PATH")
    shebang = Path(cli).open().readline().strip()
    interpreter = shlex.split(shebang[2:]) if shebang.startswith("#!") else []
    if interpreter and Path(interpreter[0]).name == "env":
        interpreter = interpreter[1:]
    if not interpreter or not Path(interpreter[0]).name.startswith("python"):
        raise RuntimeError(f"Cannot validate CLI interpreter: {shebang}")
    probe = subprocess.run(
        interpreter
        + [
            "-c",
            "import json,tensorrt_llm; from importlib.metadata import version; "
            "print('TRT_ENV='+json.dumps([version('tensorrt_llm'),tensorrt_llm.__file__]))",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    rows = [
        line.removeprefix("TRT_ENV=")
        for line in probe.stdout.splitlines()
        if line.startswith("TRT_ENV=")
    ]
    actual = json.loads(rows[-1]) if rows else None
    if (
        not actual
        or actual[0] != "1.4.0rc3"
        or Path(actual[1]).resolve() != Path(tensorrt_llm.__file__).resolve()
    ):
        raise RuntimeError("Kernel and serving CLI load different TRT packages")
    return str(Path(cli).absolute())


def runtime_help():
    result = subprocess.run(
        [resolve_cli(), "serve", "--help"],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", result.stdout + result.stderr)


def audit_command(command, help_text):
    advertised = set(re.findall(r"--[a-z][a-z0-9_-]*", help_text))
    used = {
        token.split("=", 1)[0]
        for token in shlex.split(command)
        if token.startswith("--")
    }
    if used - advertised:
        raise ValueError(f"CLI does not advertise {sorted(used - advertised)}")


def native_command(lab, config, model, concurrency=8, port=8000, executable=None):
    native = lab.serialize_engine_config("trtllm", concurrency, config)
    arguments = [
        executable or resolve_cli(),
        "serve",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    for name, value in native.items():
        arguments.append("--" + name)
        if not isinstance(value, bool):
            arguments.append(str(value))
    return shlex.join(arguments)


class TRTLLMRunnerV3(Phase1Runner):
    def __init__(
        self, settings, *args, optional_params=None, telemetry_options=None, **kwargs
    ):
        optional = copy.deepcopy(
            DEFAULT_OPTIONAL_PARAMS if optional_params is None else optional_params
        )
        telemetry = {**DEFAULT_TELEMETRY, **(telemetry_options or {})}
        validate_options(optional, telemetry)
        settings = copy.deepcopy(settings)
        if settings["frameworks"] != ["trtllm"]:
            raise ValueError("TRT v3 is a dedicated TRT runner")
        if settings["run_real_gpu"]:
            verify_runtime()
        identity = {
            "adapter": "trtllm-v3-1.4.0rc3",
            "optional_params": optional,
            "telemetry": telemetry,
            "implementation_sha256": {
                name: hashlib.sha256(
                    Path(__file__).with_name(name).read_bytes()
                ).hexdigest()
                for name in (
                    "trtllm_policy_v3.py",
                    "runtime_trtllm_v3.py",
                    "telemetry_trtllm_v3.py",
                )
            },
        }
        for frameworks in settings["real_preflight"].values():
            frameworks["trtllm"]["trtllm_v3"] = identity
        super().__init__(settings, *args, **kwargs)
        self.lab.__class__ = TRTLLMLabV3
        self.lab.optional_params = optional
        self.lab.telemetry_options = telemetry
        from .runtime_trtllm_v3 import RepositoryBenchmarkV3

        self.lab.RepositoryBenchmark = partial(RepositoryBenchmarkV3, _context=self.lab)
        original_logger = self.lab.NotebookWandbLogger
        self.lab.NotebookWandbLogger = lambda study, contract: V3Logger(
            original_logger(study, contract), contract
        )
