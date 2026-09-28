"""Validated, JSON-serializable remote optimization contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from llm_optimizer.predefined import SERVER_CONFIGS


@dataclass
class OptimizationRequest:
    model: str
    frameworks: list[str]
    gpu_type: str
    images: dict[str, str]
    bucket: str
    revision: str | None = None
    speculative_heads: str | None = None
    gpu_count: int = 1
    preset: str = "1gpu-16vcpu-96gb"
    client_args: list[str] = field(default_factory=list)
    server_args: dict[str, list[str]] = field(default_factory=dict)
    search_strategy: str = "grid"
    constraints: str | None = None
    optuna: dict[str, Any] = field(default_factory=dict)
    optuna_by_framework: dict[str, dict[str, Any]] = field(default_factory=dict)
    prefix: str = "llm-optimizer"
    timeout_minutes: int = 180
    dry_run: bool = True
    preemptible: bool = False

    @classmethod
    def parse(cls, data: dict[str, Any]) -> OptimizationRequest:
        if not isinstance(data, dict):
            raise ValueError("request must be a JSON object")
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown request fields: {sorted(unknown)}")
        try:
            request = cls(**data)
        except TypeError as exc:
            raise ValueError(str(exc)) from exc
        if not isinstance(request.model, str) or not request.model.strip():
            raise ValueError("model is required")
        if (
            not isinstance(request.frameworks, list)
            or not request.frameworks
            or len(set(request.frameworks)) != len(request.frameworks)
        ):
            raise ValueError("frameworks must be a nonempty unique list")
        if any(name not in SERVER_CONFIGS for name in request.frameworks):
            raise ValueError(
                f"unsupported framework; choose from {sorted(SERVER_CONFIGS)}"
            )
        if not isinstance(request.images, dict) or any(
            not request.images.get(name) for name in request.frameworks
        ):
            raise ValueError("a container image is required for each framework")
        if not isinstance(request.gpu_type, str) or not request.gpu_type.startswith(
            "gpu-"
        ):
            raise ValueError(
                "gpu_type must be a Nebius GPU platform, for example gpu-h100-sxm"
            )
        if not isinstance(request.gpu_count, int) or request.gpu_count < 1:
            raise ValueError("gpu_count must be positive")
        if not isinstance(request.preemptible, bool):
            raise ValueError("preemptible must be a boolean")
        if not isinstance(request.preset, str) or not request.preset.startswith(
            f"{request.gpu_count}gpu-"
        ):
            raise ValueError("preset must match gpu_count")
        if (
            not isinstance(request.timeout_minutes, int)
            or not 60 <= request.timeout_minutes <= 10080
        ):
            raise ValueError("timeout_minutes must be between 60 and 10080")
        if not isinstance(request.bucket, str) or (
            not request.bucket and not request.dry_run
        ):
            raise ValueError("bucket is required for live runs")
        if request.search_strategy not in {"grid", "optuna"}:
            raise ValueError("search_strategy must be grid or optuna")
        if not isinstance(request.client_args, list) or any(
            not isinstance(x, str) for x in request.client_args
        ):
            raise ValueError("client_args must be a list of strings")
        if not isinstance(request.server_args, dict) or any(
            name not in request.frameworks
            or not isinstance(args, list)
            or any(not isinstance(x, str) for x in args)
            for name, args in request.server_args.items()
        ):
            raise ValueError(
                "server_args must map selected frameworks to lists of strings"
            )
        if request.speculative_heads and any(
            not any("speculative" in arg for arg in request.server_args.get(name, []))
            for name in request.frameworks
        ):
            raise ValueError(
                "speculative_heads requires speculative server_args for every framework"
            )
        allowed = {
            "trials",
            "timeout",
            "objective",
            "seed",
            "sampler",
            "sampler_kwargs",
            "pruner",
            "pruner_kwargs",
            "storage",
            "study_name",
            "server_args",
            "client_args",
        }
        if not isinstance(request.optuna, dict) or set(request.optuna) - allowed:
            raise ValueError("unsupported Optuna setting")
        if request.search_strategy != "optuna" and (
            request.optuna or request.optuna_by_framework
        ):
            raise ValueError("Optuna settings require search_strategy=optuna")
        if not isinstance(request.optuna_by_framework, dict) or any(
            name not in request.frameworks or not isinstance(config, dict)
            for name, config in request.optuna_by_framework.items()
        ):
            raise ValueError(
                "optuna_by_framework must map selected frameworks to objects"
            )
        for config in [request.optuna, *request.optuna_by_framework.values()]:
            if set(config) - allowed:
                raise ValueError("unsupported Optuna setting")
            storage = config.get("storage")
            if isinstance(storage, str) and storage.startswith("sqlite:"):
                raise ValueError(
                    "remote SQLite Optuna storage is managed automatically; "
                    "omit optuna.storage or use an external RDB URL"
                )
            if "trials" in config and (
                not isinstance(config["trials"], int) or config["trials"] < 1
            ):
                raise ValueError("Optuna trials must be positive")
            for key in ("sampler_kwargs", "pruner_kwargs"):
                if key in config and not isinstance(config[key], dict):
                    raise ValueError(f"{key} must be an object")
            for key in ("server_args", "client_args"):
                if key in config and (
                    not isinstance(config[key], list)
                    or any(not isinstance(x, str) for x in config[key])
                ):
                    raise ValueError(f"{key} must be a list of strings")
        return request

    def optuna_for(self, framework: str) -> dict[str, Any]:
        return {**self.optuna, **self.optuna_by_framework.get(framework, {})}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
