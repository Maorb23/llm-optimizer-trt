"""Validated, JSON-serializable remote optimization contract."""

from __future__ import annotations

import os
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
    server_config_files: dict[str, dict[str, str]] = field(default_factory=dict)
    search_strategy: str = "grid"
    constraints: str | None = None
    optuna: dict[str, Any] = field(default_factory=dict)
    optuna_by_framework: dict[str, dict[str, Any]] = field(default_factory=dict)
    prefix: str = "llm-optimizer"
    timeout_minutes: int = 180
    dry_run: bool = True
    preemptible: bool = False
    spot_pricing_policy_id: str | None = None
    follows_spot_price: bool | None = None
    semantic_context: dict[str, Any] | None = None
    workload_manifest: dict[str, Any] | None = None

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
        policy = request.spot_pricing_policy_id
        if request.preemptible and not any(
            key in data for key in ("spot_pricing_policy_id", "follows_spot_price")
        ):
            policy = os.getenv("NEBIUS_SPOT_PRICING_POLICY_ID", "").strip() or None
            request.spot_pricing_policy_id = policy
        if policy is not None and (not isinstance(policy, str) or not policy.strip()):
            raise ValueError("spot_pricing_policy_id must be a nonempty string")
        if isinstance(policy, str):
            request.spot_pricing_policy_id = policy.strip()
        if request.follows_spot_price is None:
            request.follows_spot_price = request.preemptible and policy is None
        if not isinstance(request.follows_spot_price, bool):
            raise ValueError("follows_spot_price must be a boolean")
        if policy and request.follows_spot_price:
            raise ValueError(
                "Choose spot_pricing_policy_id or follows_spot_price, not both"
            )
        if not request.preemptible and (policy or request.follows_spot_price):
            raise ValueError("Spot pricing requires preemptible execution")
        if request.preemptible and not policy and not request.follows_spot_price:
            raise ValueError(
                "Preemptible execution requires a spot pricing policy "
                "or follows_spot_price"
            )
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
        if not isinstance(request.server_config_files, dict) or any(
            framework not in request.frameworks
            or not isinstance(files, dict)
            or any(
                not isinstance(name, str)
                or not name or name in {".", ".."}
                or "/" in name or "\\" in name
                or not isinstance(content, str)
                for name, content in files.items()
            )
            for framework, files in request.server_config_files.items()
        ):
            raise ValueError(
                "server_config_files must map selected frameworks "
                "to safe filenames and text"
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
            "objectives",
            "seed",
            "sampler",
            "sampler_kwargs",
            "pruner",
            "pruner_kwargs",
            "min_completed_requests",
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
            if "objectives" in config:
                from llm_optimizer.optimization import objective_direction

                names = config["objectives"]
                if (
                    not isinstance(names, list)
                    or len(names) < 1
                    or any(not isinstance(name, str) for name in names)
                    or len(set(names)) != len(names)
                    or "objective" in config
                ):
                    raise ValueError(
                        "objectives must be a unique list of at least one metric; "
                        "omit objective"
                    )
                for name in names:
                    objective_direction(name)
            minimum = config.get("min_completed_requests", 1)
            if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
                raise ValueError("min_completed_requests must be a positive integer")
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
        for framework in request.frameworks:
            merged = request.optuna_for(framework)
            if merged.get("pruner", "NopPruner") != "NopPruner":
                if (
                    len(
                        merged.get(
                            "objectives", [merged.get("objective", "output_throughput")]
                        )
                    )
                    != 1
                    or request.constraints
                ):
                    raise ValueError(
                        "Pruning requires an unconstrained scalar objective"
                    )
                if merged.get("pruner") == "ThresholdPruner":
                    raise ValueError(
                        "GPU Threshold pruning requires independent "
                        "workload calibration"
                    )
            if "objective" in merged and "objectives" in merged:
                raise ValueError(
                    "Merged Optuna settings must omit objective when objectives is set"
                )
        if any(arg.partition('=')[0].startswith('hf_dataset') for arg in request.client_args):
            raise ValueError('Remote HF datasets must be prepared once as workload_manifest before submission')
        if request.workload_manifest is not None:
            from llm_optimizer.hf_workload import validate_manifest
            validate_manifest(request.workload_manifest)
            if any(arg.partition('=')[0] in {'dataset_path', 'workload_path'} for arg in request.client_args):
                raise ValueError('Custom workload paths are managed by the worker')
            if any(config.get('client_args') for config in [request.optuna, *request.optuna_by_framework.values()]):
                raise ValueError('Custom workloads cannot be overridden by Optuna')
        return request

    def optuna_for(self, framework: str) -> dict[str, Any]:
        return {**self.optuna, **self.optuna_by_framework.get(framework, {})}

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        # Preserve the payload accepted by existing images for default workloads.
        if self.workload_manifest is None:
            data.pop("workload_manifest")
        return data
