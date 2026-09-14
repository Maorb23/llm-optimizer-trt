"""NVIDIA Dynamo deployment validation and manifest generation."""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any

from llm_optimizer.result_utils import normalize_config

SUPPORTED_BACKENDS = ("vllm", "sglang", "trtllm")

_IMAGES = {
    "vllm": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.0",
    "sglang": "nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.4.0",
    "trtllm": "nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.4.0",
}
_MODULES = {
    "vllm": "dynamo.vllm",
    "sglang": "dynamo.sglang",
    "trtllm": "dynamo.trtllm",
}
_MODEL_FLAGS = {"vllm": "--model", "sglang": "--model-path", "trtllm": "--model-path"}
_BATCH_PARAMETERS = {
    "vllm": ("max_num_seqs", "max_num_batched_tokens"),
    "sglang": ("max_running_requests", "max_total_tokens"),
    "trtllm": ("max_batch_size", "max_num_tokens"),
}
_ALIASES = {
    "tensor_parallel_size": ("tensor_parallel_size", "tp_size"),
    "pipeline_parallel_size": ("pipeline_parallel_size", "pp_size"),
    "context_parallel_size": ("context_parallel_size", "cp_size"),
}


@dataclass(frozen=True)
class DynamoValidation:
    """Validation result for one engine configuration."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    normalized_config: dict[str, Any] = field(default_factory=dict)
    worker_gpu_count: int = 1
    cluster_gpu_count: int = 1
    batch_parameters: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "normalized_config": self.normalized_config,
            "worker_gpu_count": self.worker_gpu_count,
            "cluster_gpu_count": self.cluster_gpu_count,
            "batch_parameters": self.batch_parameters,
        }


@dataclass(frozen=True)
class AIConfiguratorRequest:
    """Inputs for NVIDIA AI Configurator's GPU-free sizing sweep."""

    model: str
    framework: str
    gpu_system: str
    total_gpus: int
    input_length: int = 4000
    output_length: int = 1000
    ttft_ms: float = 2000.0
    tpot_ms: float = 30.0
    backend_version: str | None = None
    namespace: str = "dynamo"


class AIConfiguratorError(RuntimeError):
    """Raised when AI Configurator is missing or its sizing sweep fails."""


def _run_aiconfigurator_command(
    command: list[str], timeout_seconds: int
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        raise AIConfiguratorError(
            "aiconfigurator is not installed; install the 'aisimulate' package "
            "with Python 3.11 through 3.13"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise AIConfiguratorError(
            f"AI Configurator exceeded its {timeout_seconds}s timeout"
        ) from exc


def _read_ranked_configurations(save_dir: str) -> dict[str, list[dict[str, str]]]:
    ranked: dict[str, list[dict[str, str]]] = {}
    for topology in ("agg", "disagg"):
        path = os.path.join(save_dir, topology, "best_config_topn.csv")
        if not os.path.isfile(path):
            continue
        with open(path, newline="", encoding="utf-8") as csv_file:
            ranked[topology] = list(csv.DictReader(csv_file))
    return ranked


def _list_aiconfigurator_artifacts(save_dir: str) -> list[str]:
    artifacts: list[str] = []
    for root, _, filenames in os.walk(save_dir):
        for filename in filenames:
            artifacts.append(os.path.relpath(os.path.join(root, filename), save_dir))
    return sorted(artifacts)


def run_ai_configurator(
    request: AIConfiguratorRequest,
    save_dir: str,
    *,
    executable: str = "aiconfigurator",
    timeout_seconds: int = 600,
) -> dict[str, Any]:
    """Run AI Configurator support validation and its offline sizing sweep."""
    if request.framework not in SUPPORTED_BACKENDS:
        raise ValueError(f"framework must be one of: {', '.join(SUPPORTED_BACKENDS)}")
    numeric_inputs = {
        "total_gpus": request.total_gpus,
        "input_length": request.input_length,
        "output_length": request.output_length,
        "ttft_ms": request.ttft_ms,
        "tpot_ms": request.tpot_ms,
    }
    for name, value in numeric_inputs.items():
        if value <= 0:
            raise ValueError(f"{name} must be greater than zero")

    support_command = [
        executable,
        "cli",
        "support",
        "--model-path",
        request.model,
        "--system",
        request.gpu_system,
        "--backend",
        request.framework,
    ]
    if request.backend_version:
        support_command.extend(("--backend-version", request.backend_version))
    support = _run_aiconfigurator_command(support_command, timeout_seconds)
    support_output = (support.stdout + support.stderr).strip()
    if support.returncode != 0:
        raise AIConfiguratorError(
            "AI Configurator does not support this model/system/backend "
            f"combination: {support_output or 'support check failed'}"
        )

    if os.path.isdir(save_dir) and os.listdir(save_dir):
        raise AIConfiguratorError(
            f"AI Configurator output directory is not empty: {save_dir}"
        )
    os.makedirs(save_dir, exist_ok=True)
    optimize_command = [
        executable,
        "cli",
        "default",
        "--model-path",
        request.model,
        "--total-gpus",
        str(request.total_gpus),
        "--system",
        request.gpu_system,
        "--backend",
        request.framework,
        "--isl",
        str(request.input_length),
        "--osl",
        str(request.output_length),
        "--ttft",
        str(request.ttft_ms),
        "--tpot",
        str(request.tpot_ms),
        "--database-mode",
        "SILICON",
        "--deployment-target",
        "dynamo-j2",
        "--generator-set",
        f"K8sConfig.k8s_namespace={request.namespace}",
        "--save-dir",
        save_dir,
    ]
    if request.backend_version:
        optimize_command.extend(("--backend-version", request.backend_version))
    optimized = _run_aiconfigurator_command(optimize_command, timeout_seconds)
    optimizer_output = (optimized.stdout + optimized.stderr).strip()
    if optimized.returncode != 0:
        raise AIConfiguratorError(
            f"AI Configurator optimization failed: {optimizer_output or 'no output'}"
        )

    return {
        "status": "complete",
        "gpu_required": False,
        "support_check": support_output,
        "optimizer_output": optimizer_output,
        "save_dir": save_dir,
        "ranked_configurations": _read_ranked_configurations(save_dir),
        "artifacts": _list_aiconfigurator_artifacts(save_dir),
        "commands": {
            "support": support_command,
            "optimize": optimize_command,
        },
    }


def _positive_alias(
    server_args: dict[str, Any], semantic_name: str
) -> tuple[int, list[str]]:
    values = [
        (name, server_args[name])
        for name in _ALIASES[semantic_name]
        if name in server_args
    ]
    errors: list[str] = []
    if not values:
        return 1, errors
    converted = []
    for name, value in values:
        try:
            converted.append((name, int(value)))
        except (TypeError, ValueError):
            errors.append(f"{name} must be an integer")
    if not converted:
        return 1, errors
    if len({value for _, value in converted}) > 1:
        errors.append(
            "conflicting aliases: "
            + ", ".join(f"{name}={value}" for name, value in converted)
        )
    value = converted[0][1]
    if value < 1:
        errors.append(f"{converted[0][0]} must be at least 1")
    return max(1, value), errors


def validate_dynamo_config(
    framework: str,
    server_args: dict[str, Any],
    node_count: int,
    gpus_per_node: int,
) -> DynamoValidation:
    """Validate backend parameters against a Dynamo multinode worker topology.

    Batch limits are engine/worker limits under continuous batching. They are
    intentionally reported once per worker, not multiplied per Kubernetes node.
    """
    framework = framework.lower()
    errors: list[str] = []
    warnings: list[str] = []
    if framework not in SUPPORTED_BACKENDS:
        errors.append(f"Dynamo backend must be one of: {', '.join(SUPPORTED_BACKENDS)}")
    if node_count < 1:
        errors.append("node_count must be at least 1")
    if gpus_per_node < 1:
        errors.append("gpus_per_node must be at least 1")

    dimensions: dict[str, int] = {}
    for semantic_name in _ALIASES:
        dimensions[semantic_name], alias_errors = _positive_alias(
            server_args, semantic_name
        )
        errors.extend(alias_errors)

    worker_gpu_count = (
        dimensions["tensor_parallel_size"]
        * dimensions["pipeline_parallel_size"]
        * dimensions["context_parallel_size"]
    )
    cluster_gpu_count = max(0, node_count) * max(0, gpus_per_node)
    if worker_gpu_count != cluster_gpu_count:
        errors.append(
            "distributed worker GPU count mismatch: "
            f"TP×PP×CP={worker_gpu_count}, but "
            f"node_count×gpus_per_node={cluster_gpu_count}"
        )
    if node_count > 1 and dimensions["pipeline_parallel_size"] == 1:
        warnings.append(
            "multinode TP without pipeline parallelism is communication-heavy; "
            "prefer TP within a node and PP across nodes"
        )

    topology_parameters = {
        "gpus_per_node": gpus_per_node,
        "nnodes": node_count,
    }
    applicable_topology_parameters = {
        "trtllm": ("gpus_per_node",),
        "sglang": ("nnodes",),
    }
    for parameter in applicable_topology_parameters.get(framework, ()):
        if parameter in server_args and int(server_args[parameter]) != int(
            topology_parameters[parameter]
        ):
            errors.append(
                f"{parameter}={server_args[parameter]} conflicts with Dynamo "
                f"topology value {topology_parameters[parameter]}"
            )

    batch_parameters: dict[str, int] = {}
    if framework in _BATCH_PARAMETERS:
        for name in _BATCH_PARAMETERS[framework]:
            if name not in server_args:
                continue
            try:
                value = int(server_args[name])
                if value < 1:
                    errors.append(f"{name} must be at least 1")
                else:
                    batch_parameters[name] = value
            except (TypeError, ValueError):
                errors.append(f"{name} must be an integer")
        if not batch_parameters:
            warnings.append(
                "no explicit batch/concurrency limit supplied; "
                "the backend default will be used"
            )

    normalized = normalize_config(framework, server_args)
    normalized.update(dimensions)
    normalized["world_size"] = worker_gpu_count
    normalized["node_count"] = node_count
    normalized["gpus_per_node"] = gpus_per_node
    normalized["batch_scope"] = "distributed_worker"
    return DynamoValidation(
        valid=not errors,
        errors=errors,
        warnings=warnings,
        normalized_config=normalized,
        worker_gpu_count=worker_gpu_count,
        cluster_gpu_count=cluster_gpu_count,
        batch_parameters=batch_parameters,
    )


def _kebab(name: str) -> str:
    return name.replace("_", "-")


def _safe_name(model: str, framework: str) -> str:
    name = re.sub(r"[^a-z0-9-]+", "-", f"{model}-{framework}".lower()).strip("-")
    return name[-63:].strip("-") or f"llm-{framework}"


def _engine_args(framework: str, server_args: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for name, value in server_args.items():
        if name in {"host", "port", "model", "model_path"}:
            continue
        option = f"--{_kebab(name)}"
        if isinstance(value, bool):
            if value:
                result.append(option)
        else:
            result.extend((option, str(value)))
    return result


def build_dynamo_manifest(
    framework: str,
    model: str,
    server_args: dict[str, Any],
    node_count: int,
    gpus_per_node: int,
    *,
    namespace: str = "dynamo",
    name: str | None = None,
    image: str | None = None,
    hf_secret: str | None = "hf-token-secret",
) -> dict[str, Any]:
    """Build a Kubernetes DynamoGraphDeployment for a validated configuration."""
    validation = validate_dynamo_config(
        framework, server_args, node_count, gpus_per_node
    )
    if not validation.valid:
        raise ValueError("; ".join(validation.errors))

    framework = framework.lower()
    deployment_name = name or _safe_name(model, framework)
    runtime_image = image or _IMAGES[framework]
    args = [_MODEL_FLAGS[framework], model, *_engine_args(framework, server_args)]
    container: dict[str, Any] = {
        "name": "main",
        "image": runtime_image,
        "command": ["python3", "-m", _MODULES[framework]],
        "args": args,
        "resources": {"limits": {"nvidia.com/gpu": str(gpus_per_node)}},
    }
    if hf_secret:
        container["envFrom"] = [{"secretRef": {"name": hf_secret}}]
    worker: dict[str, Any] = {
        "name": "worker",
        "type": "worker",
        "replicas": 1,
        "podTemplate": {"spec": {"containers": [container]}},
    }
    if node_count > 1:
        worker["multinode"] = {"nodeCount": node_count}

    return {
        "apiVersion": "nvidia.com/v1beta1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": deployment_name, "namespace": namespace},
        "spec": {
            "backendFramework": framework,
            "components": [
                {
                    "name": "Frontend",
                    "type": "frontend",
                    "replicas": 1,
                    "podTemplate": {
                        "spec": {
                            "containers": [{"name": "main", "image": runtime_image}]
                        }
                    },
                },
                worker,
            ],
        },
    }


def kubernetes_server_dry_run(manifest: dict[str, Any]) -> tuple[bool, str]:
    """Ask the active Kubernetes API server to validate a generated DGD."""
    try:
        completed = subprocess.run(
            ["kubectl", "apply", "--dry-run=server", "-f", "-"],
            input=json.dumps(manifest),
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        return False, "kubectl is not installed or not on PATH"
    output = (completed.stdout + completed.stderr).strip()
    return completed.returncode == 0, output


def kubernetes_file_server_dry_run(path: str) -> tuple[bool, str]:
    """Ask Kubernetes to validate an AI Configurator-generated manifest file."""
    try:
        completed = subprocess.run(
            ["kubectl", "apply", "--dry-run=server", "-f", path],
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        return False, "kubectl is not installed or not on PATH"
    output = (completed.stdout + completed.stderr).strip()
    return completed.returncode == 0, output
