"""Backward-compatible result and cross-framework configuration helpers."""

import copy
import typing as t

NORMALIZED_PARAMETER_ALIASES: dict[str, dict[str, tuple[str, ...]]] = {
    "vllm": {
        "tensor_parallel_size": ("tensor_parallel_size",),
        "pipeline_parallel_size": ("pipeline_parallel_size",),
        "data_parallel_size": ("data_parallel_size",),
        "max_batched_tokens": ("max_num_batched_tokens",),
        "max_concurrent_sequences": ("max_num_seqs",),
        "max_sequence_length": ("max_model_len",),
        "kv_cache_memory_fraction": ("gpu_memory_utilization",),
        "kv_cache_dtype": ("kv_cache_dtype",),
        "engine_precision": ("dtype", "quantization"),
    },
    "sglang": {
        "tensor_parallel_size": ("tp_size", "tensor_parallel_size"),
        "pipeline_parallel_size": ("pp_size", "pipeline_parallel_size"),
        "data_parallel_size": ("dp_size", "data_parallel_size"),
        "max_batched_tokens": ("max_total_tokens",),
        "max_concurrent_sequences": ("max_running_requests",),
        "max_sequence_length": ("context_length",),
        "kv_cache_memory_fraction": ("mem_fraction_static",),
        "kv_cache_dtype": ("kv_cache_dtype",),
        "engine_precision": ("dtype", "quantization"),
    },
    "trtllm": {
        "tensor_parallel_size": ("tp_size", "tensor_parallel_size"),
        "pipeline_parallel_size": ("pp_size", "pipeline_parallel_size"),
        "context_parallel_size": ("cp_size", "context_parallel_size"),
        "expert_parallel_size": ("ep_size", "moe_expert_parallel_size"),
        "max_batched_tokens": ("max_num_tokens",),
        "max_concurrent_sequences": ("max_batch_size",),
        "max_sequence_length": ("max_seq_len",),
        "kv_cache_memory_fraction": (
            "free_gpu_memory_fraction",
            "kv_cache_free_gpu_memory_fraction",
        ),
        "kv_cache_dtype": ("kv_cache_dtype",),
    },
}


def infer_framework(record: dict[str, t.Any]) -> t.Optional[str]:
    """Read framework identity from new records or infer it for legacy files."""
    config = record.get("config", {})
    framework = config.get("framework") or record.get("framework")
    if framework:
        return str(framework).lower()

    command = str(record.get("cmd", "")).lower()
    if "trtllm-serve" in command:
        return "trtllm"
    if "sglang" in command:
        return "sglang"
    if "vllm" in command:
        return "vllm"
    if "max serve" in command:
        return "max"
    return None


def normalize_config(
    framework: t.Optional[str],
    server_args: dict[str, t.Any],
    client_args: t.Optional[dict[str, t.Any]] = None,
) -> dict[str, t.Any]:
    """Map native names to comparable semantics without removing native data."""
    normalized: dict[str, t.Any] = {}
    if framework:
        for semantic_name, aliases in NORMALIZED_PARAMETER_ALIASES.get(
            framework, {}
        ).items():
            for alias in aliases:
                if alias in server_args:
                    normalized[semantic_name] = server_args[alias]
                    break

    client_args = client_args or {}
    for key in (
        "max_concurrency",
        "request_rate",
        "random_input_len",
        "random_output_len",
        "num_prompts",
    ):
        if key in client_args:
            normalized[key] = client_args[key]
    if "random_input_len" in client_args:
        normalized["input_length"] = client_args["random_input_len"]
    if "random_output_len" in client_args:
        normalized["output_length"] = client_args["random_output_len"]

    parallel_dimensions = (
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "context_parallel_size",
        "data_parallel_size",
    )
    if any(name in normalized for name in parallel_dimensions):
        world_size = 1
        for name in parallel_dimensions:
            world_size *= int(normalized.get(name, 1))
        normalized["world_size"] = world_size
    return normalized


def normalize_result_record(record: dict[str, t.Any]) -> dict[str, t.Any]:
    """Return a normalized copy while accepting old serialized records."""
    normalized_record = copy.deepcopy(record)
    config = normalized_record.setdefault("config", {})
    framework = infer_framework(normalized_record)
    if framework:
        config.setdefault("framework", framework)
    config.setdefault("server_args", {})
    config.setdefault("client_args", {})
    config.setdefault(
        "normalized_config",
        normalize_config(
            framework,
            config["server_args"],
            config["client_args"],
        ),
    )
    return normalized_record
