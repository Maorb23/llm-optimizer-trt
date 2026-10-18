"""TensorRT-LLM native ``trtllm-serve`` arguments and lifecycle metadata.

The list follows the supported PyTorch-backend CLI in TensorRT-LLM main.  More
specialized nested LLM API settings remain available through the native
``config`` YAML argument instead of being represented by invented flat flags.
"""

import typing as t

from llm_optimizer.args import ArgConfig, ConfigsDict

PARAMETER_MAPPING = {
    "tensor_parallel": "tp_size",
    "data_parallel": None,
    "max_concurrent_requests": "max_batch_size",
    "prefill_chunk_size": None,
    "batch_size": "max_num_tokens",
}

# Values used to construct the engine/executor. A change causes a fresh engine
# initialization when the managed server is restarted. Current TensorRT-LLM no
# longer has a separate trtllm-build stage for this serving backend.
ENGINE_INIT_PARAMETERS = frozenset(
    {
        "backend",
        "config",
        "extra_llm_api_options",
        "max_beam_width",
        "max_batch_size",
        "max_num_tokens",
        "max_seq_len",
        "tensor_parallel_size",
        "tp_size",
        "pipeline_parallel_size",
        "pp_size",
        "context_parallel_size",
        "cp_size",
        "moe_expert_parallel_size",
        "ep_size",
        "gpus_per_node",
        "free_gpu_memory_fraction",
        "kv_cache_free_gpu_memory_fraction",
        "kv_cache_dtype",
        "enable_chunked_prefill",
        "enable_attention_dp",
    }
)

SERVER_CONFIGS: ConfigsDict = {
    # Model and executor
    "tokenizer": ArgConfig[str](name="tokenizer"),
    "custom_tokenizer": ArgConfig[str](name="custom_tokenizer"),
    "backend": ArgConfig[str](name="backend"),
    "generation_config": ArgConfig[str](name="generation_config"),
    "max_beam_width": ArgConfig[int](name="max_beam_width"),
    "max_batch_size": ArgConfig[int](name="max_batch_size"),
    "max_num_tokens": ArgConfig[int](name="max_num_tokens"),
    "max_seq_len": ArgConfig[int](name="max_seq_len"),
    "trust_remote_code": ArgConfig[bool](name="trust_remote_code"),
    "hf_revision": ArgConfig[str](name="hf_revision"),
    "revision": ArgConfig[str](name="revision"),
    "config": ArgConfig[str](name="config"),
    "extra_llm_api_options": ArgConfig[str](name="extra_llm_api_options"),
    # Parallelism
    "tensor_parallel_size": ArgConfig[int](name="tensor_parallel_size"),
    "tp_size": ArgConfig[int](name="tp_size"),
    "pipeline_parallel_size": ArgConfig[int](name="pipeline_parallel_size"),
    "pp_size": ArgConfig[int](name="pp_size"),
    "context_parallel_size": ArgConfig[int](name="context_parallel_size"),
    "cp_size": ArgConfig[int](name="cp_size"),
    "moe_expert_parallel_size": ArgConfig[int](name="moe_expert_parallel_size"),
    "ep_size": ArgConfig[int](name="ep_size"),
    "gpus_per_node": ArgConfig[int](name="gpus_per_node"),
    "enable_attention_dp": ArgConfig[bool](name="enable_attention_dp"),
    # KV cache and batching
    "free_gpu_memory_fraction": ArgConfig[float](name="free_gpu_memory_fraction"),
    "kv_cache_free_gpu_memory_fraction": ArgConfig[float](
        name="kv_cache_free_gpu_memory_fraction"
    ),
    "kv_cache_dtype": ArgConfig[str](name="kv_cache_dtype"),
    "enable_chunked_prefill": ArgConfig[bool](name="enable_chunked_prefill"),
    # HTTP serving and observability
    "host": ArgConfig[str](name="host"),
    "port": ArgConfig[int](name="port"),
    "log_level": ArgConfig[str](name="log_level"),
    "num_postprocess_workers": ArgConfig[int](name="num_postprocess_workers"),
    "num_serve_frontends": ArgConfig[int](name="num_serve_frontends"),
    "num_input_processor_workers": ArgConfig[int](name="num_input_processor_workers"),
    "num_media_load_workers": ArgConfig[int](name="num_media_load_workers"),
    "reasoning_parser": ArgConfig[str](name="reasoning_parser"),
    "tool_parser": ArgConfig[str](name="tool_parser"),
    "chat_template": ArgConfig[str](name="chat_template"),
    "otlp_traces_endpoint": ArgConfig[str](name="otlp_traces_endpoint"),
    "telemetry": ArgConfig[bool](name="telemetry"),
    "server_role": ArgConfig[str](name="server_role"),
    "disagg_cluster_uri": ArgConfig[str](name="disagg_cluster_uri"),
}


def classify_server_args(
    server_args: dict[str, t.Any],
) -> tuple[dict[str, t.Any], dict[str, t.Any]]:
    """Split native arguments into engine-init and HTTP/runtime settings."""
    engine_args = {
        key: value
        for key, value in server_args.items()
        if key in ENGINE_INIT_PARAMETERS
    }
    runtime_args = {
        key: value
        for key, value in server_args.items()
        if key not in ENGINE_INIT_PARAMETERS
    }
    return engine_args, runtime_args


def required_gpu_count(server_args: dict[str, t.Any]) -> int:
    """Return the minimum local GPU count implied by TRT parallel dimensions."""
    tp = _parallel_value(server_args, "tp_size", "tensor_parallel_size")
    pp = _parallel_value(server_args, "pp_size", "pipeline_parallel_size")
    cp = _parallel_value(server_args, "cp_size", "context_parallel_size")
    return tp * pp * cp


def _parallel_value(
    server_args: dict[str, t.Any], short_name: str, long_name: str
) -> int:
    short_value = server_args.get(short_name)
    long_value = server_args.get(long_name)
    if (
        short_value is not None
        and long_value is not None
        and int(short_value) != int(long_value)
    ):
        raise ValueError(
            f"Conflicting TensorRT-LLM aliases: {short_name}={short_value} and "
            f"{long_name}={long_value}."
        )
    value = short_value if short_value is not None else long_value
    value = 1 if value is None else int(value)
    if value < 1:
        raise ValueError(f"{short_name} must be at least 1")
    return value
