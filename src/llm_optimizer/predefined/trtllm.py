"""TensorRT-LLM native ``trtllm-serve`` arguments and lifecycle metadata.

The list follows the PyTorch-backend CLI in TensorRT-LLM 1.3.0rc26.  More
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


# Native YAML fields, not flat trtllm-serve flags (v1.3.0rc26).
NATIVE_CONFIG_FIELDS = (
    "kv_cache_config",
    "cuda_graph_config",
    "scheduler_config",
    "attn_backend",
    "enable_chunked_prefill",
    "speculative_config",
)
DRAFT_ALGORITHMS = {
    "Eagle3": "Eagle3DecodingConfig",
    "DFlash": "DFlashDecodingConfig",
    "DSpark": "DSparkDecodingConfig",
}


def validate_draft_request(algorithm, max_draft_len, draft_model_id):
    """Fail before downloads when a requested algorithm/runtime is unavailable."""
    import importlib.metadata

    if algorithm not in DRAFT_ALGORITHMS:
        raise ValueError(f"Select an explicit TRT algorithm: {tuple(DRAFT_ALGORITHMS)}")
    if not draft_model_id:
        raise ValueError("TRT speculative decoding requires a draft model ID")
    if (
        isinstance(max_draft_len, bool)
        or not isinstance(max_draft_len, int)
        or max_draft_len < 1
    ):
        raise ValueError("max_draft_len must be a positive integer")
    if importlib.metadata.version("tensorrt_llm") != "1.3.0rc26":
        raise ValueError("Draft setup is verified for TensorRT-LLM 1.3.0rc26")
    from tensorrt_llm.llmapi import llm_args

    if not hasattr(llm_args, DRAFT_ALGORITHMS[algorithm]):
        raise ValueError(f"Installed TRT-LLM does not implement {algorithm}")


def draft_config(algorithm, max_draft_len, target_path, draft_path):
    """Validate checkpoint semantics and use the installed native schema.

    Algorithm availability alone cannot establish checkpoint compatibility.
    Target-dependent drafts must use the target's hidden width and vocabulary.
    GPU loading/generation remains a separate preflight requirement.
    """
    import json
    from pathlib import Path

    from tensorrt_llm.llmapi import llm_args

    target = json.loads((Path(target_path) / "config.json").read_text())
    draft = json.loads((Path(draft_path) / "config.json").read_text())
    for key in ("hidden_size", "vocab_size"):
        if not target.get(key) or draft.get(key) != target[key]:
            raise ValueError(f"Draft and target {key} must match")
    if not target.get("architectures") or not draft.get("architectures"):
        raise ValueError("Target and draft must declare architectures")
    if algorithm == "Eagle3":
        # Generic Llama draft labels are used by the official EAGLE3 pair.
        # draft_vocab_size is optional in this release (defaults to vocab_size).
        if any(
            "eagle" in arch.lower() and "eagle3" not in arch.lower()
            for arch in draft["architectures"]
        ):
            raise ValueError("EAGLE v1/v2 checkpoints are incompatible with Eagle3")
        if draft.get("dflash_config") or draft.get("dspark_config"):
            raise ValueError("Use an EAGLE3 checkpoint for Eagle3")
        supported_architectures = {
            "LlamaForCausalLM",
            "EAGLE3LlamaForCausalLM",
            "Eagle3DeepSeekV3ForCausalLM",
        }
        if draft["architectures"][0] not in supported_architectures:
            raise ValueError(
                "Eagle3 helper supports Llama-style and DeepSeek draft architectures"
            )
    else:
        dflash = draft.get("dflash_config") or {}
        dspark = draft.get("dspark_config") or {}

        def setting(key, prefixed):
            for source, name in (
                (dspark, key),
                (dflash, key),
                (draft, prefixed),
                (draft, key),
            ):
                if source.get(name) is not None:
                    return source[name]
            return None

        has_dspark_heads = (
            str(setting("projector_type", "dspark_projector_type")).lower() == "dspark"
            or setting("shift_label", "dspark_shift_label")
            or setting("use_confidence_head", "dspark_use_confidence_head")
            or setting("enable_confidence_head", "dspark_enable_confidence_head")
            or int(setting("markov_rank", "dspark_markov_rank") or 0) > 0
        )
        if algorithm == "DFlash" and (not dflash or has_dspark_heads):
            raise ValueError("DFlash requires dflash_config without DSpark heads")
        if algorithm == "DSpark" and not has_dspark_heads:
            raise ValueError("DSpark requires a checkpoint declaring DSpark heads")
        layers = setting("target_layer_ids", "dspark_target_layer_ids")
        mask = setting("mask_token_id", "dspark_noise_token_id")
        if not layers or mask is None:
            raise ValueError("Draft must declare target_layer_ids and mask_token_id")
        if any(
            isinstance(i, bool)
            or not isinstance(i, int)
            or not 0 <= i < target.get("num_hidden_layers", 0)
            for i in layers
        ):
            raise ValueError("Draft target_layer_ids exceed target architecture")
        if (
            algorithm == "DSpark"
            and setting("block_size", "dspark_block_size") != max_draft_len
        ):
            raise ValueError("DSpark checkpoint block_size must equal max_draft_len")
    # Resolve the architecture using the installed backend's own registry/fallback.
    # The loader applies the same model_type fallback for custom DFlash labels.
    from types import SimpleNamespace

    from tensorrt_llm._torch.models.modeling_utils import get_model_architecture

    get_model_architecture(SimpleNamespace(**target))
    cfg = SimpleNamespace(**draft)
    if algorithm != "Eagle3":
        try:
            get_model_architecture(cfg)
        except RuntimeError:
            cfg.architectures = [
                "".join(w.capitalize() for w in draft.get("model_type", "").split("_"))
                + "ForCausalLM"
            ]
            get_model_architecture(cfg)
    native = {
        "decoding_type": algorithm,
        "max_draft_len": max_draft_len,
        "speculative_model": str(draft_path),
    }
    cls = getattr(llm_args, DRAFT_ALGORITHMS[algorithm])
    cls(**native)  # Validate without adding version-dependent defaults to the study.
    return native


def serialize_config(config, run_dir):
    """Move nested API options to content-addressed native YAML."""
    import hashlib
    import json
    from pathlib import Path

    native = dict(config)
    if "config" in native or "extra_llm_api_options" in native:
        raise ValueError(
            "Merge a fixed TRT YAML explicitly; do not override it silently."
        )
    nested = {key: native.pop(key) for key in NATIVE_CONFIG_FIELDS if key in native}
    if nested:
        import yaml

        digest = hashlib.sha256(
            json.dumps(nested, sort_keys=True).encode()
        ).hexdigest()[:16]
        path = Path(run_dir) / f"trt-config-{digest}.yaml"
        path.write_text(yaml.safe_dump(nested, sort_keys=True))
        native["config"] = str(path)
    return native


def load_native_config(server_args):
    """Read native YAML for result metadata without importing Optuna."""
    from pathlib import Path

    for key in ("config", "extra_llm_api_options"):
        path = server_args.get(key)
        if isinstance(path, str) and Path(path).is_file():
            try:
                import yaml
            except ImportError:
                # Keep raw CLI serving usable without the notebook extra.
                return None

            config = yaml.safe_load(Path(path).read_text())
            if isinstance(config, dict):
                return config
    return None
