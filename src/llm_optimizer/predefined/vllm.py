# =========================================================================
#  vLLM Engine Argument Configurations
# =========================================================================
# This dictionary is generated based on vllm/engine/arg_utils.py source.

from llm_optimizer.args import ArgConfig, ConfigsDict

# Parameter mapping from common names to framework-specific parameter names
PARAMETER_MAPPING = {
    "tensor_parallel": "tensor_parallel_size",
    "data_parallel": "data_parallel_size",
    "max_concurrent_requests": "max_num_seqs",
    "prefill_chunk_size": None,  # Not used
    "batch_size": "max_num_batched_tokens",
}


SERVER_CONFIGS: ConfigsDict = {
    # --- ModelConfig Arguments ---
    "model": ArgConfig[str](name="model"),
    "tokenizer": ArgConfig[str](name="tokenizer"),
    "hf_config_path": ArgConfig[str](name="hf_config_path"),
    "runner": ArgConfig[str](name="runner"),
    "convert": ArgConfig[str](name="convert"),
    "task": ArgConfig[str](name="task"),
    "tokenizer_mode": ArgConfig[str](name="tokenizer_mode"),
    "trust_remote_code": ArgConfig[bool](name="trust_remote_code"),
    "dtype": ArgConfig[str](name="dtype"),
    "seed": ArgConfig[int](name="seed"),
    "revision": ArgConfig[str](name="revision"),
    "code_revision": ArgConfig[str](name="code_revision"),
    "rope_scaling": ArgConfig[str](name="rope_scaling"),
    "rope_theta": ArgConfig[float](name="rope_theta"),
    "tokenizer_revision": ArgConfig[str](name="tokenizer_revision"),
    "max_model_len": ArgConfig[int](name="max_model_len"),
    "quantization": ArgConfig[str](name="quantization"),
    "enforce_eager": ArgConfig[bool](name="enforce_eager"),
    "max_seq_len_to_capture": ArgConfig[int](name="max_seq_len_to_capture"),
    "max_logprobs": ArgConfig[int](name="max_logprobs"),
    "logprobs_mode": ArgConfig[str](name="logprobs_mode"),
    "disable_sliding_window": ArgConfig[bool](name="disable_sliding_window"),
    "disable_cascade_attn": ArgConfig[bool](name="disable_cascade_attn"),
    "skip_tokenizer_init": ArgConfig[bool](name="skip_tokenizer_init"),
    "enable_prompt_embeds": ArgConfig[bool](name="enable_prompt_embeds"),
    "served_model_name": ArgConfig[str](name="served_model_name"),
    "allowed_local_media_path": ArgConfig[str](name="allowed_local_media_path"),
    "config_format": ArgConfig[str](name="config_format"),
    "hf_token": ArgConfig[str](name="hf_token"),
    "hf_overrides": ArgConfig[str](name="hf_overrides"),
    "override_neuron_config": ArgConfig[str](name="override_neuron_config"),
    "override_pooler_config": ArgConfig[str](name="override_pooler_config"),
    "logits_processor_pattern": ArgConfig[str](name="logits_processor_pattern"),
    "generation_config": ArgConfig[str](name="generation_config"),
    "override_generation_config": ArgConfig[str](name="override_generation_config"),
    "enable_sleep_mode": ArgConfig[bool](name="enable_sleep_mode"),
    "model_impl": ArgConfig[str](name="model_impl"),
    "override_attention_dtype": ArgConfig[str](name="override_attention_dtype"),
    # --- CacheConfig Arguments ---
    "block_size": ArgConfig[int](name="block_size"),
    "gpu_memory_utilization": ArgConfig[float](name="gpu_memory_utilization"),
    "swap_space": ArgConfig[float](name="swap_space"),
    "kv_cache_dtype": ArgConfig[str](name="kv_cache_dtype"),
    "num_gpu_blocks_override": ArgConfig[int](name="num_gpu_blocks_override"),
    "enable_prefix_caching": ArgConfig[bool](name="enable_prefix_caching"),
    "prefix_caching_hash_algo": ArgConfig[str](name="prefix_caching_hash_algo"),
    "cpu_offload_gb": ArgConfig[float](name="cpu_offload_gb"),
    "calculate_kv_scales": ArgConfig[bool](name="calculate_kv_scales"),
    # --- LoadConfig Arguments ---
    "load_format": ArgConfig[str](name="load_format"),
    "download_dir": ArgConfig[str](name="download_dir"),
    "model_loader_extra_config": ArgConfig[str](name="model_loader_extra_config"),
    "ignore_patterns": ArgConfig[str](name="ignore_patterns"),
    "use_tqdm_on_load": ArgConfig[bool](name="use_tqdm_on_load"),
    "pt_load_map_location": ArgConfig[str](name="pt_load_map_location"),
    # --- ParallelConfig Arguments ---
    "pipeline_parallel_size": ArgConfig[int](name="pipeline_parallel_size"),
    "tensor_parallel_size": ArgConfig[int](name="tensor_parallel_size"),
    "data_parallel_size": ArgConfig[int](name="data_parallel_size"),
    "data_parallel_rank": ArgConfig[int](name="data_parallel_rank"),
    "data_parallel_start_rank": ArgConfig[int](name="data_parallel_start_rank"),
    "data_parallel_size_local": ArgConfig[int](name="data_parallel_size_local"),
    "data_parallel_address": ArgConfig[str](name="data_parallel_address"),
    "data_parallel_rpc_port": ArgConfig[int](name="data_parallel_rpc_port"),
    "data_parallel_hybrid_lb": ArgConfig[bool](name="data_parallel_hybrid_lb"),
    "data_parallel_backend": ArgConfig[str](name="data_parallel_backend"),
    "enable_expert_parallel": ArgConfig[bool](name="enable_expert_parallel"),
    "enable_eplb": ArgConfig[bool](name="enable_eplb"),
    "num_redundant_experts": ArgConfig[int](name="num_redundant_experts"),
    "eplb_window_size": ArgConfig[int](name="eplb_window_size"),
    "eplb_step_interval": ArgConfig[int](name="eplb_step_interval"),
    "eplb_log_balancedness": ArgConfig[bool](name="eplb_log_balancedness"),
    "max_parallel_loading_workers": ArgConfig[int](name="max_parallel_loading_workers"),
    "disable_custom_all_reduce": ArgConfig[bool](name="disable_custom_all_reduce"),
    "ray_workers_use_nsight": ArgConfig[bool](name="ray_workers_use_nsight"),
    "distributed_executor_backend": ArgConfig[str](name="distributed_executor_backend"),
    "worker_cls": ArgConfig[str](name="worker_cls"),
    "worker_extension_cls": ArgConfig[str](name="worker_extension_cls"),
    "enable_multimodal_encoder_data_parallel": ArgConfig[bool](
        name="enable_multimodal_encoder_data_parallel"
    ),
    # --- SchedulerConfig Arguments ---
    "max_num_batched_tokens": ArgConfig[int](name="max_num_batched_tokens"),
    "max_num_seqs": ArgConfig[int](name="max_num_seqs"),
    "max_num_partial_prefills": ArgConfig[int](name="max_num_partial_prefills"),
    "max_long_partial_prefills": ArgConfig[int](name="max_long_partial_prefills"),
    "long_prefill_token_threshold": ArgConfig[int](name="long_prefill_token_threshold"),
    "num_lookahead_slots": ArgConfig[int](name="num_lookahead_slots"),
    "cuda_graph_sizes": ArgConfig[str](name="cuda_graph_sizes"),
    "scheduler_delay_factor": ArgConfig[float](name="scheduler_delay_factor"),
    "enable_chunked_prefill": ArgConfig[bool](name="enable_chunked_prefill"),
    "disable_chunked_mm_input": ArgConfig[bool](name="disable_chunked_mm_input"),
    "preemption_mode": ArgConfig[str](name="preemption_mode"),
    "num_scheduler_steps": ArgConfig[int](name="num_scheduler_steps"),
    "multi_step_stream_outputs": ArgConfig[bool](name="multi_step_stream_outputs"),
    "scheduling_policy": ArgConfig[str](name="scheduling_policy"),
    "scheduler_cls": ArgConfig[str](name="scheduler_cls"),
    "disable_hybrid_kv_cache_manager": ArgConfig[bool](
        name="disable_hybrid_kv_cache_manager"
    ),
    "async_scheduling": ArgConfig[bool](name="async_scheduling"),
    # --- LoRAConfig Arguments ---
    "enable_lora": ArgConfig[bool](name="enable_lora"),
    "enable_lora_bias": ArgConfig[bool](name="enable_lora_bias"),
    "max_loras": ArgConfig[int](name="max_loras"),
    "max_lora_rank": ArgConfig[int](name="max_lora_rank"),
    "default_mm_loras": ArgConfig[str](name="default_mm_loras"),
    "fully_sharded_loras": ArgConfig[bool](name="fully_sharded_loras"),
    "max_cpu_loras": ArgConfig[int](name="max_cpu_loras"),
    "lora_dtype": ArgConfig[str](name="lora_dtype"),
    "lora_extra_vocab_size": ArgConfig[int](name="lora_extra_vocab_size"),
    # --- MultiModalConfig Arguments ---
    "limit_mm_per_prompt": ArgConfig[str](name="limit_mm_per_prompt"),
    "interleave_mm_strings": ArgConfig[bool](name="interleave_mm_strings"),
    "media_io_kwargs": ArgConfig[str](name="media_io_kwargs"),
    "mm_processor_kwargs": ArgConfig[str](name="mm_processor_kwargs"),
    "disable_mm_preprocessor_cache": ArgConfig[bool](
        name="disable_mm_preprocessor_cache"
    ),
    # --- SpeculativeConfig Argument ---
    "speculative_config": ArgConfig[str](name="speculative_config"),
    # --- DecodingConfig Arguments ---
    "guided_decoding_backend": ArgConfig[str](name="guided_decoding_backend"),
    "guided_decoding_disable_fallback": ArgConfig[bool](
        name="guided_decoding_disable_fallback"
    ),
    "guided_decoding_disable_any_whitespace": ArgConfig[bool](
        name="guided_decoding_disable_any_whitespace"
    ),
    "guided_decoding_disable_additional_properties": ArgConfig[bool](
        name="guided_decoding_disable_additional_properties"
    ),
    "reasoning_parser": ArgConfig[str](name="reasoning_parser"),
    # --- ObservabilityConfig Arguments ---
    "show_hidden_metrics_for_version": ArgConfig[str](
        name="show_hidden_metrics_for_version"
    ),
    "otlp_traces_endpoint": ArgConfig[str](name="otlp_traces_endpoint"),
    "collect_detailed_traces": ArgConfig[str](name="collect_detailed_traces"),
    # --- VllmConfig Arguments ---
    "kv_transfer_config": ArgConfig[str](name="kv_transfer_config"),
    "kv_events_config": ArgConfig[str](name="kv_events_config"),
    "compilation_config": ArgConfig[str](name="compilation_config"),
    "additional_config": ArgConfig[str](name="additional_config"),
    # --- Other/Top-level Arguments ---
    "disable_log_stats": ArgConfig[bool](name="disable_log_stats"),
    "enable_prompt_adapter": ArgConfig[bool](name="enable_prompt_adapter"),
    # --- AsyncEngineArgs ---
    "disable_log_requests": ArgConfig[bool](name="disable_log_requests"),
}
