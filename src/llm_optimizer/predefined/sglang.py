# =========================================================================
#  SGLang Predefined Server Argument Configurations
# =========================================================================
# This dictionary is generated based on the sglang/srt/server_args.py source.
# The keys are the normalized argument names.

from llm_optimizer.args import ArgConfig, ConfigsDict

# Parameter mapping from common names to framework-specific parameter names
PARAMETER_MAPPING = {
    "tensor_parallel": "tp_size",
    "data_parallel": "dp_size",
    "max_concurrent_requests": "max_running_requests",
    "prefill_chunk_size": "chunked_prefill_size",
    "batch_size": None,  # Not directly used
    "schedule_conservativeness": "schedule_conservativeness",  # SGLang-specific
    "schedule_policy": "schedule_policy",  # SGLang-specific
}


SERVER_CONFIGS: ConfigsDict = {
    "cuda_graph_backend_decode": ArgConfig[str](name="cuda_graph_backend_decode"),
    "cuda_graph_backend_prefill": ArgConfig[str](name="cuda_graph_backend_prefill"),
    # --- Model and tokenizer ---
    "model_path": ArgConfig[str](name="model_path"),
    "tokenizer_path": ArgConfig[str](name="tokenizer_path"),
    "tokenizer_mode": ArgConfig[str](name="tokenizer_mode"),
    "skip_tokenizer_init": ArgConfig[bool](name="skip_tokenizer_init"),
    "skip_server_warmup": ArgConfig[bool](name="skip_server_warmup"),
    "load_format": ArgConfig[str](name="load_format"),
    "model_loader_extra_config": ArgConfig[str](name="model_loader_extra_config"),
    "trust_remote_code": ArgConfig[bool](name="trust_remote_code"),
    "dtype": ArgConfig[str](name="dtype"),
    "kv_cache_dtype": ArgConfig[str](name="kv_cache_dtype"),
    "quantization": ArgConfig[str](name="quantization"),
    "quantization_param_path": ArgConfig[str](name="quantization_param_path"),
    "context_length": ArgConfig[int](name="context_length"),
    "device": ArgConfig[str](name="device"),
    "served_model_name": ArgConfig[str](name="served_model_name"),
    "chat_template": ArgConfig[str](name="chat_template"),
    "completion_template": ArgConfig[str](name="completion_template"),
    "is_embedding": ArgConfig[bool](name="is_embedding"),
    "enable_multimodal": ArgConfig[bool](name="enable_multimodal"),
    "revision": ArgConfig[str](name="revision"),
    "hybrid_kvcache_ratio": ArgConfig[float](name="hybrid_kvcache_ratio"),
    "impl": ArgConfig[str](name="impl"),
    # --- HTTP server ---
    "host": ArgConfig[str](name="host"),
    "port": ArgConfig[int](name="port"),
    "nccl_port": ArgConfig[int](name="nccl_port"),
    # --- Memory and scheduling ---
    "mem_fraction_static": ArgConfig[float](name="mem_fraction_static"),
    "max_running_requests": ArgConfig[int](name="max_running_requests"),
    "max_total_tokens": ArgConfig[int](name="max_total_tokens"),
    "chunked_prefill_size": ArgConfig[int](name="chunked_prefill_size"),
    "max_prefill_tokens": ArgConfig[int](name="max_prefill_tokens"),
    "schedule_policy": ArgConfig[str](name="schedule_policy"),
    "schedule_conservativeness": ArgConfig[float](name="schedule_conservativeness"),
    "cpu_offload_gb": ArgConfig[int](name="cpu_offload_gb"),
    "page_size": ArgConfig[int](name="page_size"),
    # --- Runtime options ---
    "tp_size": ArgConfig[int](name="tp_size"),
    "tensor_parallel_size": ArgConfig[int](name="tensor_parallel_size"),  # Alias
    "pp_size": ArgConfig[int](name="pp_size"),
    "pipeline_parallel_size": ArgConfig[int](name="pipeline_parallel_size"),  # Alias
    "max_micro_batch_size": ArgConfig[int](name="max_micro_batch_size"),
    "stream_interval": ArgConfig[int](name="stream_interval"),
    "stream_output": ArgConfig[bool](name="stream_output"),
    "random_seed": ArgConfig[int](name="random_seed"),
    "constrained_json_whitespace_pattern": ArgConfig[str](
        name="constrained_json_whitespace_pattern"
    ),
    "watchdog_timeout": ArgConfig[float](name="watchdog_timeout"),
    "dist_timeout": ArgConfig[int](name="dist_timeout"),
    "download_dir": ArgConfig[str](name="download_dir"),
    "base_gpu_id": ArgConfig[int](name="base_gpu_id"),
    "gpu_id_step": ArgConfig[int](name="gpu_id_step"),
    "sleep_on_idle": ArgConfig[bool](name="sleep_on_idle"),
    # --- Logging ---
    "log_level": ArgConfig[str](name="log_level"),
    "log_level_http": ArgConfig[str](name="log_level_http"),
    "log_requests": ArgConfig[bool](name="log_requests"),
    "log_requests_level": ArgConfig[int](name="log_requests_level"),
    "crash_dump_folder": ArgConfig[str](name="crash_dump_folder"),
    "show_time_cost": ArgConfig[bool](name="show_time_cost"),
    "enable_metrics": ArgConfig[bool](name="enable_metrics"),
    "bucket_time_to_first_token": ArgConfig[str](name="bucket_time_to_first_token"),
    "bucket_e2e_request_latency": ArgConfig[str](name="bucket_e2e_request_latency"),
    "bucket_inter_token_latency": ArgConfig[str](name="bucket_inter_token_latency"),
    "collect_tokens_histogram": ArgConfig[bool](name="collect_tokens_histogram"),
    "decode_log_interval": ArgConfig[int](name="decode_log_interval"),
    "enable_request_time_stats_logging": ArgConfig[bool](
        name="enable_request_time_stats_logging"
    ),
    "kv_events_config": ArgConfig[str](name="kv_events_config"),
    # --- API related ---
    "api_key": ArgConfig[str](name="api_key"),
    "file_storage_path": ArgConfig[str](name="file_storage_path"),
    "enable_cache_report": ArgConfig[bool](name="enable_cache_report"),
    "reasoning_parser": ArgConfig[str](name="reasoning_parser"),
    "tool_call_parser": ArgConfig[str](name="tool_call_parser"),
    # --- Data parallelism ---
    "dp_size": ArgConfig[int](name="dp_size"),
    "data_parallel_size": ArgConfig[int](name="data_parallel_size"),  # Alias
    "load_balance_method": ArgConfig[str](name="load_balance_method"),
    # --- Multi-node distributed serving ---
    "dist_init_addr": ArgConfig[str](name="dist_init_addr"),
    "nnodes": ArgConfig[int](name="nnodes"),
    "node_rank": ArgConfig[int](name="node_rank"),
    # --- Model override args in JSON ---
    "json_model_override_args": ArgConfig[str](name="json_model_override_args"),
    "preferred_sampling_params": ArgConfig[str](name="preferred_sampling_params"),
    # --- LoRA ---
    "lora_paths": ArgConfig[str](name="lora_paths"),
    "max_loras_per_batch": ArgConfig[int](name="max_loras_per_batch"),
    "lora_backend": ArgConfig[str](name="lora_backend"),
    # --- Kernel backend ---
    "attention_backend": ArgConfig[str](name="attention_backend"),
    "sampling_backend": ArgConfig[str](name="sampling_backend"),
    "grammar_backend": ArgConfig[str](name="grammar_backend"),
    "mm_attention_backend": ArgConfig[str](name="mm_attention_backend"),
    # --- Speculative decoding ---
    "speculative_algorithm": ArgConfig[str](name="speculative_algorithm"),
    "speculative_draft_model_path": ArgConfig[str](name="speculative_draft_model_path"),
    "speculative_num_steps": ArgConfig[int](name="speculative_num_steps"),
    "speculative_eagle_topk": ArgConfig[int](name="speculative_eagle_topk"),
    "speculative_num_draft_tokens": ArgConfig[int](name="speculative_num_draft_tokens"),
    "speculative_accept_threshold_single": ArgConfig[float](
        name="speculative_accept_threshold_single"
    ),
    "speculative_accept_threshold_acc": ArgConfig[float](
        name="speculative_accept_threshold_acc"
    ),
    "speculative_token_map": ArgConfig[str](name="speculative_token_map"),
    # --- Expert parallelism ---
    "ep_size": ArgConfig[int](name="ep_size"),
    "expert_parallel_size": ArgConfig[int](name="expert_parallel_size"),  # Alias
    "enable_ep_moe": ArgConfig[bool](name="enable_ep_moe"),
    "enable_deepep_moe": ArgConfig[bool](name="enable_deepep_moe"),
    "enable_flashinfer_moe": ArgConfig[bool](name="enable_flashinfer_moe"),
    "enable_flashinfer_allreduce_fusion": ArgConfig[bool](
        name="enable_flashinfer_allreduce_fusion"
    ),
    "deepep_mode": ArgConfig[str](name="deepep_mode"),
    "ep_num_redundant_experts": ArgConfig[int](name="ep_num_redundant_experts"),
    "ep_dispatch_algorithm": ArgConfig[str](name="ep_dispatch_algorithm"),
    "init_expert_location": ArgConfig[str](name="init_expert_location"),
    "enable_eplb": ArgConfig[bool](name="enable_eplb"),
    "eplb_algorithm": ArgConfig[str](name="eplb_algorithm"),
    "eplb_rebalance_num_iterations": ArgConfig[int](
        name="eplb_rebalance_num_iterations"
    ),
    "eplb_rebalance_layers_per_chunk": ArgConfig[int](
        name="eplb_rebalance_layers_per_chunk"
    ),
    "expert_distribution_recorder_mode": ArgConfig[str](
        name="expert_distribution_recorder_mode"
    ),
    "expert_distribution_recorder_buffer_size": ArgConfig[int](
        name="expert_distribution_recorder_buffer_size"
    ),
    "enable_expert_distribution_metrics": ArgConfig[bool](
        name="enable_expert_distribution_metrics"
    ),
    "deepep_config": ArgConfig[str](name="deepep_config"),
    "moe_dense_tp_size": ArgConfig[int](name="moe_dense_tp_size"),
    # --- Double Sparsity ---
    "enable_double_sparsity": ArgConfig[bool](name="enable_double_sparsity"),
    "ds_channel_config_path": ArgConfig[str](name="ds_channel_config_path"),
    "ds_heavy_channel_num": ArgConfig[int](name="ds_heavy_channel_num"),
    "ds_heavy_token_num": ArgConfig[int](name="ds_heavy_token_num"),
    "ds_heavy_channel_type": ArgConfig[str](name="ds_heavy_channel_type"),
    "ds_sparse_decode_threshold": ArgConfig[int](name="ds_sparse_decode_threshold"),
    # --- Optimization/debug options ---
    "disable_radix_cache": ArgConfig[bool](name="disable_radix_cache"),
    "cuda_graph_max_bs": ArgConfig[int](name="cuda_graph_max_bs"),
    "cuda_graph_bs": ArgConfig[str](name="cuda_graph_bs"),
    "disable_cuda_graph": ArgConfig[bool](name="disable_cuda_graph"),
    "disable_cuda_graph_padding": ArgConfig[bool](name="disable_cuda_graph_padding"),
    "enable_profile_cuda_graph": ArgConfig[bool](name="enable_profile_cuda_graph"),
    "enable_nccl_nvls": ArgConfig[bool](name="enable_nccl_nvls"),
    "enable_tokenizer_batch_encode": ArgConfig[bool](
        name="enable_tokenizer_batch_encode"
    ),
    "disable_outlines_disk_cache": ArgConfig[bool](name="disable_outlines_disk_cache"),
    "disable_custom_all_reduce": ArgConfig[bool](name="disable_custom_all_reduce"),
    "enable_mscclpp": ArgConfig[bool](name="enable_mscclpp"),
    "disable_overlap_schedule": ArgConfig[bool](name="disable_overlap_schedule"),
    "disable_overlap_cg_plan": ArgConfig[bool](name="disable_overlap_cg_plan"),
    "enable_mixed_chunk": ArgConfig[bool](name="enable_mixed_chunk"),
    "enable_dp_attention": ArgConfig[bool](name="enable_dp_attention"),
    "enable_dp_lm_head": ArgConfig[bool](name="enable_dp_lm_head"),
    "enable_two_batch_overlap": ArgConfig[bool](name="enable_two_batch_overlap"),
    "enable_torch_compile": ArgConfig[bool](name="enable_torch_compile"),
    "torch_compile_max_bs": ArgConfig[int](name="torch_compile_max_bs"),
    "torchao_config": ArgConfig[str](name="torchao_config"),
    "enable_nan_detection": ArgConfig[bool](name="enable_nan_detection"),
    "enable_p2p_check": ArgConfig[bool](name="enable_p2p_check"),
    "triton_attention_reduce_in_fp32": ArgConfig[bool](
        name="triton_attention_reduce_in_fp32"
    ),
    "triton_attention_num_kv_splits": ArgConfig[int](
        name="triton_attention_num_kv_splits"
    ),
    "num_continuous_decode_steps": ArgConfig[int](name="num_continuous_decode_steps"),
    "delete_ckpt_after_loading": ArgConfig[bool](name="delete_ckpt_after_loading"),
    "enable_memory_saver": ArgConfig[bool](name="enable_memory_saver"),
    "allow_auto_truncate": ArgConfig[bool](name="allow_auto_truncate"),
    "enable_custom_logit_processor": ArgConfig[bool](
        name="enable_custom_logit_processor"
    ),
    "enable_hierarchical_cache": ArgConfig[bool](name="enable_hierarchical_cache"),
    "hicache_ratio": ArgConfig[float](name="hicache_ratio"),
    "hicache_size": ArgConfig[int](name="hicache_size"),
    "hicache_write_policy": ArgConfig[str](name="hicache_write_policy"),
    "hicache_io_backend": ArgConfig[str](name="hicache_io_backend"),
    "flashinfer_mla_disable_ragged": ArgConfig[bool](
        name="flashinfer_mla_disable_ragged"
    ),
    "disable_shared_experts_fusion": ArgConfig[bool](
        name="disable_shared_experts_fusion"
    ),
    "disable_chunked_prefix_cache": ArgConfig[bool](
        name="disable_chunked_prefix_cache"
    ),
    "disable_fast_image_processor": ArgConfig[bool](
        name="disable_fast_image_processor"
    ),
    "enable_return_hidden_states": ArgConfig[bool](name="enable_return_hidden_states"),
    "enable_triton_kernel_moe": ArgConfig[bool](name="enable_triton_kernel_moe"),
    "warmups": ArgConfig[str](name="warmups"),
    # --- Debug tensor dumps ---
    "debug_tensor_dump_output_folder": ArgConfig[str](
        name="debug_tensor_dump_output_folder"
    ),
    "debug_tensor_dump_input_file": ArgConfig[str](name="debug_tensor_dump_input_file"),
    "debug_tensor_dump_inject": ArgConfig[bool](name="debug_tensor_dump_inject"),
    "debug_tensor_dump_prefill_only": ArgConfig[bool](
        name="debug_tensor_dump_prefill_only"
    ),
    # --- PD disaggregation ---
    "disaggregation_mode": ArgConfig[str](name="disaggregation_mode"),
    "disaggregation_transfer_backend": ArgConfig[str](
        name="disaggregation_transfer_backend"
    ),
    "disaggregation_bootstrap_port": ArgConfig[int](
        name="disaggregation_bootstrap_port"
    ),
    "disaggregation_decode_tp": ArgConfig[int](name="disaggregation_decode_tp"),
    "disaggregation_decode_dp": ArgConfig[int](name="disaggregation_decode_dp"),
    "disaggregation_prefill_pp": ArgConfig[int](name="disaggregation_prefill_pp"),
    "disaggregation_ib_device": ArgConfig[str](name="disaggregation_ib_device"),
    "num_reserved_decode_tokens": ArgConfig[int](name="num_reserved_decode_tokens"),
    "pdlb_url": ArgConfig[str](name="pdlb_url"),
    # --- Model weight update ---
    "custom_weight_loader": ArgConfig[str](name="custom_weight_loader"),
    "weight_loader_disable_mmap": ArgConfig[bool](name="weight_loader_disable_mmap"),
}
