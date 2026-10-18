# =========================================================================
#  Benchmark Client Argument Configurations
# =========================================================================
# This dictionary is generated based on the bench_serving.py source.

from llm_optimizer.args import ArgConfig, ConfigsDict

CLIENT_CONFIGS: ConfigsDict = {
    # --- Main benchmark arguments from add_parser_args ---
    "backend": ArgConfig[str](name="backend"),
    "base_url": ArgConfig[str](name="base_url"),
    "dataset_name": ArgConfig[str](name="dataset_name"),
    "dataset_path": ArgConfig[str](name="dataset_path"),
    "tokenizer": ArgConfig[str](name="tokenizer"),
    "num_prompts": ArgConfig[int](name="num_prompts"),
    "sharegpt_output_len": ArgConfig[int](name="sharegpt_output_len"),
    "sharegpt_context_len": ArgConfig[int](name="sharegpt_context_len"),
    "random_input_len": ArgConfig[int](name="random_input_len"),
    "random_output_len": ArgConfig[int](name="random_output_len"),
    "random_range_ratio": ArgConfig[float](name="random_range_ratio"),
    "request_rate": ArgConfig[float](name="request_rate"),
    "max_concurrency": ArgConfig[int](name="max_concurrency"),
    "output_file": ArgConfig[str](name="output_file"),
    "output_details": ArgConfig[bool](name="output_details"),
    "disable_tqdm": ArgConfig[bool](name="disable_tqdm"),
    "disable_stream": ArgConfig[bool](name="disable_stream"),
    "return_logprob": ArgConfig[bool](name="return_logprob"),
    "seed": ArgConfig[int](name="seed"),
    "disable_ignore_eos": ArgConfig[bool](name="disable_ignore_eos"),
    "extra_request_body": ArgConfig[str](name="extra_request_body"),
    "apply_chat_template": ArgConfig[bool](name="apply_chat_template"),
    "profile": ArgConfig[bool](name="profile"),
    "lora_name": ArgConfig[str](name="lora_name"),
    "prompt_suffix": ArgConfig[str](name="prompt_suffix"),
    "pd_separated": ArgConfig[bool](name="pd_separated"),
    "flush_cache": ArgConfig[bool](name="flush_cache"),
    "warmup_requests": ArgConfig[int](name="warmup_requests"),
    "tokenize_prompt": ArgConfig[bool](name="tokenize_prompt"),
    # --- Arguments for the generated-shared-prefix dataset ---
    "gsp_num_groups": ArgConfig[int](name="gsp_num_groups"),
    "gsp_prompts_per_group": ArgConfig[int](name="gsp_prompts_per_group"),
    "gsp_system_prompt_len": ArgConfig[int](name="gsp_system_prompt_len"),
    "gsp_question_len": ArgConfig[int](name="gsp_question_len"),
    "gsp_output_len": ArgConfig[int](name="gsp_output_len"),
    "gsp_enable_system_prompt_partial_randomize": ArgConfig[bool](
        name="gsp_enable_system_prompt_partial_randomize"
    ),
    "gsp_system_prompt_partial_randomize_start_min": ArgConfig[float](
        name="gsp_system_prompt_partial_randomize_start_min"
    ),
    "gsp_system_prompt_partial_randomize_start_max": ArgConfig[float](
        name="gsp_system_prompt_partial_randomize_start_max"
    ),
    # --- Arguments from the main block ---
    "host": ArgConfig[str](name="host"),
    "port": ArgConfig[int](name="port"),
    "model": ArgConfig[str](name="model"),
}
