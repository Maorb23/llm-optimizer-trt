"""Reusable full-range expanded search profile; no Optuna execution lives here."""

from __future__ import annotations

import hashlib
import itertools
import json
import shlex

PREFILL_BUDGETS_1 = list(range(1024, 8193, 1024))
TOKEN_BUDGETS_128 = list(range(1024, 32769, 256))
CAPACITIES_128 = list(range(8, 129, 8))
SPACES = {
    1: {
        "vllm": {"max_num_batched_tokens": PREFILL_BUDGETS_1, "cuda_graph_on": [0, 1]},
        "sglang": {"chunked_prefill_size": PREFILL_BUDGETS_1, "cuda_graph_on": [0, 1]},
        "trtllm": {"max_num_tokens": PREFILL_BUDGETS_1, "cuda_graph_on": [0, 1]},
    },
    128: {
        "vllm": {
            "max_num_seqs": CAPACITIES_128,
            "max_num_batched_tokens": TOKEN_BUDGETS_128,
            "gpu_memory_utilization": [0.75, 0.80, 0.85, 0.90],
        },
        "sglang": {
            "max_running_requests": CAPACITIES_128,
            "chunked_prefill_size": TOKEN_BUDGETS_128,
            "mem_fraction_static": [0.70, 0.75, 0.80, 0.85],
        },
        "trtllm": {
            "max_batch_size": CAPACITIES_128,
            "max_num_tokens": TOKEN_BUDGETS_128,
            "kv_cache_free_gpu_memory_fraction": [0.65, 0.75, 0.85, 0.95],
        },
    },
}
CONFIG_PREFIX = "@LLM_OPTIMIZER_CONFIG@/"
EXPANDED_CHOICES = {
    "vllm": {
        "async_scheduling": [False, True],
        "performance_mode": ["balanced", "interactivity", "throughput"],
        "attention_backend": ["auto", "FLASH_ATTN", "FLASHINFER"],
        "kv_cache_dtype": ["auto", "fp8"],
        "enable_prefix_caching": [False, True],
        "graph_mode": ["NONE", "PIECEWISE", "FULL_AND_PIECEWISE"],
    },
    "sglang": {
        "attention_backend": ["triton", "fa3", "flashinfer"],
        "disable_radix_cache": [False, True],
        "schedule_policy": ["fcfs", "lpm", "shortest-prefill-first"],
        "enable_mixed_chunk": [False, True],
        "disable_overlap_schedule": [False, True],
        "cuda_graph_backend_decode": ["disabled", "full", "tc_piecewise"],
        "cuda_graph_backend_prefill": ["disabled", "breakable"],
    },
    "trtllm": {
        "attn_backend": ["TRTLLM"],
        "enable_chunked_prefill": [False, True],
        "enable_block_reuse": [False, True],
        "capacity_scheduler_policy": ["GUARANTEED_NO_EVICT", "MAX_UTILIZATION"],
        "graph_enabled": [False, True],
    },
}


def argument_names(argument):
    return set(argument.partition("=")[0].partition(":")[0].split("*"))


def merge_server_args(defaults, overrides):
    """Explicit fixed or tuned settings replace overlapping profile dimensions."""
    names = set().union(*(argument_names(a) for a in overrides)) if overrides else set()
    return [a for a in defaults if not argument_names(a) & names] + list(overrides)


def expanded_server_args(framework, concurrency, input_tokens=512):
    """Return native argument domains and worker-local YAML/JSON config files.

    As in experiment.Lab.regime_key, any concurrency above one uses the batched
    regime. Synthetic graph knobs are translated, never passed to a framework.
    TRT nested settings use JSON documents (valid YAML), coupled to batch size.
    """
    if (
        isinstance(concurrency, bool)
        or not isinstance(concurrency, int)
        or concurrency < 1
    ):
        raise ValueError("Concurrency must be a positive integer")
    if framework not in EXPANDED_CHOICES:
        raise ValueError(f"Expanded search is unavailable for {framework}")
    regime = 1 if concurrency == 1 else 128
    space = SPACES[regime][framework]
    choices = EXPANDED_CHOICES[framework]
    args, files = [], {}

    def add(name, values):
        args.append(name + "=" + repr(values))

    for name, values in space.items():
        if name != "cuda_graph_on" and not (
            framework == "trtllm"
            and name in {"max_batch_size", "kv_cache_free_gpu_memory_fraction"}
        ):
            add(name, values)
    memory_name = {
        "vllm": "gpu_memory_utilization",
        "sglang": "mem_fraction_static",
        "trtllm": "kv_cache_free_gpu_memory_fraction",
    }[framework]
    if regime == 1 and framework != "trtllm":
        add(memory_name, SPACES[128][framework][memory_name])
    if framework == "vllm":
        if regime == 1:
            add("max_num_seqs", [1])
        add("enable_chunked_prefill", [True])
        for name, values in choices.items():
            if name in {"async_scheduling", "enable_prefix_caching"}:
                add(name + "*no_" + name + ":(bool,bool)", [[v, not v] for v in values])
            elif name == "graph_mode":
                add(
                    "compilation_config",
                    [
                        shlex.quote(json.dumps({"cudagraph_mode": mode}))
                        for mode in values
                    ],
                )
            else:
                add(name, values)
    elif framework == "sglang":
        from llm_optimizer.experiment.sglang_space import (
            resolve_cache_policy,
            sampling_choices,
        )

        if regime == 1:
            add("max_running_requests", [1])
        for name, values in choices.items():
            if name not in {"disable_radix_cache", "schedule_policy"}:
                add(name, values)
        policies = [
            resolve_cache_policy(p) for p in sampling_choices(choices)["cache_policy"]
        ]
        add(
            "disable_radix_cache*schedule_policy:(bool,str)",
            [[p["disable_radix_cache"], p["schedule_policy"]] for p in policies],
        )
    else:
        capacities = [1] if regime == 1 else space["max_batch_size"]
        pairs = []
        for (
            capacity,
            memory,
            chunked,
            reuse,
            policy,
            graphs,
            backend,
        ) in itertools.product(
            capacities,
            SPACES[128][framework][memory_name],
            choices["enable_chunked_prefill"],
            choices["enable_block_reuse"],
            choices["capacity_scheduler_policy"],
            choices["graph_enabled"],
            choices["attn_backend"],
        ):
            config = {
                "attn_backend": backend,
                "enable_chunked_prefill": chunked,
                "kv_cache_config": {
                    "free_gpu_memory_fraction": memory,
                    "enable_block_reuse": reuse,
                },
                "scheduler_config": {"capacity_scheduler_policy": policy},
                "cuda_graph_config": {
                    "batch_sizes": [
                        n for n in [1, 2, 4, 8, 16, 24, 32, 64, 128] if n <= capacity
                    ],
                    "enable_padding": True,
                }
                if graphs
                else None,
            }
            text = json.dumps(config, sort_keys=True)
            name = hashlib.sha256(text.encode()).hexdigest()[:16] + ".yaml"
            files[name] = text
            pairs.append([capacity, CONFIG_PREFIX + name])
        add("max_batch_size*config:(int,str)", pairs)
        # A disabled chunking trial must accommodate a full input prompt.
        args = [a for a in args if not a.startswith("max_num_tokens=")]
        budgets = [n for n in space["max_num_tokens"] if n >= input_tokens]
        if not budgets:
            raise ValueError("Input token length exceeds the expanded TRT token budget")
        add("max_num_tokens", budgets)
    return args, files
