"""Editable phase-one configuration and supervised batch entry point."""
# %%
from pathlib import Path
import sys

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / "src" / "llm_optimizer").is_dir())
sys.path.insert(0, str(ROOT / "src"))
from llm_optimizer.experiment.phase1 import Phase1Runner, plot_group

# %%
CAMPAIGN = "phase1-week1"
FRAMEWORKS = ["sglang"]  # or ["vllm", "sglang", "trtllm"]
RUN_REAL_GPU = False
CONCURRENCIES = [1, 8, 32, 128]
SEEDS = [11, 29, 47]
RUN_ROOT = ROOT / "experiment_results" / "phase1"
CAMPAIGN_HOURS = 168
GROUP_TIMEOUT_SECONDS = 6 * 3600  # includes all frameworks/seeds in a group
MAX_RESTARTS = 2  # bounded retries after a crashed/timed-out group
MODEL = "Qwen/Qwen3.5-4B"  # same model across A/B isolates search-space effects
MODEL_SNAPSHOT = "/workspace/models/pinned-model-snapshot"
TRIALS = 40  # attempted trials per framework/concurrency/sampler/seed/pruner
REQUESTS = 896
VALIDATION_REPEATS = 5

# Fill with actual verified values. Never mark checks true without performing them.
PREFLIGHT = {
    framework: {
        "gpu_type": "REPLACE", "framework_version": "REPLACE",
        "cuda_version": "REPLACE", "driver_version": "REPLACE",
        "model_revision": "REPLACE_WITH_MODEL_COMMIT",
        "native_config_verified": False, "quality_verified": False,
        "verified_precision_combinations": [],  # weights/cache pairs approved on this VM
    }
    for framework in ["vllm", "sglang", "trtllm"]
}

# %%
PREFILL_BUDGETS = list(range(1024, 8193, 1024))
TOKEN_BUDGETS = list(range(1024, 32769, 256))
CAPACITIES = list(range(8, 129, 8))
BASE_SPACES = {
    1: {
        "vllm": {"max_num_batched_tokens": PREFILL_BUDGETS, "cuda_graph_on": [0, 1]},
        "sglang": {"chunked_prefill_size": PREFILL_BUDGETS, "cuda_graph_on": [0, 1]},
        "trtllm": {"max_num_tokens": PREFILL_BUDGETS, "cuda_graph_on": [0, 1]},
    },
    128: {  # shared regime for all concurrency levels > 1
        "vllm": {
            "max_num_seqs": CAPACITIES, "max_num_batched_tokens": TOKEN_BUDGETS,
            "gpu_memory_utilization": [0.75, 0.80, 0.85, 0.90],
        },
        "sglang": {
            "max_running_requests": CAPACITIES, "chunked_prefill_size": TOKEN_BUDGETS,
            "mem_fraction_static": [0.70, 0.75, 0.80, 0.85],
        },
        "trtllm": {
            "max_batch_size": CAPACITIES, "max_num_tokens": TOKEN_BUDGETS,
            "kv_cache_free_gpu_memory_fraction": [0.65, 0.75, 0.85, 0.95],
        },
    },
}

EXPANDED_CHOICES = {'vllm': {'async_scheduling': [False, True],
          'performance_mode': ['balanced', 'interactivity', 'throughput'],
          'attention_backend': ['auto', 'FLASH_ATTN', 'FLASHINFER'],
          'kv_cache_dtype': ['auto'],  # overridden by PRECISION_CHOICES below
          'enable_prefix_caching': [False, True],
          'graph_mode': ['NONE', 'PIECEWISE', 'FULL_AND_PIECEWISE']},
 'sglang': {'attention_backend': ['triton', 'fa3', 'flashinfer'],
            'disable_radix_cache': [False, True],
            'schedule_policy': ['fcfs', 'lpm'],
            'enable_mixed_chunk': [False, True],
            'disable_overlap_schedule': [False, True],
            'cuda_graph_backend_decode': ['disabled', 'full'],
            'cuda_graph_backend_prefill': ['disabled', 'breakable']},
 'trtllm': {'attn_backend': ['TRTLLM'],
            'enable_chunked_prefill': [False, True],
            'enable_block_reuse': [False, True],
            'capacity_scheduler_policy': ['GUARANTEED_NO_EVICT', 'MAX_UTILIZATION'],
            'graph_enabled': [False, True]}}

EXTRA_SPACES = {
    "vllm": {"graph_capture_profile": ["powers", "dense"]},
    "sglang": {
        "schedule_conservativeness": [0.5, 0.75, 1.0, 1.25, 1.5],
        "max_prefill_tokens": [8192, 16384, 24576, 32768],
    },
    "trtllm": {
        "graph_capture_profile": ["powers", "dense"],
        "graph_padding": [0, 1],
    },
}

# %%
# Independent Phase B axes. To force FP8 weights + FP8 cache, use ["fp8"] for both.
# Phase A always uses unquantized BF16 weights and the default cache (BF16).
PRECISION_CHOICES = {
    "vllm": {"weight_precision": ["bf16", "fp8"], "kv_cache_precision": ["bf16", "fp8"]},
    "sglang": {"weight_precision": ["bf16", "fp8"], "kv_cache_precision": ["bf16", "fp8"]},
    "trtllm": {"weight_precision": ["bf16", "fp8"], "kv_cache_precision": ["bf16", "fp8"]},
}
# TRT on-load weight quantization needs calibration. Adjust for your pinned version;
# prepare/cache its calibration dataset before launching the unattended campaign.
TRT_FP8_CALIBRATION = {
    "calib_batches": 32, "calib_batch_size": 1,
    "calib_max_seq_length": 512, "tokenizer_max_seq_length": 512,
}

# %%
def make_runner():
    settings = {
        "wandb_group": CAMPAIGN,
        "frameworks": FRAMEWORKS, "real_frameworks": FRAMEWORKS,
        "run_real_gpu": RUN_REAL_GPU, "quick_demo": False,
        "phases": ["A", "B"], "concurrencies": CONCURRENCIES,
        "seeds": SEEDS, "sampler_names": ["tpe"], "pareto_samplers": ["tpe"],
        "phase_models": {"A": MODEL, "B": MODEL},
        "model_snapshots": {"A": MODEL_SNAPSHOT, "B": MODEL_SNAPSHOT},
        "phase_requests": {"A": REQUESTS, "B": REQUESTS},
        "search_trials": {p: {"1": TRIALS, "128": TRIALS} for p in ["A", "B"]},
        "base_workload": {
            "dataset_name": "random", "random_input_len": 512,
            "random_output_len": 128, "random_range_ratio": 1.0,
            "seed": 2026, "request_rate": "inf",
        },
        "validation_repeats": VALIDATION_REPEATS,
        "validation_requests": {"A": 1000, "B": 1000},
        "gpu_count": 1, "rest_seconds": 10,
        "real_preflight": {
            phase: {f: {**PREFLIGHT[f], "expanded_choices": EXPANDED_CHOICES[f]}
                    for f in FRAMEWORKS}
            for phase in ["A", "B"]
        },
    }
    return Phase1Runner(
        settings, BASE_SPACES, EXPANDED_CHOICES, EXTRA_SPACES, RUN_ROOT,
        precision_choices=PRECISION_CHOICES, trt_fp8_calibration=TRT_FP8_CALIBRATION,
    )

# Lazy construction keeps script imports free of run artifacts.
_runner = None

def get_runner():
    global _runner
    if _runner is None:
        _runner = make_runner()
    return _runner

def run_objective_group(phase, target, concurrencies, pruner_name="nop"):
    return get_runner().run_objective_group(phase, target, concurrencies, pruner_name)

# %% Batch supervisor: notebooks call the same runner directly.
def main():
    global TRIALS, REQUESTS, SEEDS, CONCURRENCIES, RUN_REAL_GPU, CAMPAIGN
    import argparse
    import os
    import signal
    import psutil
    import subprocess
    import time
    from llm_optimizer.experiment.workflow import write_json

    tasks = [(p, t, cs, "nop") for p in ["A", "B"]
             for t, cs in [("e2e", [1]), ("ttft", [1]),
                           ("throughput", [c for c in CONCURRENCIES if c > 1]),
                           ("e2e", [c for c in CONCURRENCIES if c > 1])]]
    tasks.append(("B", "throughput", [c for c in CONCURRENCIES if c > 1], "median"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, choices=range(len(tasks)))
    parser.add_argument("--smoke", action="store_true", help="two CPU trials, one task")
    args = parser.parse_args()
    if args.smoke:
        TRIALS, REQUESTS, SEEDS, CONCURRENCIES = 2, 32, [11], [1, 8]
        RUN_REAL_GPU, CAMPAIGN = False, "phase1-smoke"
        tasks = [("B", "throughput", [8], "nop")]
    if args.task is not None:
        phase, target, cs, pruner = tasks[args.task]
        run_objective_group(phase, target, cs, pruner_name=pruner)
        return 0
    runner = get_runner()
    status_path = runner.run_dir / "queue.json"
    # Hold a separate supervisor lock: child workers need the GPU lock themselves.
    import fcntl
    with (runner.run_dir / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        import json
        status = (json.loads(status_path.read_text()) if status_path.exists() else
                  {"deadline": time.time() + CAMPAIGN_HOURS * 3600,
                   "completed": [], "attempts": {}, "failed": []})
        write_json(status_path, status)
        for index, _ in enumerate(tasks):
            if index in status["completed"]:
                continue
            key = str(index)
            while status["attempts"].get(key, 0) <= MAX_RESTARTS:
                remaining = status["deadline"] - time.time()
                if remaining <= 0:
                    print("Campaign deadline reached", flush=True)
                    return 0  # never let service restarts extend the allocation
                status["attempts"][key] = status["attempts"].get(key, 0) + 1
                write_json(status_path, status)
                command = [sys.executable, str(Path(__file__).resolve()), "--task", key]
                if args.smoke:
                    command.append("--smoke")
                process = subprocess.Popen(command, start_new_session=True)
                try:
                    result = process.wait(timeout=min(GROUP_TIMEOUT_SECONDS, remaining))
                except subprocess.TimeoutExpired:
                    result = 124
                finally:
                    # TRT engines can create their own session. Capture descendants
                    # before killing the worker so those servers are cleaned up too.
                    try:
                        descendants = psutil.Process(process.pid).children(recursive=True)
                    except psutil.NoSuchProcess:
                        descendants = []
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    for child in descendants:
                        try:
                            child.kill()
                        except psutil.NoSuchProcess:
                            pass
                    psutil.wait_procs(descendants, timeout=10)
                    process.wait()
                if result == 0:
                    status["completed"].append(index)
                    break
                print(f"Group {index} exited {result}; attempt "
                      f"{status['attempts'][key]}/{MAX_RESTARTS + 1}", flush=True)
            else:
                if index not in status["failed"]:
                    status["failed"].append(index)
            write_json(status_path, status)
        return 1 if status["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
