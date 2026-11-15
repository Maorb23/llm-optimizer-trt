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
CONCURRENCIES = [1, 8, 16, 32, 128]
SEEDS = [11, 29, 47]
RUN_ROOT = ROOT / "experiment_results" / "phase1"
CAMPAIGN_HOURS = 168
GROUP_TIMEOUT_SECONDS = 6 * 3600  # includes all frameworks/seeds in a group
MAX_RESTARTS = 2  # bounded retries after a crashed/timed-out group
MODEL = "Qwen/Qwen3.5-4B"  # same model across A/B isolates search-space effects
MODEL_SNAPSHOT = "/workspace/models/pinned-model-snapshot"
TRIALS = 20  # attempts PER sampler/seed/framework/concurrency/pruner
SAMPLER_NAMES = ["gp", "tpe", "tpe_multivariate"]  # select any nonempty subset
PARETO_SAMPLERS = [*SAMPLER_NAMES, "nsga2"]  # available to optional Pareto calls
REQUESTS_BY_CONCURRENCY = {1: 32, 8: 62, 16: 125, 32: 250, 128: 1000}
# Preserve the v2 400/1000 pruning-observation fraction at each request budget.
PRUNING_MIN_COMPLETED_REQUESTS = {
    c: max(1, int(n * 0.4)) for c, n in REQUESTS_BY_CONCURRENCY.items()
}
SAMPLER_SETTINGS = {
    "startup_single": 4, "startup_multi": 10,
    "n_ei_candidates": 24, "population_size": 8,
}
VALIDATION_REPEATS = 5

# W&B controls match optuna_inference_optimization_lab_v2.
WANDB_ENABLED = True
WANDB_PROJECT = "inference-optimization"
WANDB_ENTITY = "maorblumberg-tel-aviv-university"  # or None for your SDK default
WANDB_MODE = "online"  # "offline" keeps runs local; False above disables the SDK
WANDB_GROUP = CAMPAIGN  # same label across framework VMs

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
# Illustrative SLOs; replace after measuring the pinned model/workload/GPU.
TTFT_LIMITS_MS = {"1": 100.0, "128": 200.0}
E2E_LIMITS_MS = {"1": 2000.0, "128": 6000.0}
FAILURE_RATE_LIMIT = 0.0
CONSTRAINED = False
PRUNER_NAMES = ["nop", "median", "percentile", "halving", "hyperband", "threshold"]
PRUNING_SAMPLES = 32
# Ordinary serving is explicit. Setup is opt-in and may download checkpoints.
HF_SETUP = {
    "target_model_id": MODEL, "target_revision": "REPLACE_WITH_MODEL_COMMIT",
    "framework": "sglang", "enable_dflash": False,
    "speculative_algorithm": None, "draft_model_id": None, "draft_revision": None,
    "block_size": 16, "num_speculative_tokens": 15, "max_draft_len": 3,
}
# SGLang: DFLASH/DSPARK; vLLM: enable_dflash=True selects DSpark.
# TRT: explicit algorithms accepted by predefined.trtllm.validate_draft_request.
# Configure a separate campaign/run root to compare ordinary and speculative serving.
MODEL_PROFILES = {}  # populated only by explicit setup_pinned_models()
MODEL_SETUP_IDENTITY = None
MODEL_SETUP_CHOICES = None
PHASE_B_MODEL = None
PHASE_B_SNAPSHOT = None

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
          'kv_cache_dtype': ['auto'],
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
def make_runner(for_model_setup=False):
    if MODEL_SETUP_IDENTITY is not None and MODEL_SETUP_IDENTITY != HF_SETUP:
        raise ValueError("HF_SETUP changed; clear MODEL_PROFILES and MODEL_SETUP_IDENTITY, then rerun setup")
    if MODEL_SETUP_CHOICES is not None and MODEL_SETUP_CHOICES != EXPANDED_CHOICES[HF_SETUP["framework"]]:
        raise ValueError("Expanded choices changed after HF setup; clear setup profiles and rerun setup")
    settings = {
        "ttft_limits_ms": TTFT_LIMITS_MS, "e2e_limits_ms": E2E_LIMITS_MS,
        "failure_rate_limit": FAILURE_RATE_LIMIT, "pruning_samples": PRUNING_SAMPLES,
        "constrained": CONSTRAINED,
        "wandb_enabled": WANDB_ENABLED,
        "wandb_project": WANDB_PROJECT, "wandb_entity": WANDB_ENTITY,
        "wandb_mode": WANDB_MODE, "wandb_group": WANDB_GROUP,
        "frameworks": FRAMEWORKS, "real_frameworks": FRAMEWORKS,
        "run_real_gpu": RUN_REAL_GPU and not for_model_setup, "quick_demo": False,
        "phases": ["A", "B"], "concurrencies": CONCURRENCIES,
        "seeds": SEEDS, "sampler_names": SAMPLER_NAMES, "pareto_samplers": PARETO_SAMPLERS,
        "phase_models": {"A": MODEL, "B": PHASE_B_MODEL or MODEL},
        "model_snapshots": {"A": MODEL_SNAPSHOT, "B": PHASE_B_SNAPSHOT or MODEL_SNAPSHOT},
        # Lab's phase-level fallback; actual study counts use the map below.
        "phase_requests": {"A": REQUESTS_BY_CONCURRENCY[128],
                           "B": REQUESTS_BY_CONCURRENCY[128]},
        "sampler_settings": SAMPLER_SETTINGS,
        "search_trials": {p: {"1": TRIALS, "128": TRIALS} for p in ["A", "B"]},
        "base_workload": {
            "dataset_name": "random", "apply_chat_template": True, "random_input_len": 512,
            "random_output_len": 128, "random_range_ratio": 1.0,
            "seed": 2026, "request_rate": "inf",
        },
        "validation_repeats": VALIDATION_REPEATS,
        "validation_requests": {"A": 1000, "B": 1000},
        "gpu_count": 1, "rest_seconds": 10,
        "real_preflight": {
            phase: {f: {**PREFLIGHT[f], "expanded_choices": EXPANDED_CHOICES[f],
                           **(MODEL_PROFILES.get(f, {}) if phase == "B" else {})}
                    for f in FRAMEWORKS}
            for phase in ["A", "B"]
        },
    }
    return Phase1Runner(
        settings, BASE_SPACES, EXPANDED_CHOICES, EXTRA_SPACES, RUN_ROOT,
        precision_choices=PRECISION_CHOICES, trt_fp8_calibration=TRT_FP8_CALIBRATION,
        requests_by_concurrency=REQUESTS_BY_CONCURRENCY,
        pruning_min_completed_requests=PRUNING_MIN_COMPLETED_REQUESTS,
    )

# Construct from the current controls for each invocation; SQLite resumes studies.
def get_runner():
    return make_runner()

def run_objective_group(phase, target, concurrencies, pruner_name="nop", sampler_names=None, constrained=None):
    return get_runner().run_objective_group(
        phase, target, concurrencies, pruner_name, sampler_names=sampler_names,
        constrained=CONSTRAINED if constrained is None else constrained,
    )

def setup_pinned_models():
    """Explicit GPU setup; commit resolved profiles into the editable controls."""
    global PHASE_B_MODEL, PHASE_B_SNAPSHOT, MODEL_SETUP_IDENTITY, MODEL_SETUP_CHOICES
    if not RUN_REAL_GPU:
        raise ValueError("Set RUN_REAL_GPU=True before model setup")
    runner = make_runner(for_model_setup=True)
    runner.lab.RUN_REAL_GPU = True
    with runner.lock():
        summary = runner.lab.configure_hf_models(**HF_SETUP)
    framework = HF_SETUP["framework"]
    import copy
    profile = runner.lab.REAL_PREFLIGHT["B", framework]
    # Verification declarations remain editable in PREFLIGHT after setup.
    MODEL_PROFILES[framework] = {
        key: copy.deepcopy(profile[key]) for key in (
            "model_revision", "hf_target_model_id", "fixed_server", "draft_model_id",
            "draft_revision", "speculative_decoding", "framework_version", "cuda_version",
            "driver_version", "gpu_type",
        ) if key in profile
    }
    if profile["expanded_choices"] != EXPANDED_CHOICES[framework]:
        MODEL_PROFILES[framework]["expanded_choices"] = copy.deepcopy(profile["expanded_choices"])
    MODEL_SETUP_CHOICES = copy.deepcopy(EXPANDED_CHOICES[framework])
    PHASE_B_MODEL, PHASE_B_SNAPSHOT = summary["target_model_id"], summary["target_path"]
    MODEL_SETUP_IDENTITY = dict(HF_SETUP)
    return summary

# %% Batch entry point: python -m llm_optimizer.experiment.phase1_experiments
def main():
    global TRIALS, SEEDS, CONCURRENCIES, RUN_REAL_GPU, CAMPAIGN, RUN_ROOT, WANDB_ENABLED
    global REQUESTS_BY_CONCURRENCY, PRUNING_MIN_COMPLETED_REQUESTS
    global SAMPLER_NAMES, PARETO_SAMPLERS
    import argparse
    import fcntl
    import json
    import os
    import signal
    import subprocess
    import time

    import psutil

    from llm_optimizer.experiment.workflow import write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, choices=range(9))
    parser.add_argument("--smoke", action="store_true", help="two CPU trials, one group")
    parser.add_argument("--run-root", type=Path, help="persistent VM-wide output root")
    args = parser.parse_args()
    if args.run_root is not None:
        RUN_ROOT = args.run_root
    if args.smoke:
        TRIALS, SEEDS, CONCURRENCIES = 2, [11], [1, 8]
        REQUESTS_BY_CONCURRENCY = {1: 32, 8: 32, 128: 32}
        PRUNING_MIN_COMPLETED_REQUESTS = {1: 12, 8: 12, 128: 12}
        SAMPLER_NAMES = ["tpe"]
        PARETO_SAMPLERS = ["tpe"]
        RUN_REAL_GPU, CAMPAIGN = False, "phase1-smoke"
        WANDB_ENABLED = False
    tasks = [(p, t, cs, "nop") for p in ["A", "B"]
             for t, cs in [("e2e", [1]), ("ttft", [1]),
                           ("throughput", [c for c in CONCURRENCIES if c > 1]),
                           ("e2e", [c for c in CONCURRENCIES if c > 1])]]
    tasks.append(("B", "throughput", [c for c in CONCURRENCIES if c > 1], "median"))
    if args.smoke:
        tasks = [("B", "throughput", [8], "nop")]
    if args.task is not None:
        if args.task >= len(tasks):
            parser.error("Task index is outside the selected queue")
        phase, target, cs, pruner = tasks[args.task]
        run_objective_group(phase, target, cs, pruner_name=pruner)
        return 0
    if WANDB_ENABLED and WANDB_MODE == "online":
        import wandb
        wandb.login()  # preauthenticate/set WANDB_API_KEY for unattended workers
    runner = get_runner()
    status_path = runner.run_dir / "queue.json"
    with (runner.run_dir / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
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
                    return 0
                status["attempts"][key] = status["attempts"].get(key, 0) + 1
                write_json(status_path, status)
                command = [sys.executable, str(Path(__file__).resolve()),
                           "--task", key, "--run-root", str(runner.run_root)]
                if args.smoke:
                    command.append("--smoke")
                process = subprocess.Popen(command, start_new_session=True)
                try:
                    result = process.wait(timeout=min(GROUP_TIMEOUT_SECONDS, remaining))
                except subprocess.TimeoutExpired:
                    result = 124
                finally:
                    # TRT engines can start a new session: capture the whole tree.
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
