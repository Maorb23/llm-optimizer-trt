"""Algorithms extracted from the reference notebook, with instance settings."""

import hashlib
import importlib.util
import json
import math
import platform
import random
import shlex
import time
import warnings
from collections import Counter
from contextlib import closing
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import optuna
from optuna.trial import TrialState
from scipy.stats import spearmanr
from scipy.stats import t as student_t


def display(value):
    """Notebook display calls are deliberately inert in command workflows."""


PREFILL_BUDGETS_1 = list(range(1024, 8193, 1024))
TOKEN_BUDGETS_128 = [2048, 4096, 6144, 8192]
CAPACITIES_128 = [8, 16, 24, 32]
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
            "gpu_memory_utilization": [0.75, 0.8, 0.85, 0.9],
        },
        "sglang": {
            "max_running_requests": CAPACITIES_128,
            "chunked_prefill_size": TOKEN_BUDGETS_128,
            "mem_fraction_static": [0.7, 0.75, 0.8, 0.85],
        },
        "trtllm": {
            "max_batch_size": CAPACITIES_128,
            "max_num_tokens": TOKEN_BUDGETS_128,
            "kv_cache_free_gpu_memory_fraction": [0.65, 0.75, 0.85, 0.95],
        },
    },
}
FLOAT_PARAMS = {
    "gpu_memory_utilization",
    "mem_fraction_static",
    "kv_cache_free_gpu_memory_fraction",
}
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


class BenchmarkFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class Measurement:
    output_throughput: float
    p95_ttft_ms: float
    completed: int
    total_requests: int
    failed: int
    wall_seconds: float
    gpu_seconds: float
    source: str
    p95_e2e_latency_ms: float

    @property
    def failure_rate(self):
        return self.failed / self.total_requests if self.total_requests else 1.0


class Simulator:
    supports_checkpoints = True

    def __init__(
        self, seed=None, phase="A", num_prompts=None, workload_seed=None, *, _context
    ):
        self._context = _context
        self.phase = phase
        self.num_prompts = num_prompts
        self.workload_seed = workload_seed
        self.environment = {
            **self._context.ENVIRONMENT,
            "phase": phase,
            "target_model": self._context.PHASE_MODELS[phase],
        }
        self.rng = np.random.default_rng(
            self._context.SIMULATION_SEED if seed is None else seed
        )

    def measure(self, framework, config, concurrency, fraction=1.0):
        workload = self._context.workload_for(
            concurrency, self.phase, self.num_prompts, self.workload_seed
        )
        capacity, tokens, memory, graph_on = self._context.engine_features(
            framework, config
        )
        if concurrency > 1 and capacity == 32 and (tokens == 2048) and (memory >= 0.9):
            raise BenchmarkFailure("Synthetic invalid high-memory configuration.")
        prefill_shape = np.log2(tokens / 4096) ** 2
        if concurrency == 1:
            ttft = (55 + 12 * prefill_shape) * (0.83 if graph_on else 1.0)
            rate = 420 / (1 + 0.05 * prefill_shape) * (1.1 if graph_on else 1.0)
        else:
            saturation = (1 - np.exp(-capacity / 12)) * (1 - np.exp(-tokens / 2200))
            rate = 2600 * saturation * (1 - 0.5 * (memory - 0.85) ** 2)
            rate *= 1.05 if graph_on else 1.0
            ttft = 50 + 2.7 * capacity + tokens / 160
        if "max_num_partial_prefills" in config:
            rate *= 1 + 0.015 * config["max_num_partial_prefills"]
            ttft *= 0.97
        if config.get("schedule_policy") == "lpm":
            rate *= 1.01
        graph = config.get("cuda_graph_config")
        if isinstance(graph, dict) and graph.get("enable_padding") is False:
            rate *= 0.98
        if self.phase == "B":
            rate *= 0.65
            ttft *= 1.25
        if config.get("async_scheduling"):
            rate *= 1.03
        if config.get("performance_mode") == "interactivity":
            ttft *= 0.97
        if config.get("performance_mode") == "throughput":
            rate *= 1.02
        if config.get("kv_cache_dtype") == "fp8":
            rate *= 1.02
        if "schedule_conservativeness" in config:
            ttft *= 1 + 0.03 * (config["schedule_conservativeness"] - 1)
        early_bias = (1 - fraction) * (-0.12 if capacity >= 24 else 0.03)
        rate *= np.exp(self.rng.normal(early_bias, 0.02 / np.sqrt(fraction)))
        ttft *= np.exp(self.rng.normal(0, 0.02 / np.sqrt(fraction)))
        per_request_rate = rate / max(1, min(concurrency, capacity))
        e2e = ttft + workload["random_output_len"] / per_request_rate * 1000
        e2e *= np.exp(self.rng.normal(0, 0.02 / np.sqrt(fraction)))
        count = max(1, int(workload["num_prompts"] * fraction))
        wall = 35 + 8 * graph_on + count * workload["random_output_len"] / rate
        return self._context.validate_measurement(
            Measurement(
                float(rate),
                float(ttft),
                count,
                count,
                0,
                float(wall),
                float(wall),
                "SYNTHETIC",
                float(e2e),
            )
        )

    def stream(self, framework, config, concurrency, checkpoint_requests=None):
        previous_gpu = 0.0
        from llm_optimizer.optimization import checkpoint_schedule

        count = self._context.workload_for(concurrency, self.phase, self.num_prompts)[
            "num_prompts"
        ]
        for step in checkpoint_requests or checkpoint_schedule(count):
            fraction = step / count
            m = self.measure(framework, config, concurrency, fraction)
            increment = max(0.001, m.gpu_seconds - previous_gpu)
            previous_gpu += increment
            yield (step, fraction, m, increment)


class CartesianPoints:
    """Indexable mixed-radix grid with a seeded bijection and lazy allocation."""

    def __init__(self, menu, seed):
        self.values = [menu[k] for k in sorted(menu)]
        self.size = math.prod(map(len, self.values))
        rng = random.Random(seed)
        self.offset = rng.randrange(self.size)
        self.stride = rng.randrange(1, self.size + 1)
        while math.gcd(self.stride, self.size) != 1:
            self.stride = rng.randrange(1, self.size + 1)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if index < 0 or index >= self.size:
            raise IndexError(index)
        code = (self.offset + index * self.stride) % self.size
        point = []
        for values in reversed(self.values):
            code, digit = divmod(code, len(values))
            point.append(values[digit])
        return tuple(reversed(point))


class LazyGridSampler(optuna.samplers.GridSampler):
    """Optuna 4.9 GridSampler behavior with lazy, seeded Cartesian enumeration.

    Uses tested private grid fields, so retain the documented Optuna pin.
    The affine traversal is bijective, not Optuna's full-list random shuffle.
    """

    def __init__(self, menu, seed):
        super().__init__({k: [v[0]] for k, v in menu.items()}, seed=seed)
        self._search_space = {k: list(menu[k]) for k in sorted(menu)}
        self._all_grids = CartesianPoints(self._search_space, seed)
        self._n_min_trials = len(self._all_grids)

    def _get_unvisited_grid_ids(self, study):
        trials = study.get_trials(deepcopy=False)
        used = {
            t.system_attrs["grid_id"] for t in trials if "grid_id" in t.system_attrs
        }
        return [
            i for i in range(min(self._n_min_trials, len(used) + 1024)) if i not in used
        ]


class NotebookWandbLogger:
    """One optional W&B run per study; telemetry never changes search outcomes.

    Use the repository tracker for failure-isolated shutdown. Its CLI trial
    helper is scalar-only, so this callback logs objective vectors explicitly.
    """

    def __init__(self, study, contract, *, _context):
        self._context = _context
        self.tracker = None
        self.contract = contract
        if not self._context.WANDB_ENABLED:
            return
        try:
            import wandb

            from llm_optimizer.wandb_tracking import WandbTracker

            run = wandb.init(
                project=self._context.WANDB_PROJECT,
                entity=self._context.WANDB_ENTITY,
                mode=self._context.WANDB_MODE,
                group=self._context.WANDB_GROUP,
                name=study.study_name,
                job_type="optuna-study",
                tags=[
                    contract["phase"],
                    contract["framework"],
                    contract["target"],
                    contract["environment"]["measurement"],
                ],
                config={"study_name": study.study_name, **contract},
                dir=str(self._context.RUN_DIR),
            )
            self.tracker = WandbTracker(run)
            study.set_user_attr("wandb", {"id": run.id, "url": run.url})
        except Exception:
            warnings.warn(
                "W&B initialization failed; continuing without tracking", stacklevel=2
            )
            if self.tracker is not None:
                self.tracker.finish()
                self.tracker = None

    def __call__(self, study, trial):
        if self.tracker is None:
            return
        try:
            payload = {
                "trial/number": trial.number,
                "trial/state": trial.state.name,
                "trial/config_id": trial.user_attrs.get("config_id"),
                "trial/config": trial.user_attrs.get("config", {}),
                "trial/failure_reason": trial.user_attrs.get("failure_reason"),
                "trial/slo_feasible": trial.user_attrs.get("slo_feasible"),
                "trial/constraint_values": trial.user_attrs.get(
                    "constraint_values", []
                ),
                "trial/intermediate_values": {
                    str(k): v for k, v in trial.intermediate_values.items()
                },
                "cost/cumulative_gpu_seconds": sum(
                    t.user_attrs.get("gpu_seconds", 0) for t in study.trials
                ),
            }
            payload.update({f"parameters/{k}": v for k, v in trial.params.items()})
            payload.update(
                {
                    f"benchmark/{k}": v
                    for k, v in trial.user_attrs.get("measurement", {}).items()
                }
            )
            payload.update(
                {
                    f"requests/{k}": v
                    for k, v in trial.user_attrs.get("request_accounting", {}).items()
                }
            )
            for key in (
                "gpu_seconds",
                "orchestrator_wall_seconds",
                "pruned_at_fraction",
            ):
                if key in trial.user_attrs:
                    payload[f"trial/{key}"] = trial.user_attrs[key]
            if trial.state == TrialState.COMPLETE and trial.values is not None:
                payload.update(
                    {
                        f"objective/{metric}": value
                        for metric, value in zip(self.contract["metrics"], trial.values)
                    }
                )
            candidates = self._context.completed(study, self.contract["constrained"])
            if len(study.directions) == 1 and candidates:
                best = (min if study.directions[0].name == "MINIMIZE" else max)(
                    t.value for t in candidates
                )
                payload[f"best/{self.contract['metrics'][0]}"] = best
            self.tracker.run.log(payload, step=trial.number)
        except Exception:
            warnings.warn(
                "W&B trial logging failed; continuing optimization", stacklevel=2
            )

    def finish(self, study, interrupted=False):
        if self.tracker is None:
            return
        try:
            candidates = self._context.completed(study, self.contract["constrained"])
            winners = (
                self._context.pareto_front(candidates, study.directions)
                if len(study.directions) > 1
                else [self._context.choose_single(study, self.contract["constrained"])]
            )
            self.tracker.run.summary.update(
                {
                    "study": self._context.summarize(study)
                    if "host_search_and_storage_seconds" in study.user_attrs
                    else {"attempts": len(study.trials)},
                    "interrupted": interrupted,
                    "selection": "feasible"
                    if self.contract["constrained"]
                    else "unconstrained",
                    "best_or_pareto_trials": [
                        {
                            "number": t.number,
                            "values": t.values,
                            "params": t.params,
                            "config": t.user_attrs.get("config"),
                            "slo_feasible": t.user_attrs.get("slo_feasible"),
                        }
                        for t in winners
                        if t is not None
                    ],
                }
            )
        except Exception:
            warnings.warn("W&B summary logging failed; continuing", stacklevel=2)
        finally:
            self.tracker.finish()


class RepositoryBenchmark:
    supports_checkpoints = True

    def __init__(
        self,
        framework,
        model_snapshot,
        gpu_type,
        concurrency,
        gpu_count=1,
        rest_seconds=10,
        fixed_server=None,
        environment=None,
        phase="A",
        num_prompts=None,
        workload_seed=None,
        run_label="search",
        *,
        _context,
    ):
        self._context = _context
        from llm_optimizer.bench_client import prepare_workload_manifest
        from llm_optimizer.predefined import get_framework_spec

        self.framework = framework
        self.concurrency = concurrency
        self.phase = phase
        if phase not in self._context.PHASE_MODELS:
            raise ValueError("Unknown experiment phase")
        self.workload = self._context.workload_for(
            concurrency, phase, num_prompts, workload_seed
        )
        self.model = str(model_snapshot)
        self.gpu_type = gpu_type
        self.gpu_count = gpu_count
        self.rest_seconds = rest_seconds
        self.fixed_server = dict(fixed_server or {})
        self.environment = {
            **dict(environment or {}),
            "measurement": "MEASURED",
            "python": platform.python_version(),
            "optuna": optuna.__version__,
            "gpu_type": gpu_type,
            "gpu_count": gpu_count,
            "model": self.model,
            "phase": phase,
            "target_model": self._context.PHASE_MODELS[phase],
            "concurrency": concurrency,
        }
        if not Path(self.model).is_dir():
            raise ValueError("Use an existing pinned local model/tokenizer snapshot.")
        required = {
            "framework_version",
            "cuda_version",
            "driver_version",
            "model_revision",
        }
        if not required.issubset(self.environment):
            raise ValueError(f"Record environment fields: {sorted(required)}")
        if self.environment.get("native_config_verified") is not True:
            raise ValueError(
                "Verify native graph/config semantics for the pinned version."
            )
        self.spec = get_framework_spec(framework)
        self.host = "127.0.0.1"
        self.port = self.spec.default_port
        workload_args = {
            **self.workload,
            "model": self.model,
            "phase": phase,
            "target_model": self._context.PHASE_MODELS[phase],
            "backend": self.spec.benchmark_backend,
            "model_revision": self.environment["model_revision"],
            "tokenizer_revision": self.environment["model_revision"],
            "host": self.host,
            "port": self.port,
        }
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        try:
            random.seed(self.workload["seed"])
            np.random.seed(self.workload["seed"])
            self.manifest, self.workload_metadata = prepare_workload_manifest(
                workload_args
            )
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
        request_payload = json.dumps(self.manifest["requests"], sort_keys=True)
        self.workload_metadata["requests_sha256"] = hashlib.sha256(
            request_payload.encode()
        ).hexdigest()
        workload_path = (
            self._context.RUN_DIR
            / f"{run_label}-{phase}-{framework}-c{concurrency}-workload.json"
        )
        workload_path.write_text(json.dumps(self.manifest, indent=2))
        self.environment.update(self.workload_metadata)
        self.output_path = (
            self._context.RUN_DIR
            / f"{run_label}-{phase}-{framework}-c{concurrency}-measured.jsonl"
        )

    def measure(
        self,
        framework,
        config,
        concurrency,
        checkpoint_callback=None,
        checkpoint_requests=None,
    ):
        from concurrent.futures import ThreadPoolExecutor

        from llm_optimizer.args import Arg, ArgScope
        from llm_optimizer.cli import (
            construct_benchmark_settings,
            run_single_configuration,
        )
        from llm_optimizer.predefined import CLIENT_CONFIGS, SERVER_CONFIGS

        if framework != self.framework or concurrency != self.concurrency:
            raise ValueError("Benchmark framework/concurrency differs from the study.")

        def execute():
            started = time.perf_counter()
            accounting = {}
            try:
                merged = {**self.fixed_server, **config}
                if framework == "sglang":
                    source = (
                        "fixed_server"
                        if "disable_cuda_graph" in self.fixed_server
                        else "sampled config"
                        if "disable_cuda_graph" in config
                        else "absent"
                    )
                    print(
                        "SGLang config diagnostic:",
                        {
                            "disable_cuda_graph_source": source,
                            "disable_cuda_graph_value": merged.get(
                                "disable_cuda_graph", "<absent>"
                            ),
                            "fixed_server": self.fixed_server,
                            "sampled_config": config,
                            "merged_config": merged,
                        },
                    )
                native = self._context.serialize_engine_config(
                    framework, concurrency, merged
                )
                args = []
                for scope, values, registry in [
                    (ArgScope.SERVER, native, SERVER_CONFIGS[framework]),
                    (ArgScope.CLIENT, self.workload, CLIENT_CONFIGS),
                ]:
                    for key, value in values.items():
                        extra_types = {
                            "vllm": {
                                "attention_backend": str,
                                "performance_mode": str,
                                "no_async_scheduling": bool,
                                "no_enable_prefix_caching": bool,
                            },
                            "sglang": {
                                "cuda_graph_backend_decode": str,
                                "cuda_graph_backend_prefill": str,
                            },
                            "trtllm": {},
                        }[framework]
                        if scope == ArgScope.SERVER and key in extra_types:
                            args.append(
                                Arg[extra_types[key]](
                                    scope=scope, name=key, value=value
                                )
                            )
                            continue
                        if key not in registry:
                            raise ValueError(f"Unknown native argument: {key}")
                        args.append(
                            Arg[registry[key].arg_type](
                                scope=scope, name=key, value=value
                            )
                        )
                settings = construct_benchmark_settings(args, framework)
                settings["server_cmd_args"] = [
                    shlex.quote(arg) for arg in settings["server_cmd_args"]
                ]
                command = self.spec.server_command.format(
                    model=shlex.quote(self.model), host=self.host, port=self.port
                )

                def report_checkpoint(values):
                    nonlocal accounting
                    accounting = {
                        key: int(values[key])
                        for key in ("completed", "total_requests", "failed")
                    }
                    wall = max(time.perf_counter() - started, 1e-09)
                    m = self._context.validate_measurement(
                        Measurement(
                            float(values["output_throughput"]),
                            float(values["p95_ttft_ms"]),
                            accounting["completed"],
                            accounting["total_requests"],
                            accounting["failed"],
                            wall,
                            wall * self.gpu_count,
                            "MEASURED_ALLOCATION_WALL_PROXY",
                            float(values["p95_e2e_latency_ms"]),
                        )
                    )
                    checkpoint_callback(m)

                result, _ = run_single_configuration(
                    settings,
                    framework=framework,
                    model=self.model,
                    server_cmd=command,
                    managed_server=True,
                    gpus=self.gpu_count,
                    gpu_type=self.gpu_type,
                    host=self.host,
                    port=self.port,
                    ready_url=f"http://{self.host}:{self.port}{self.spec.ready_endpoint}",
                    mute_server=False,
                    constraints_for_viz=[],
                    output_jsonl_path=self.output_path,
                    checkpoint_requests=checkpoint_requests,
                    checkpoint_callback=report_checkpoint
                    if checkpoint_callback
                    else None,
                    workload_manifest=self.manifest,
                    workload_metadata={**self.workload_metadata, **self.environment},
                )
                values = result["results"]
                expected = self.workload["num_prompts"]
                done = int(values["completed"])
                total = int(values.get("total_requests", expected))
                accounting = {
                    "completed": done,
                    "total_requests": total,
                    "failed": int(values.get("failed", total - done)),
                }
                if total != expected or values.get("benchmark_valid") is False:
                    raise BenchmarkFailure("Invalid or changed benchmark workload.")
                if self.rest_seconds:
                    time.sleep(self.rest_seconds)
                wall = time.perf_counter() - started
                return self._context.validate_measurement(
                    Measurement(
                        float(values["output_throughput"]),
                        float(values["p95_ttft_ms"]),
                        done,
                        total,
                        int(values.get("failed", total - done)),
                        wall,
                        wall * self.gpu_count,
                        "MEASURED_ALLOCATION_WALL_PROXY",
                        float(values["p95_e2e_latency_ms"]),
                    )
                )
            except Exception as exc:
                from llm_optimizer.exceptions import ServerCleanupError

                if isinstance(exc, (ServerCleanupError, optuna.TrialPruned)):
                    exc.gpu_seconds = (time.perf_counter() - started) * self.gpu_count
                    exc.request_accounting = accounting
                    raise
                failure = BenchmarkFailure(f"{type(exc).__name__}: {exc}")
                failure.request_accounting = getattr(
                    exc, "request_accounting", accounting
                )
                failure.gpu_seconds = (time.perf_counter() - started) * self.gpu_count
                raise failure from exc

        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(execute).result()


class Lab:
    def __init__(self, settings):
        self.__dict__.update(settings)
        self.Simulator = partial(Simulator, _context=self)
        self.NotebookWandbLogger = partial(NotebookWandbLogger, _context=self)
        self.RepositoryBenchmark = partial(RepositoryBenchmark, _context=self)
        self.EXPANDED_CHOICES = EXPANDED_CHOICES
        self.BenchmarkFailure = BenchmarkFailure
        self.LazyGridSampler = LazyGridSampler
        self.Measurement = Measurement
        self.FLOAT_PARAMS = FLOAT_PARAMS
        self.SPACES = SPACES
        self.PREFILL_BUDGETS_1 = PREFILL_BUDGETS_1
        self.CAPACITIES_128 = CAPACITIES_128
        self.TOKEN_BUDGETS_128 = TOKEN_BUDGETS_128
        self.CartesianPoints = CartesianPoints

    def regime_key(self, concurrency):
        if (
            not isinstance(concurrency, int)
            or isinstance(concurrency, bool)
            or concurrency < 1
        ):
            raise ValueError("Concurrency must be a positive integer")
        return 1 if concurrency == 1 else 128

    def workload_for(self, concurrency, phase="A", num_prompts=None, seed=None):
        self.regime_key(concurrency)
        if concurrency not in self.CONCURRENCIES:
            raise ValueError(f"Supported concurrency regimes: {self.CONCURRENCIES}")
        if phase not in self.PHASE_MODELS:
            raise ValueError("Unknown experiment phase")
        count = self.PHASE_REQUESTS[phase] if num_prompts is None else num_prompts
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError("num_prompts must be a positive integer")
        return {
            **self.BASE_WORKLOAD,
            "num_prompts": count,
            "max_concurrency": concurrency,
            "seed": self.BASE_WORKLOAD["seed"] if seed is None else seed,
        }

    def default_target(self, concurrency):
        self.regime_key(concurrency)
        return "e2e"

    def objectives_for(self, concurrency):
        self.regime_key(concurrency)
        return (
            ("e2e", "ttft", "pareto_latency")
            if concurrency == 1
            else ("e2e", "throughput", "pareto_throughput")
        )

    def objective_spec(self, target, concurrency):
        if target == "pareto":
            target = "pareto_latency" if concurrency == 1 else "pareto_throughput"
        specs = {
            "e2e": (["p95_e2e_latency_ms"], ["minimize"]),
            "ttft": (["p95_ttft_ms"], ["minimize"]),
            "throughput": (["output_throughput"], ["maximize"]),
            "pareto_latency": (
                ["p95_e2e_latency_ms", "p95_ttft_ms"],
                ["minimize", "minimize"],
            ),
            "pareto_throughput": (
                ["output_throughput", "p95_e2e_latency_ms"],
                ["maximize", "minimize"],
            ),
        }
        if target not in specs:
            raise ValueError(f"Unknown objective: {target}")
        return (target, *specs[target])

    def trial_budget(self, phase, concurrency):
        return (self.DEMO_TRIALS if self.QUICK_DEMO else self.SEARCH_TRIALS)[phase][
            self.regime_key(concurrency)
        ]

    def grid_size(self, concurrency, framework):
        return math.prod(
            len(v)
            for v in self.SPACES[self.regime_key(concurrency)][framework].values()
        )

    def suggest_numeric(self, trial, name, values):
        if len(values) < 2:
            raise ValueError(
                "Define at least two distinct values for each sampled dimension."
            )
        step = round(values[1] - values[0], 6)
        if step <= 0 or not all(
            (
                math.isclose(v, values[0] + i * step, rel_tol=0, abs_tol=1e-08)
                for i, v in enumerate(values)
            )
        ):
            raise ValueError(f"{name} must be an increasing regular numerical grid.")
        if name in self.FLOAT_PARAMS:
            return round(trial.suggest_float(name, values[0], values[-1], step=step), 6)
        if not all(isinstance(v, int) for v in values):
            raise ValueError(f"{name} requires integer values.")
        return trial.suggest_int(name, values[0], values[-1], step=int(step))

    def render_engine_config(self, framework, concurrency, p):
        self.workload_for(concurrency)
        graph_on = bool(p["cuda_graph_on"]) if concurrency == 1 else True
        if framework == "vllm":
            return {
                "max_num_seqs": 1 if concurrency == 1 else p["max_num_seqs"],
                "max_num_batched_tokens": p["max_num_batched_tokens"],
                "gpu_memory_utilization": 0.85
                if concurrency == 1
                else p["gpu_memory_utilization"],
                "enable_chunked_prefill": True,
                "compilation_config": {
                    "cudagraph_mode": "FULL_AND_PIECEWISE" if graph_on else "NONE"
                },
            }
        if framework == "sglang":
            return {
                "max_running_requests": 1
                if concurrency == 1
                else p["max_running_requests"],
                "chunked_prefill_size": p["chunked_prefill_size"],
                "mem_fraction_static": 0.8
                if concurrency == 1
                else p["mem_fraction_static"],
                "cuda_graph_backend_decode": "full" if graph_on else "disabled",
                "cuda_graph_backend_prefill": "full" if graph_on else "disabled",
                "schedule_conservativeness": 1.0,
            }
        if framework == "trtllm":
            max_batch = 1 if concurrency == 1 else p["max_batch_size"]
            return {
                "max_batch_size": max_batch,
                "max_num_tokens": p["max_num_tokens"],
                "kv_cache_config": {
                    "free_gpu_memory_fraction": 0.8
                    if concurrency == 1
                    else p["kv_cache_free_gpu_memory_fraction"]
                },
                "cuda_graph_config": {
                    "batch_sizes": [
                        s for s in [1, 2, 4, 8, 16, 24, 32] if s <= max_batch
                    ],
                    "enable_padding": True,
                }
                if graph_on
                else None,
            }
        raise ValueError(framework)

    def expanded_choices(self, framework, choices=None):
        selected = self.EXPANDED_CHOICES[framework] if choices is None else choices
        if set(selected) != set(self.EXPANDED_CHOICES[framework]):
            raise ValueError(
                "Preflight must account for every expanded categorical knob."
            )
        if any(
            (
                not values
                or any((v not in self.EXPANDED_CHOICES[framework][key] for v in values))
                for key, values in selected.items()
            )
        ):
            raise ValueError(
                "Use nonempty subsets of the documented candidate choices."
            )
        return selected

    def expanded_config(self, trial, framework, concurrency, config, choices=None):
        choices = self.expanded_choices(framework, choices)

        def pick(name):
            return trial.suggest_categorical(name, choices[name])

        memory_name = {
            "vllm": "gpu_memory_utilization",
            "sglang": "mem_fraction_static",
            "trtllm": "kv_cache_free_gpu_memory_fraction",
        }[framework]
        if concurrency == 1:
            memory = self.suggest_numeric(
                trial, memory_name, self.SPACES[128][framework][memory_name]
            )
            if framework == "trtllm":
                config["kv_cache_config"]["free_gpu_memory_fraction"] = memory
            else:
                config[memory_name] = memory
        capacity = self.engine_features(framework, config)[0]

        def graph_sizes():
            dense = trial.suggest_categorical(
                "graph_capture_profile", ["powers", "dense"]
            )
            return (
                list(range(1, capacity + 1))
                if dense == "dense"
                else sorted(
                    {1, capacity, *[s for s in [2, 4, 8, 16, 32] if s <= capacity]}
                )
            )

        if framework == "vllm":
            for key in [
                "async_scheduling",
                "performance_mode",
                "attention_backend",
                "kv_cache_dtype",
                "enable_prefix_caching",
            ]:
                config[key] = pick(key)
            mode = pick("graph_mode")
            config["compilation_config"] = {"cudagraph_mode": mode}
            if mode != "NONE":
                config["compilation_config"]["cudagraph_capture_sizes"] = graph_sizes()
        elif framework == "sglang":
            from llm_optimizer.experiment.sglang_space import (
                CACHE_POLICY_PARAMETER,
                resolve_cache_policy,
                sampling_choices,
            )

            domain = sampling_choices(choices)[CACHE_POLICY_PARAMETER]
            config.update(
                resolve_cache_policy(
                    trial.suggest_categorical(CACHE_POLICY_PARAMETER, domain)
                )
            )
            for key in [
                "attention_backend",
                "enable_mixed_chunk",
                "disable_overlap_schedule",
                "cuda_graph_backend_decode",
                "cuda_graph_backend_prefill",
            ]:
                config[key] = pick(key)
            config["schedule_conservativeness"] = round(
                trial.suggest_float("schedule_conservativeness", 0.5, 1.5, step=0.25), 6
            )
            config["max_prefill_tokens"] = trial.suggest_int(
                "max_prefill_tokens", 8192, 32768, step=8192
            )
        else:
            config["attn_backend"] = pick("attn_backend")
            config["enable_chunked_prefill"] = pick("enable_chunked_prefill")
            if not config["enable_chunked_prefill"]:
                config["max_num_tokens"] = max(
                    config["max_num_tokens"], self.BASE_WORKLOAD["random_input_len"]
                )
            config["kv_cache_config"]["enable_block_reuse"] = pick("enable_block_reuse")
            config["scheduler_config"] = {
                "capacity_scheduler_policy": pick("capacity_scheduler_policy")
            }
            config["cuda_graph_config"] = (
                {
                    "batch_sizes": graph_sizes(),
                    "enable_padding": bool(trial.suggest_int("graph_padding", 0, 1)),
                }
                if pick("graph_enabled")
                else None
            )
        return config

    def suggest_config(
        self, trial, framework, concurrency=128, space_kind="baseline", choices=None
    ):
        self.workload_for(concurrency)
        if space_kind not in {"baseline", "expanded"}:
            raise ValueError("Use baseline or expanded, with a fresh study per space.")
        sampled = {
            name: self.suggest_numeric(trial, name, values)
            for name, values in self.SPACES[self.regime_key(concurrency)][
                framework
            ].items()
            if not (space_kind == "expanded" and name == "cuda_graph_on")
        }
        if space_kind == "expanded" and concurrency == 1:
            sampled["cuda_graph_on"] = 1
        config = self.render_engine_config(framework, concurrency, sampled)
        return (
            self.expanded_config(trial, framework, concurrency, config, choices)
            if space_kind == "expanded"
            else config
        )

    def config_fingerprint(self, framework, concurrency, config, phase="A"):
        if phase not in self.PHASE_MODELS:
            raise ValueError("Unknown experiment phase")
        payload = json.dumps(
            [self.PHASE_MODELS[phase], framework, concurrency, config], sort_keys=True
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def measurement_failure(self, reason, m):
        failure = BenchmarkFailure(reason)
        failure.request_accounting = {
            "completed": m.completed,
            "failed": m.failed,
            "total_requests": m.total_requests,
        }
        failure.gpu_seconds = m.gpu_seconds
        return failure

    def validate_measurement(self, m):
        if not all(
            np.isfinite(v) and v > 0
            for v in [
                m.output_throughput,
                m.p95_ttft_ms,
                m.p95_e2e_latency_ms,
                m.wall_seconds,
                m.gpu_seconds,
            ]
        ):
            raise self.measurement_failure(
                "Missing, nonfinite, zero, or negative metrics.", m
            )
        if (
            m.total_requests <= 0
            or m.completed <= 0
            or m.failed < 0
            or (m.completed + m.failed != m.total_requests)
        ):
            raise self.measurement_failure("Invalid request accounting.", m)
        if m.failure_rate > self.FAILURE_RATE_LIMIT:
            raise self.measurement_failure(
                "Incomplete/rejected requests invalidate this trial.", m
            )
        return m

    def engine_features(self, framework, config):
        if framework == "vllm":
            return (
                config["max_num_seqs"],
                config["max_num_batched_tokens"],
                config["gpu_memory_utilization"],
                config["compilation_config"]["cudagraph_mode"] != "NONE",
            )
        if framework == "sglang":
            graph_on = (
                config.get("cuda_graph_backend_decode", "disabled") != "disabled"
                or config.get("cuda_graph_backend_prefill", "disabled") != "disabled"
            )
            return (
                config["max_running_requests"],
                config["chunked_prefill_size"],
                config["mem_fraction_static"],
                graph_on,
            )
        graph = config["cuda_graph_config"]
        return (
            config["max_batch_size"],
            config["max_num_tokens"],
            config["kv_cache_config"]["free_gpu_memory_fraction"],
            graph is not None,
        )

    def constraints(self, frozen_trial):
        return frozen_trial.user_attrs.get("constraint_values", [float("inf")])

    def record_measurement(self, trial, m, concurrency):
        trial.set_user_attr("measurement", vars(m))
        trial.set_user_attr(
            "constraint_values",
            [
                m.p95_ttft_ms - self.TTFT_LIMITS_MS[self.regime_key(concurrency)],
                m.p95_e2e_latency_ms - self.E2E_LIMITS_MS[self.regime_key(concurrency)],
                m.failure_rate - self.FAILURE_RATE_LIMIT,
            ],
        )
        trial.set_user_attr(
            "slo_feasible", all(v <= 0 for v in self.constraints(trial))
        )

    def sampling_menu(
        self, framework, concurrency, space_kind="baseline", choices=None
    ):
        menu = dict(self.SPACES[self.regime_key(concurrency)][framework])
        if space_kind == "baseline":
            return menu
        menu.pop("cuda_graph_on", None)
        if concurrency == 1:
            memory = {
                "vllm": "gpu_memory_utilization",
                "sglang": "mem_fraction_static",
                "trtllm": "kv_cache_free_gpu_memory_fraction",
            }[framework]
            menu[memory] = self.SPACES[128][framework][memory]
        selected = self.expanded_choices(framework, choices)
        if framework == "sglang":
            from llm_optimizer.experiment.sglang_space import sampling_choices

            selected = sampling_choices(selected)
        menu.update(selected)
        menu["graph_capture_profile"] = ["powers", "dense"]
        if framework == "sglang":
            menu.update(
                schedule_conservativeness=[0.5, 0.75, 1.0, 1.25, 1.5],
                max_prefill_tokens=[8192, 16384, 24576, 32768],
                graph_padding_off=[0, 1],
            )
        else:
            menu["graph_padding"] = [0, 1]
        return menu

    def make_sampler(
        self,
        name,
        framework,
        concurrency=128,
        seed=42,
        constrained=False,
        space_kind="baseline",
        choices=None,
    ):
        self.workload_for(concurrency)
        if constrained and name not in {"tpe", "tpe_multivariate", "gp", "nsga2"}:
            raise ValueError("Select a constraint-aware sampler.")
        extra = {"constraints_func": self.constraints} if constrained else {}
        startup = self.SAMPLER_SETTINGS[
            "startup_single" if concurrency == 1 else "startup_multi"
        ]
        if name == "grid":
            menu = self.sampling_menu(framework, concurrency, space_kind, choices)
            return (
                optuna.samplers.GridSampler(menu, seed=seed)
                if space_kind == "baseline"
                else LazyGridSampler(menu, seed)
            )
        if name == "random":
            return optuna.samplers.RandomSampler(seed=seed)
        if name == "qmc":
            return optuna.samplers.QMCSampler(
                qmc_type="sobol",
                scramble=True,
                seed=seed,
                independent_sampler=optuna.samplers.RandomSampler(seed=seed),
            )
        if name in {"tpe", "tpe_multivariate"}:
            return optuna.samplers.TPESampler(
                seed=seed,
                n_startup_trials=startup,
                n_ei_candidates=self.SAMPLER_SETTINGS["n_ei_candidates"],
                multivariate=name == "tpe_multivariate",
                group=name == "tpe_multivariate" and space_kind == "expanded",
                constant_liar=False,
                **extra,
            )
        if name == "gp":
            if importlib.util.find_spec("torch") is None:
                raise ImportError(
                    "Install torch; GP is not replaced with another sampler."
                )
            return optuna.samplers.GPSampler(
                seed=seed,
                n_startup_trials=startup,
                deterministic_objective=False,
                **extra,
            )
        if name == "nsga2":
            return optuna.samplers.NSGAIISampler(
                seed=seed,
                population_size=self.SAMPLER_SETTINGS["population_size"],
                **extra,
            )
        raise ValueError(name)

    def create_phase_a_study(
        self,
        sampler_name,
        framework,
        concurrency=128,
        seed=42,
        name=None,
        target=None,
        constrained=False,
        space_kind="baseline",
        pruner=None,
        phase=None,
        choices=None,
    ):
        target = target or self.default_target(concurrency)
        sampler = self.make_sampler(
            sampler_name, framework, concurrency, seed, constrained, space_kind, choices
        )
        study = optuna.create_study(
            study_name=name,
            storage=self.STORAGE,
            sampler=sampler,
            pruner=pruner or optuna.pruners.NopPruner(),
            directions=self.objective_spec(target, concurrency)[2],
            load_if_exists=True,
        )
        if sampler_name == "grid":
            return (
                study,
                None
                if space_kind == "baseline"
                else self.trial_budget(phase or "B", concurrency),
            )
        if not study.trials:
            initial = {
                name: values[len(values) // 2]
                for name, values in self.SPACES[self.regime_key(concurrency)][
                    framework
                ].items()
                if not (space_kind == "expanded" and name == "cuda_graph_on")
            }
            study.enqueue_trial(initial)
        return (
            study,
            self.trial_budget(
                phase or ("B" if space_kind == "expanded" else "A"), concurrency
            ),
        )

    def completed(self, study, feasible_only=False):
        return [
            t
            for t in study.trials
            if t.state == TrialState.COMPLETE
            and (not feasible_only or t.user_attrs.get("slo_feasible", False))
        ]

    def run_study(
        self,
        name,
        framework,
        sampler_name="tpe_multivariate",
        concurrency=128,
        seed=11,
        trials=None,
        target=None,
        constrained=False,
        space_kind="baseline",
        pruner=None,
        benchmark=None,
        gpu_budget=None,
        pruner_settings=None,
        phase=None,
        choices=None,
        num_prompts=None,
        min_completed_requests=1,
        checkpointed=False,
    ):
        phase = phase or ("B" if space_kind == "expanded" else "A")
        workload = self.workload_for(concurrency, phase, num_prompts)
        target, metrics, directions = self.objective_spec(
            target or self.default_target(concurrency), concurrency
        )
        benchmark = benchmark or self.Simulator(
            seed=self.SIMULATION_SEED, phase=phase, num_prompts=workload["num_prompts"]
        )
        if hasattr(benchmark, "phase") and benchmark.phase != phase:
            raise ValueError("Benchmark adapter and study phase/model must match.")
        if hasattr(benchmark, "concurrency") and benchmark.concurrency != concurrency:
            raise ValueError("Benchmark adapter and study concurrency must match.")
        if hasattr(benchmark, "workload") and benchmark.workload != workload:
            raise ValueError("Benchmark adapter and study workload must match.")
        choices = (
            self.expanded_choices(framework, choices)
            if space_kind == "expanded"
            else None
        )
        pruner = pruner or optuna.pruners.NopPruner()
        active_pruning = not isinstance(pruner, optuna.pruners.NopPruner)
        if active_pruning and (len(metrics) > 1 or constrained):
            raise ValueError(
                "Pruning is restricted to unconstrained scalar objectives."
            )
        use_checkpoints = active_pruning or checkpointed
        if use_checkpoints and (not benchmark.supports_checkpoints):
            raise ValueError("Real pruning needs checkpoints and clean cancellation.")
        if use_checkpoints and self.RUN_REAL_GPU:
            self.verify_pruning_workload(
                benchmark, phase, concurrency, framework, target, sampler_name, seed
            )
        study, default_budget = self.create_phase_a_study(
            sampler_name,
            framework,
            concurrency,
            seed,
            name,
            target,
            constrained,
            space_kind,
            pruner,
            phase,
            choices,
        )
        budget = default_budget if trials is None else trials
        if budget is None:
            budget = self.grid_size(concurrency, framework)
        menu = self.sampling_menu(framework, concurrency, space_kind, choices)
        from llm_optimizer.optimization import checkpoint_schedule

        schedule = checkpoint_schedule(workload["num_prompts"], min_completed_requests)
        contract = {
            "environment": getattr(benchmark, "environment", self.ENVIRONMENT),
            "framework": framework,
            "phase": phase,
            "target_model": self.PHASE_MODELS[phase],
            "concurrency": concurrency,
            "workload": {
                k: "inf" if isinstance(v, float) and math.isinf(v) else v
                for k, v in workload.items()
            },
            "space_kind": space_kind,
            "phase_a_space": self.SPACES[self.regime_key(concurrency)][framework],
            "expanded_choices": choices,
            "sampling_menu": menu,
            "target": target,
            "metrics": metrics,
            "directions": [d.upper() for d in directions],
            "constrained": constrained,
            "ttft_limit_ms": self.TTFT_LIMITS_MS[self.regime_key(concurrency)],
            "e2e_limit_ms": self.E2E_LIMITS_MS[self.regime_key(concurrency)],
            "sampler": sampler_name,
            "sampler_class": type(study.sampler).__name__,
            "sampler_semantics": (
                "Sobol numeric plus independent categorical/conditional"
            )
            if sampler_name == "qmc" and space_kind == "expanded"
            else "GP intersection plus independent conditional"
            if sampler_name == "gp" and space_kind == "expanded"
            else "lazy affine Cartesian traversal"
            if sampler_name == "grid" and space_kind == "expanded"
            else sampler_name,
            "sampler_settings": {
                "seed": seed,
                "n_startup_trials": self.SAMPLER_SETTINGS[
                    "startup_single" if concurrency == 1 else "startup_multi"
                ],
                "multivariate": sampler_name == "tpe_multivariate",
                "group": sampler_name == "tpe_multivariate"
                and space_kind == "expanded",
            },
            "pruner_class": type(pruner).__name__,
            "pruner_settings": pruner_settings or {},
            "budget_trials": budget,
            "budget_gpu_seconds": gpu_budget,
            "quick_demo": self.QUICK_DEMO,
            "grid_declared_points": math.prod(map(len, menu.values()))
            if sampler_name == "grid"
            else None,
        }
        if use_checkpoints:
            contract["checkpoint_requests"] = schedule
        previous = study.user_attrs.get("contract")
        if previous is not None:
            ignored = {"budget_trials", "budget_gpu_seconds", "quick_demo"}
            old = {k: v for k, v in previous.items() if k not in ignored}
            new = {k: v for k, v in contract.items() if k not in ignored}
            if json.dumps(old, sort_keys=True) != json.dumps(new, sort_keys=True):
                raise ValueError(
                    "Existing study has different experimental settings. "
                    "Use its original settings or create a new study."
                )
            contract = previous
        else:
            study.set_user_attr("contract", contract)

        gp_policy = None
        if sampler_name == "gp":
            from llm_optimizer.experiment.gp_policy import GPFailurePolicy

            gp_policy = GPFailurePolicy(
                study,
                self.RUN_DIR,
                {
                    "framework": framework,
                    "phase": phase,
                    "model": getattr(benchmark, "model", self.PHASE_MODELS[phase]),
                    "concurrency": concurrency,
                },
                fixed_server=getattr(benchmark, "fixed_server", {}),
            )
            study.set_user_attr("gp_failure_log", str(gp_policy.path))

        def objective(trial):
            config = self.suggest_config(
                trial, framework, concurrency, space_kind, choices
            )
            trial.set_user_attr("config", config)
            trial.set_user_attr("phase", phase)
            trial.set_user_attr("concurrency", concurrency)
            trial.set_user_attr(
                "config_id",
                self.config_fingerprint(framework, concurrency, config, phase),
            )
            trial.set_user_attr("gpu_seconds", 0.0)
            if gp_policy is not None:
                gp_policy.check(trial, config)
            started = time.perf_counter()
            try:
                if use_checkpoints and isinstance(benchmark, RepositoryBenchmark):

                    def report(m):
                        self.validate_measurement(m)
                        trial.set_user_attr("gpu_seconds", m.gpu_seconds)
                        trial.set_user_attr(
                            "request_accounting",
                            {
                                "completed": m.completed,
                                "failed": m.failed,
                                "total_requests": m.total_requests,
                            },
                        )
                        trial.report(getattr(m, metrics[0]), step=m.completed)
                        if (
                            active_pruning
                            and m.completed < schedule[-1]
                            and trial.should_prune()
                        ):
                            trial.set_user_attr(
                                "pruned_at_completed_requests", m.completed
                            )
                            trial.set_user_attr(
                                "pruned_at_fraction", m.completed / schedule[-1]
                            )
                            raise optuna.TrialPruned()

                    m = benchmark.measure(
                        framework,
                        config,
                        concurrency,
                        checkpoint_callback=report,
                        checkpoint_requests=schedule,
                    )
                    trial.set_user_attr("gpu_seconds", m.gpu_seconds)
                elif use_checkpoints:
                    with closing(
                        benchmark.stream(
                            framework, config, concurrency, checkpoint_requests=schedule
                        )
                    ) as trajectory:
                        for step, fraction, m, increment in trajectory:
                            self.validate_measurement(m)
                            trial.set_user_attr(
                                "gpu_seconds",
                                trial.user_attrs["gpu_seconds"] + increment,
                            )
                            trial.report(getattr(m, metrics[0]), step)
                            if active_pruning and fraction < 1 and trial.should_prune():
                                trial.set_user_attr("pruned_at_fraction", fraction)
                                raise optuna.TrialPruned()
                else:
                    m = benchmark.measure(framework, config, concurrency)
                    trial.set_user_attr("gpu_seconds", m.gpu_seconds)
                self.validate_measurement(m)
                if m.total_requests != workload["num_prompts"]:
                    raise self.measurement_failure(
                        "Final request count differs from trial workload", m
                    )
                self.record_measurement(trial, m, concurrency)
                scores = tuple(getattr(m, metric) for metric in metrics)
                return scores if len(scores) > 1 else scores[0]
            except optuna.TrialPruned as exc:
                trial.set_user_attr(
                    "gpu_seconds",
                    max(
                        trial.user_attrs["gpu_seconds"], getattr(exc, "gpu_seconds", 0)
                    ),
                )
                raise
            except BenchmarkFailure as exc:
                trial.set_user_attr("failure_reason", str(exc))
                trial.set_user_attr(
                    "request_accounting", getattr(exc, "request_accounting", {})
                )
                trial.set_user_attr(
                    "gpu_seconds",
                    max(
                        trial.user_attrs["gpu_seconds"],
                        getattr(exc, "gpu_seconds", 12.0),
                    ),
                )
                raise
            finally:
                trial.set_user_attr(
                    "orchestrator_wall_seconds", time.perf_counter() - started
                )

        def stop_at_budget(study, frozen_trial):
            if (
                gpu_budget is not None
                and sum(t.user_attrs.get("gpu_seconds", 0) for t in study.trials)
                >= gpu_budget
            ):
                study.stop()

        running = [t.number for t in study.trials if t.state == TrialState.RUNNING]
        if running:
            raise RuntimeError(
                f"Study {study.study_name} contains RUNNING trials "
                f"{running}. Verify that the previous worker has stopped "
                "and resolve abandoned trials before resuming."
            )
        attempts = sum(
            t.state != TrialState.WAITING and not t.user_attrs.get("gp_rejected")
            for t in study.trials
        )
        remaining = max(0, budget - attempts)
        print(
            f"{study.study_name}: {len(study.trials)} existing "
            f"trials, {remaining} remaining"
        )
        if remaining == 0:
            print(f"Study {study.study_name} has reached its trial budget.")
            return study
        wandb_logger = self.NotebookWandbLogger(study, contract)
        interrupted = True
        try:
            started = time.perf_counter()
            if gp_policy is not None:
                gp_policy.optimize(
                    objective,
                    remaining,
                    BenchmarkFailure,
                    [stop_at_budget, wandb_logger],
                )
            else:
                study.optimize(
                    objective,
                    n_trials=remaining,
                    n_jobs=1,
                    catch=(BenchmarkFailure,),
                    callbacks=[stop_at_budget, wandb_logger],
                )
            elapsed = time.perf_counter() - started
            previous_wall = study.user_attrs.get("optimization_wall_seconds", 0.0)
            total_wall = previous_wall + elapsed
            study.set_user_attr("optimization_wall_seconds", total_wall)
            study.set_user_attr(
                "host_search_and_storage_seconds",
                max(
                    0.0,
                    total_wall
                    - sum(
                        t.user_attrs.get("orchestrator_wall_seconds", 0)
                        for t in study.trials
                    ),
                ),
            )
            interrupted = False
        finally:
            wandb_logger.finish(study, interrupted=interrupted)
        return study

    def summarize(self, study):
        contract = study.user_attrs["contract"]
        valid = self.completed(study)
        scalar = len(study.directions) == 1
        minimizing = study.directions[0].name == "MINIMIZE"
        return {
            "study": study.study_name,
            "phase": contract["phase"],
            "target_model": contract["target_model"],
            "concurrency": contract["concurrency"],
            "num_prompts": contract["workload"]["num_prompts"],
            "target": contract["target"],
            "directions": contract["directions"],
            "attempts": len(study.trials),
            "states": dict(Counter(t.state.name for t in study.trials)),
            "unique_configs": len(
                {t.user_attrs.get("config_id") for t in study.trials}
            ),
            "allocated_gpu_seconds": sum(
                t.user_attrs.get("gpu_seconds", 0) for t in study.trials
            ),
            "host_search_and_storage_seconds": study.user_attrs[
                "host_search_and_storage_seconds"
            ],
            "best_observed_scalar": (min if minimizing else max)(t.value for t in valid)
            if valid and scalar
            else None,
            "grid_exhausted": study.sampler.is_exhausted(study)
            if isinstance(study.sampler, optuna.samplers.GridSampler)
            else None,
            "sampler_semantics": contract["sampler_semantics"],
        }

    def dominates(self, a, b, directions):
        directions = [d.name if hasattr(d, "name") else str(d) for d in directions]
        signed_a = [
            v if str(d).upper().endswith("MINIMIZE") else -v
            for v, d in zip(a, directions)
        ]
        signed_b = [
            v if str(d).upper().endswith("MINIMIZE") else -v
            for v, d in zip(b, directions)
        ]
        return all((x <= y for x, y in zip(signed_a, signed_b))) and any(
            (x < y for x, y in zip(signed_a, signed_b))
        )

    def pareto_front(self, trials, directions=("maximize", "minimize")):
        return [
            a
            for a in trials
            if not any(self.dominates(b.values, a.values, directions) for b in trials)
        ]

    def choose_single(self, study, feasible_only=False):
        if len(study.directions) != 1:
            raise ValueError("Use a Pareto front, not a single winner")
        candidates = self.completed(study, feasible_only)
        return (
            (min if study.directions[0].name == "MINIMIZE" else max)(
                candidates, key=lambda t: t.value
            )
            if candidates
            else None
        )

    def convergence(self, study, feasible_only=False):
        if len(study.directions) > 1:
            raise ValueError("Scalar convergence cannot summarize a Pareto vector")
        minimizing = study.directions[0].name == "MINIMIZE"
        best = float("inf") if minimizing else -float("inf")
        attempts, costs, values = ([], [], [])
        spent = 0.0
        for trial in study.trials:
            spent += trial.user_attrs.get("gpu_seconds", 0)
            if trial.state == TrialState.COMPLETE and (
                not feasible_only or trial.user_attrs.get("slo_feasible")
            ):
                best = min(best, trial.value) if minimizing else max(best, trial.value)
            if np.isfinite(best):
                attempts.append(trial.number + 1)
                costs.append(spent)
                values.append(best)
        return (np.array(attempts), np.array(costs), np.array(values))

    def coverage_report(self, study):
        counts = {}
        for trial in study.trials:
            for name, value in trial.params.items():
                counts.setdefault(name, Counter())[str(value)] += 1
        expected = study.user_attrs["contract"]["sampling_menu"]
        return {
            "study": study.study_name,
            "attempted_choice_counts": {k: dict(v) for k, v in counts.items()},
            "unvisited_choices": {
                k: [v for v in choices if str(v) not in counts.get(k, {})]
                for k, choices in expected.items()
            },
            "note": "Counts include failures; absent branches are not engine settings",
        }

    def serialize_engine_config(self, framework, concurrency, config):
        import json

        import yaml

        native = dict(config)
        if framework == "trtllm":
            if "config" in native:
                raise ValueError(
                    "Merge a fixed TRT YAML explicitly; do not override it silently."
                )
            nested = {
                key: native.pop(key)
                for key in [
                    "kv_cache_config",
                    "cuda_graph_config",
                    "scheduler_config",
                    "attn_backend",
                    "enable_chunked_prefill",
                ]
                if key in native
            }
            if nested:
                path = self.RUN_DIR / (
                    "trt-config-"
                    + self.config_fingerprint(framework, concurrency, config)
                    + ".yaml"
                )
                path.write_text(yaml.safe_dump(nested, sort_keys=True))
                native["config"] = str(path)
        for key, value in list(native.items()):
            if framework == "vllm" and key in {
                "async_scheduling",
                "enable_prefix_caching",
            }:
                del native[key]
                native[key if value else "no_" + key] = True
                continue
            if framework == "sglang" and key in {
                "disable_radix_cache",
                "enable_mixed_chunk",
                "disable_overlap_schedule",
            }:
                if value is False:
                    del native[key]
                continue
            if isinstance(value, (dict, list)):
                native[key] = json.dumps(value, sort_keys=True)
                continue
            if value is False:
                raise BenchmarkFailure(
                    f"False boolean for {key} needs explicit framework handling."
                )
            if value is None:
                raise BenchmarkFailure(f"Unsupported top-level null setting: {key}")
        return native

    def cli_dry_run_command(self, framework, concurrency, target="e2e", phase="A"):
        _, metrics, _ = self.objective_spec(target, concurrency)
        if len(metrics) != 1:
            raise ValueError("The repository CLI supports scalar dry runs only")
        space = self.SPACES[self.regime_key(concurrency)][framework]
        initial = {k: v[len(v) // 2] for k, v in space.items()}
        native = self.serialize_engine_config(
            framework,
            concurrency,
            self.render_engine_config(framework, concurrency, initial),
        )
        server = ";".join(
            (
                f"{k}={repr(v)}" if isinstance(v, str) else f"{k}={v}"
                for k, v in native.items()
            )
        )
        client = ";".join(
            (f"{k}={v}" for k, v in self.workload_for(concurrency, phase).items())
        )
        return [
            "llm-optimizer",
            "--framework",
            framework,
            "--model",
            "REPLACE_WITH_PINNED_MODEL_SNAPSHOT",
            "--gpus",
            "1",
            "--server-args",
            server,
            "--client-args",
            client,
            "--search-strategy",
            "optuna",
            "--optuna-sampler",
            "TPESampler",
            "--optuna-objective",
            metrics[0],
            "--optuna-trials",
            "1",
            "--output-json",
            str(self.RUN_DIR / f"{phase}-{framework}-{target}-dryrun.json"),
            "--dry-run",
        ]

    def benchmark_for(
        self,
        phase,
        framework,
        concurrency,
        run_label,
        num_prompts=None,
        workload_seed=None,
        repeat=0,
    ):
        if not self.RUN_REAL_GPU:
            return self.Simulator(
                seed=self.SIMULATION_SEED + repeat
                if workload_seed is None
                else self.VALIDATION_NOISE_SEED + repeat,
                phase=phase,
                num_prompts=num_prompts,
                workload_seed=workload_seed,
            )
        if framework not in self.REAL_FRAMEWORKS:
            raise ValueError("Framework is not enabled in this kernel")
        environment = self.REAL_PREFLIGHT[phase, framework]
        if phase == "B":
            if (
                environment.get("quality_verified") is not True
                or environment.get("expanded_choices") is None
            ):
                raise ValueError(
                    "Phase B needs support and numerical-quality preflight"
                )
            self.expanded_choices(framework, environment["expanded_choices"])
        return self.RepositoryBenchmark(
            framework,
            self.MODEL_SNAPSHOTS[phase],
            environment["gpu_type"],
            concurrency,
            environment=environment,
            gpu_count=self.GPU_COUNT,
            rest_seconds=self.REST_SECONDS,
            fixed_server=self.FIXED_SERVER.get(framework),
            phase=phase,
            num_prompts=num_prompts,
            workload_seed=workload_seed,
            run_label=run_label,
        )

    def run_objective_group(self, phase, target, concurrencies):
        results = {}
        frameworks = self.REAL_FRAMEWORKS if self.RUN_REAL_GPU else self.FRAMEWORKS
        samplers = (
            self.PARETO_SAMPLERS if target.startswith("pareto") else self.SAMPLER_NAMES
        )
        group = (phase, target, tuple(concurrencies))
        self.GROUP_RUNS[group] += 1
        for concurrency in concurrencies:
            if target not in self.objectives_for(concurrency):
                raise ValueError(
                    "Objective is not part of the requested concurrency task"
                )
            for framework in frameworks:
                choices = (
                    self.REAL_PREFLIGHT[phase, framework].get("expanded_choices")
                    if self.RUN_REAL_GPU and phase == "B"
                    else None
                )
                for arm in samplers:
                    if arm == "gp" and (not self.ARM_AVAILABILITY["gp"]["enabled"]):
                        self.SKIPPED_ARMS.append(
                            {
                                "phase": phase,
                                "framework": framework,
                                "concurrency": concurrency,
                                "target": target,
                                "sampler": arm,
                                "reason": self.ARM_AVAILABILITY[arm]["reason"],
                            }
                        )
                        continue
                    for seed in (
                        self.ACTIVE_SEEDS[:1] if arm == "grid" else self.ACTIVE_SEEDS
                    ):
                        label = (
                            f"{phase}-{framework}-c{concurrency}-{target}-{arm}"
                            f"-s{seed}-r{self.GROUP_RUNS[group]}"
                        )
                        benchmark = self.benchmark_for(
                            phase, framework, concurrency, label
                        )
                        study = self.run_study(
                            label,
                            framework,
                            arm,
                            concurrency=concurrency,
                            phase=phase,
                            seed=seed,
                            target=target,
                            space_kind="baseline" if phase == "A" else "expanded",
                            choices=choices,
                            benchmark=benchmark,
                        )
                        key = (phase, concurrency, framework, target, arm, seed)
                        results[key] = study
                        self.STUDIES[key] = study
        display([self.summarize(study) for study in results.values()])
        display([self.coverage_report(study) for study in results.values()])
        if self.SKIPPED_ARMS:
            display({"skipped_arms": self.SKIPPED_ARMS})
        return results

    def run_constrained_objectives(self, phase="B", sampler_name="tpe_multivariate"):
        self.GROUP_RUNS["constrained", phase] += 1
        results = {}
        for c in self.CONCURRENCIES:
            for framework in (
                self.REAL_FRAMEWORKS if self.RUN_REAL_GPU else self.FRAMEWORKS
            ):
                for target in self.objectives_for(c):
                    label = (
                        f"constrained-{phase}-{framework}-c{c}-{target}"
                        f"-r{self.GROUP_RUNS['constrained', phase]}"
                    )
                    benchmark = self.benchmark_for(phase, framework, c, label)
                    choices = (
                        self.REAL_PREFLIGHT[phase, framework].get("expanded_choices")
                        if self.RUN_REAL_GPU and phase == "B"
                        else None
                    )
                    study = self.run_study(
                        label,
                        framework,
                        sampler_name,
                        concurrency=c,
                        phase=phase,
                        target=target,
                        constrained=True,
                        benchmark=benchmark,
                        choices=choices,
                        space_kind="baseline" if phase == "A" else "expanded",
                    )
                    results[phase, c, framework, target, "constrained"] = study
        self.CONSTRAINED_STUDIES.update(results)
        display([self.summarize(s) for s in results.values()])
        return results

    def scalar_tasks(self):
        return [
            (phase, c, fw, target)
            for phase in self.PHASE_MODELS
            for c in self.CONCURRENCIES
            for fw in (self.REAL_FRAMEWORKS if self.RUN_REAL_GPU else self.FRAMEWORKS)
            for target in self.objectives_for(c)
            if not target.startswith("pareto")
        ]

    def calibrate_pruning(self, phase, concurrency, framework, target, samples=32):
        target, metrics, directions = self.objective_spec(target, concurrency)
        if len(metrics) != 1:
            raise ValueError("Pareto pruning is unsupported")
        pilot = optuna.create_study(
            sampler=optuna.samplers.RandomSampler(seed=self.CALIBRATION_SAMPLER_SEED)
        )
        benchmark = self.Simulator(seed=self.CALIBRATION_NOISE_SEED, phase=phase)
        trajectories = []
        for _ in range(samples):
            trial = pilot.ask()
            cfg = self.suggest_config(
                trial,
                framework,
                concurrency,
                "baseline" if phase == "A" else "expanded",
            )
            try:
                rows = list(benchmark.stream(framework, cfg, concurrency))
            except BenchmarkFailure:
                pilot.tell(trial, state=TrialState.FAIL)
                continue
            scores = [getattr(row[2], metrics[0]) for row in rows]
            pilot.tell(trial, scores[-1])
            trajectories.append({"scores": scores, "costs": [row[3] for row in rows]})
        if len(trajectories) < 3:
            raise ValueError("Insufficient valid calibration trajectories")
        scores = np.array([r["scores"] for r in trajectories])
        minimizing = directions[0] == "minimize"
        order = np.argsort(scores[:, -1])
        top_count = max(1, math.ceil(len(scores) * 0.1))
        top = set(order[:top_count] if minimizing else order[-top_count:])
        history, stopped = ([], set())
        for i in np.random.default_rng(self.CALIBRATION_SAMPLER_SEED).permutation(
            len(scores)
        ):
            worse = len(history) >= 8 and (
                scores[i, 0] > np.median(history)
                if minimizing
                else scores[i, 0] < np.median(history)
            )
            if worse:
                stopped.add(i)
            else:
                history.append(scores[i, 0])
        total = sum(sum(r["costs"]) for r in trajectories)
        replay = sum(
            (
                r["costs"][0] if i in stopped else sum(r["costs"])
                for i, r in enumerate(trajectories)
            )
        )
        correlations = [
            float(spearmanr(scores[:, step], scores[:, -1]).statistic)
            for step in [0, 1]
        ]
        return {
            "phase": phase,
            "model": self.PHASE_MODELS[phase],
            "framework": framework,
            "concurrency": concurrency,
            "target": target,
            "metric": metrics[0],
            "direction": directions[0],
            "workload": self.workload_for(concurrency, phase),
            "valid_trajectories": len(scores),
            "rank_correlations": [v if np.isfinite(v) else None for v in correlations],
            "false_prune_rate_among_top_10pct": len(top & stopped) / len(top),
            "replayed_time_saved_fraction": 1 - replay / total,
            "threshold": float(np.percentile(scores[:, 0], 90 if minimizing else 10)),
        }

    def pruner_menu(
        self, direction, threshold=None, max_requests=None, min_completed_requests=1
    ):
        from llm_optimizer.optimization import checkpoint_schedule

        minimizing = str(direction).lower() == "minimize"
        max_requests = (
            self.PHASE_REQUESTS["A"] if max_requests is None else max_requests
        )
        schedule = checkpoint_schedule(max_requests, min_completed_requests)
        first = schedule[0]
        menu = {
            "nop": (optuna.pruners.NopPruner, {}),
            "median": (
                optuna.pruners.MedianPruner,
                {"n_startup_trials": 10, "n_warmup_steps": first, "n_min_trials": 3},
            ),
            "percentile": (
                optuna.pruners.PercentilePruner,
                {
                    "percentile": 75 if minimizing else 25,
                    "n_startup_trials": 10,
                    "n_warmup_steps": first,
                    "n_min_trials": 3,
                },
            ),
            "halving": (
                optuna.pruners.SuccessiveHalvingPruner,
                {"min_resource": first, "reduction_factor": 2},
            ),
            "hyperband": (
                optuna.pruners.HyperbandPruner,
                {
                    "min_resource": first,
                    "max_resource": max_requests,
                    "reduction_factor": 2,
                },
            ),
        }
        if threshold is not None:
            menu["threshold"] = (
                optuna.pruners.ThresholdPruner,
                {
                    "upper" if minimizing else "lower": threshold,
                    "n_warmup_steps": first,
                },
            )
        return menu

    def run_pruning_objective(self, target):
        results = {}
        self.GROUP_RUNS["pruning", target] += 1
        for phase, c, fw, task_target in self.scalar_tasks():
            if task_target != target:
                continue
            threshold = None
            if not self.RUN_REAL_GPU:
                if (phase, c, fw, target) not in self.calibration:
                    raise ValueError("Calibrate this task before pruning")
                cal = self.calibration[phase, c, fw, target]
                if cal["workload"] != self.workload_for(c, phase):
                    raise ValueError("Workload changed after calibration")
                threshold = cal["threshold"]
            direction = self.objective_spec(target, c)[2][0]
            for name, (pruner_type, settings) in self.pruner_menu(
                direction,
                threshold,
                max_requests=self.workload_for(c, phase)["num_prompts"],
            ).items():
                key = (phase, c, fw, target, name)
                study = self.run_study(
                    f"pruning-{phase}-{fw}-c{c}-{target}-{name}"
                    f"-r{self.GROUP_RUNS['pruning', target]}",
                    fw,
                    concurrency=c,
                    phase=phase,
                    target=target,
                    space_kind="baseline" if phase == "A" else "expanded",
                    pruner=pruner_type(**settings),
                    pruner_settings=settings,
                    checkpointed=True,
                    trials=12 if self.QUICK_DEMO else 32,
                    benchmark=self.benchmark_for(
                        phase, fw, c, f"pruning-{target}-{name}"
                    ),
                    choices=self.REAL_PREFLIGHT[phase, fw].get("expanded_choices")
                    if self.RUN_REAL_GPU and phase == "B"
                    else None,
                )
                results[key] = study
        display([self.summarize(s) for s in results.values()])
        return results

    def task_identity(self, contract):
        payload = {
            k: contract[k]
            for k in [
                "phase",
                "target_model",
                "framework",
                "concurrency",
                "target",
                "metrics",
                "directions",
                "workload",
                "environment",
                "constrained",
                "ttft_limit_ms",
                "e2e_limit_ms",
            ]
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[
            :16
        ]

    def shortlist(self, studies, k=3):
        candidates = {}
        for study in studies:
            contract = study.user_attrs["contract"]
            task_id = self.task_identity(contract)
            valid = self.completed(study, feasible_only=contract["constrained"])
            if len(study.directions) > 1:
                selected = self.pareto_front(valid, study.directions)
            else:
                selected = sorted(
                    valid,
                    key=lambda t: t.value,
                    reverse=study.directions[0].name == "MAXIMIZE",
                )[:k]
            for trial in selected:
                cfg = trial.user_attrs["config"]
                cfg_id = self.config_fingerprint(
                    contract["framework"],
                    contract["concurrency"],
                    cfg,
                    contract["phase"],
                )
                key = task_id + ":" + cfg_id
                candidates[key] = {
                    "task_id": task_id,
                    "config_id": cfg_id,
                    "config": cfg,
                    "contract": contract,
                }
        return candidates

    def repeat_summary(self, values):
        values = np.asarray(values, dtype=float)
        if len(values) == 0:
            return {
                "mean": None,
                "median": None,
                "sd": None,
                "mean_95pct_t_interval": None,
                "replicates": [],
            }
        mean = float(values.mean())
        sd = float(values.std(ddof=1)) if len(values) > 1 else None
        half = (
            float(student_t.ppf(0.975, len(values) - 1) * sd / np.sqrt(len(values)))
            if sd is not None
            else None
        )
        return {
            "mean": mean,
            "median": float(np.median(values)),
            "sd": sd,
            "mean_95pct_t_interval": [mean - half, mean + half]
            if half is not None
            else None,
            "replicates": values.tolist(),
        }

    def validate_candidates(self, candidates, repeats=None, benchmark_factory=None):
        repeats = self.VALIDATION_REPEATS if repeats is None else repeats
        if not isinstance(repeats, int) or repeats < 1:
            raise ValueError("Validation repeats must be positive")
        rows = []
        for key, candidate in candidates.items():
            contract = candidate["contract"]
            phase, fw, c = (
                contract["phase"],
                contract["framework"],
                contract["concurrency"],
            )
            count = self.VALIDATION_REQUESTS[phase]
            measurements, failures, spent = ([], [], 0.0)
            for repeat in range(repeats):
                label = f"validation-{key.replace(':', '-')}-repeat{repeat}"
                seed = self.VALIDATION_SEED + repeat
                benchmark = (
                    benchmark_factory(fw, c, repeat, phase, count, seed)
                    if benchmark_factory
                    else self.benchmark_for(
                        phase,
                        fw,
                        c,
                        label,
                        num_prompts=count,
                        workload_seed=seed,
                        repeat=repeat,
                    )
                )
                if hasattr(benchmark, "phase") and benchmark.phase != phase:
                    raise ValueError(
                        "Validation adapter and candidate phase/model differ"
                    )
                source = contract["environment"].get("measurement", "SYNTHETIC")
                if (
                    source == "MEASURED"
                    and getattr(benchmark, "environment", {}).get("measurement")
                    != "MEASURED"
                ):
                    raise ValueError(
                        "Measured candidates cannot be validated synthetically"
                    )
                if source == "MEASURED":
                    for field in [
                        "framework_version",
                        "cuda_version",
                        "driver_version",
                        "model_revision",
                        "model",
                        "gpu_type",
                        "gpu_count",
                    ]:
                        if benchmark.environment.get(field) != contract[
                            "environment"
                        ].get(field):
                            raise ValueError(f"Validation environment changed: {field}")
                try:
                    m = self.validate_measurement(
                        benchmark.measure(fw, candidate["config"], c)
                    )
                    if m.total_requests != count:
                        raise self.measurement_failure(
                            "Validation request count changed", m
                        )
                    measurements.append(m)
                    spent += m.gpu_seconds
                except BenchmarkFailure as exc:
                    failures.append(
                        {
                            "repeat": repeat,
                            "reason": str(exc),
                            "request_accounting": getattr(
                                exc, "request_accounting", {}
                            ),
                        }
                    )
                    spent += getattr(exc, "gpu_seconds", 12.0)
            stats = {
                metric: self.repeat_summary([getattr(m, metric) for m in measurements])
                for metric in ["output_throughput", "p95_ttft_ms", "p95_e2e_latency_ms"]
            }
            complete = len(measurements) == repeats
            feasible = complete and all(
                m.p95_ttft_ms <= contract["ttft_limit_ms"]
                and m.p95_e2e_latency_ms <= contract["e2e_limit_ms"]
                for m in measurements
            )
            rows.append(
                {
                    "candidate_key": key,
                    "task_id": candidate["task_id"],
                    "config_id": candidate["config_id"],
                    "phase": phase,
                    "target_model": self.PHASE_MODELS[phase],
                    "framework": fw,
                    "concurrency": c,
                    "target": contract["target"],
                    "metrics": contract["metrics"],
                    "directions": contract["directions"],
                    "constrained": contract["constrained"],
                    "config": candidate["config"],
                    "attempts": repeats,
                    "num_prompts_per_repeat": count,
                    "successful_repeats": len(measurements),
                    "failures": failures,
                    "gpu_seconds": spent,
                    "statistics": stats,
                    "validation_complete": complete,
                    "validation_feasible": feasible,
                    "validated_values": [
                        stats[metric]["mean"] for metric in contract["metrics"]
                    ],
                    "measurement_source": measurements[0].source
                    if measurements
                    else None,
                }
            )
        return rows

    def validation_recommendations(self, rows, require_slo=False):
        recommendations = {}
        for task_id in sorted({row["task_id"] for row in rows}):
            eligible = [
                row
                for row in rows
                if row["task_id"] == task_id
                and row["validation_complete"]
                and (
                    not (require_slo or row["constrained"])
                    or row["validation_feasible"]
                )
            ]
            if not eligible:
                recommendations[task_id] = None
                continue
            directions = eligible[0]["directions"]
            if len(directions) > 1:
                recommendations[task_id] = [
                    row
                    for row in eligible
                    if not any(
                        self.dominates(
                            other["validated_values"],
                            row["validated_values"],
                            directions,
                        )
                        for other in eligible
                    )
                ]
            else:
                recommendations[task_id] = (
                    min if directions[0] == "MINIMIZE" else max
                )(eligible, key=lambda row: row["validated_values"][0])
        return recommendations

    def artifact_value(self, value):
        if isinstance(value, dict):
            return {k: self.artifact_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.artifact_value(v) for v in value]
        if isinstance(value, float) and math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value

    def verify_pruning_workload(
        self,
        benchmark,
        phase,
        concurrency,
        framework,
        target,
        sampler_name=None,
        seed=None,
    ):
        """Check the previous GPU phase contract before starting a server."""
        if not self.RUN_REAL_GPU:
            return

        def matches(contract):
            return (
                contract.get("phase") == phase
                and contract.get("concurrency") == concurrency
                and (contract.get("framework") == framework)
                and (contract.get("target") == target)
                and (contract.get("environment", {}).get("measurement") == "MEASURED")
                and (not contract.get("checkpoint_requests"))
                and (sampler_name is None or contract.get("sampler") == sampler_name)
                and (
                    seed is None
                    or contract.get("sampler_settings", {}).get("seed") == seed
                )
            )

        previous = [
            s.user_attrs["contract"]
            for s in getattr(self, "STUDIES", {}).values()
            if matches(s.user_attrs.get("contract", {}))
        ]
        if not previous:
            summaries = optuna.get_all_study_summaries(
                storage=self.STORAGE, include_best_trial=False
            )
            previous = [
                summary.user_attrs["contract"]
                for summary in reversed(summaries)
                if matches(summary.user_attrs.get("contract", {}))
            ][:1]
        if not previous:
            return
        workload = {
            k: "inf" if isinstance(v, float) and math.isinf(v) else v
            for k, v in benchmark.workload.items()
        }
        choices = (
            self.REAL_PREFLIGHT[phase, framework].get("expanded_choices")
            if phase == "B"
            else None
        )
        menu = self.sampling_menu(
            framework, concurrency, "baseline" if phase == "A" else "expanded", choices
        )
        for contract in previous:
            if (
                contract["workload"] != workload
                or contract["environment"].get("requests_sha256")
                != benchmark.environment["requests_sha256"]
            ):
                raise ValueError(
                    "Pruning requests differ from the previous phase study"
                )
            if contract.get("sampling_menu") != menu:
                raise ValueError(
                    "Pruning search space differs from the previous phase study"
                )
            for field in (
                "framework_version",
                "cuda_version",
                "driver_version",
                "model_revision",
                "model",
                "gpu_type",
                "gpu_count",
            ):
                if contract["environment"].get(field) != benchmark.environment.get(
                    field
                ):
                    raise ValueError(f"Pruning environment changed: {field}")
