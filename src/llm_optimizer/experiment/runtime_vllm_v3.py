"""Frozen v3 execution adapters; legacy launch and benchmark modules stay unchanged.

The lifecycle functions are copied from the current shared implementation so v3
can substitute its benchmark client without global monkeypatches. Future shared
lifecycle changes must be reviewed explicitly before porting them here.
"""

import json
import pathlib
import shlex
import sys
import time
import typing as t
from dataclasses import replace

import llm_optimizer.predefined as predefined
from llm_optimizer.cli import construct_result_config, extract_token_lengths, logger
from llm_optimizer.optimization import satisfies_constraints
from llm_optimizer.server_utils import (
    _close_server_log,
    get_server_env,
    start_server,
    terminate_process_top_down,
)
from llm_optimizer.trial_telemetry import TrialTelemetry
from llm_optimizer.utils import InfinityToNullEncoder

from .lab import BenchmarkFailure, Measurement, RepositoryBenchmark, optuna


def run_single_configuration(
    benchmark_settings: dict[str, t.Any],
    *,
    framework: t.Optional[str],
    model: t.Optional[str],
    server_cmd: str,
    managed_server: bool,
    gpus: int,
    gpu_type: str,
    host: str,
    port: t.Optional[int],
    ready_url: str,
    mute_server: bool,
    constraints_for_viz: list,
    output_jsonl_path: t.Optional[pathlib.Path] = None,
    output_file_path: t.Optional[pathlib.Path] = None,
    objective: t.Optional[str] = None,
    trial_number: t.Optional[int] = None,
    study_signature: t.Optional[str] = None,
    parsed_constraints: t.Optional[list] = None,
    workload_manifest: t.Optional[dict[str, t.Any]] = None,
    workload_metadata: t.Optional[dict[str, t.Any]] = None,
    checkpoint_requests: t.Optional[list[int]] = None,
    checkpoint_callback: t.Optional[t.Callable] = None,
    telemetry: t.Optional[TrialTelemetry] = None,
) -> tuple[dict[str, t.Any], t.Optional[t.Union[float, list[float]]]]:
    """Run and persist one configuration, always cleaning up its server."""
    from llm_optimizer.optimization import extract_objective

    from . import telemetry_vllm_v3 as bench_client

    client_params = benchmark_settings["client_args"]
    server_params = benchmark_settings["server_args"]
    full_server_cmd = f"{server_cmd} {' '.join(benchmark_settings['server_cmd_args'])}"
    server_process = None
    output_path = output_jsonl_path or output_file_path
    telemetry = telemetry or TrialTelemetry(
        pathlib.Path(output_path).parent
        / (pathlib.Path(output_path).stem + "-attempts")
        if output_path
        else pathlib.Path("benchmark-attempts"),
        framework=framework,
        model=model,
        trial_number=trial_number,
        study_signature=study_signature,
        command=full_server_cmd,
        environment=workload_metadata or {},
    )
    telemetry.data.update(
        framework=framework,
        model=model,
        trial_number=trial_number,
        study_signature=study_signature,
        command=full_server_cmd,
    )
    error = None
    result = None
    try:
        telemetry.data["config"] = construct_result_config(
            benchmark_settings, framework
        )
        server_env = (
            get_server_env(framework, gpus, server_params) if managed_server else {}
        )
        telemetry.data["server_environment"] = server_env
        telemetry.save()
        with telemetry.activate():
            server_process = start_server(
                full_server_cmd, server_env, ready_url, mute_server
            )
        backend = (
            predefined.get_framework_spec(framework).benchmark_backend
            if framework
            else "vllm"
        )
        benchmark_args = {
            "backend": backend,
            "model": model,
            "host": host,
            "port": port,
            "dataset_name": "sharegpt",
            "num_prompts": 1000,
            "request_rate": float("inf"),
            "seed": 1,
        }
        benchmark_args.update(client_params)
        if workload_manifest is not None:
            benchmark_args["_request_manifest"] = workload_manifest

        def report_checkpoint(values):
            telemetry.data["request_accounting"] = {
                key: values[key]
                for key in ("completed", "total_requests", "failed")
                if key in values
            }
            if checkpoint_callback is not None:
                checkpoint_callback(values)

        if checkpoint_callback is not None:
            benchmark_args["_checkpoint_requests"] = checkpoint_requests
            benchmark_args["_checkpoint_callback"] = report_checkpoint
        benchmark_args["_telemetry"] = telemetry
        benchmark_result = bench_client.run_benchmark(benchmark_args)
        if not isinstance(benchmark_result, dict):
            raise RuntimeError("Benchmark did not return a result")

        telemetry.data["request_accounting"] = {
            key: benchmark_result[key]
            for key in ("completed", "total_requests", "failed")
            if key in benchmark_result
        }
        telemetry.data["request_errors"] = benchmark_result.get("request_errors", [])
        telemetry.save()
        feasible = satisfies_constraints(benchmark_result, parsed_constraints or [])
        input_len, output_len = extract_token_lengths(client_params)
        result: dict[str, t.Any] = {
            "config": construct_result_config(benchmark_settings, framework),
            "results": benchmark_result,
            "cmd": full_server_cmd,
            "constraints": constraints_for_viz,
            "metadata": {
                "gpu_type": gpu_type,
                "gpu_count": gpus,
                "model_tag": model,
                "input_tokens": input_len,
                "output_tokens": output_len,
                **(workload_metadata or {}),
                "execution": telemetry.data,
            },
        }
        if trial_number is not None:
            result["optimization"] = {
                "trial_number": trial_number,
                "objective_value": None,
                "slo_feasible": feasible,
                "study_signature": study_signature,
            }
        objective_error = None
        try:
            objective_value = (
                extract_objective(benchmark_result, objective) if objective else None
            )
        except ValueError as exc:
            objective_value = None
            objective_error = str(exc)
            result.setdefault("optimization", {})["failure_reason"] = objective_error
        if trial_number is not None:
            result["optimization"]["objective_value"] = objective_value
        if objective_error and objective is not None:
            raise ValueError(objective_error)
    except BaseException as exc:
        error = exc
        raise
    finally:
        try:
            if server_process:
                with telemetry.activate(), telemetry.phase("cleanup"):
                    try:
                        terminate_process_top_down(server_process)
                    finally:
                        _close_server_log(server_process)
        except BaseException as cleanup_error:
            telemetry.data["cleanup_error"] = str(cleanup_error)
            if error is not None:
                telemetry.data["original_error"] = f"{type(error).__name__}: {error}"
            error = cleanup_error
            raise
        finally:
            telemetry.finish(error)
            if error is not None:
                error.execution = telemetry.data
    # Successful metric rows include cleanup; exceptional attempts live in sidecars.
    try:
        if output_jsonl_path:
            with output_jsonl_path.open("a") as stream:
                stream.write(json.dumps(result, cls=InfinityToNullEncoder) + "\n")
            logger.info(f"Appended result to {output_jsonl_path}")
        elif output_file_path:
            with output_file_path.open("w") as stream:
                json.dump(result, stream, indent=2, cls=InfinityToNullEncoder)
            logger.info(f"Benchmark results saved to {output_file_path}")
    except BaseException as exc:
        telemetry.finish(exc)
        exc.execution = telemetry.data
        raise
    return result, objective_value


class RepositoryBenchmarkV3(RepositoryBenchmark):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spec = replace(
            self.spec,
            server_command=(
                shlex.quote(sys.executable) + " -m vllm.entrypoints.cli.main serve "
                "{model} --host {host} --port {port}"
            ),
        )
        self.environment["vllm_v3_telemetry"] = self._context.telemetry_options

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
        from llm_optimizer.cli import construct_benchmark_settings
        from llm_optimizer.predefined import CLIENT_CONFIGS, SERVER_CONFIGS

        from .vllm_policy_v3 import V3_SERVER_CONFIGS

        SERVER_CONFIGS = {**SERVER_CONFIGS, "vllm": V3_SERVER_CONFIGS}

        if framework != self.framework or concurrency != self.concurrency:
            raise ValueError("Benchmark framework/concurrency differs from the study.")

        attempts = []

        def execute():
            started = time.perf_counter()
            accounting = {}
            telemetry = TrialTelemetry(
                self.output_path.parent / (self.output_path.stem + "-attempts"),
                framework=framework,
                model=self.model,
                environment=self.environment,
                sampled_config=config,
                workload_preparation_seconds=self.workload_preparation_seconds,
                workload_preparation_id=self.workload_preparation_id,
            )
            attempts.append(telemetry)
            error = None
            try:
                merged = {**self.fixed_server, **config}
                native = self._context.serialize_engine_config(
                    framework, concurrency, merged
                )
                args = []
                for scope, values, registry in [
                    (ArgScope.SERVER, native, SERVER_CONFIGS[framework]),
                    (ArgScope.CLIENT, self.workload, CLIENT_CONFIGS),
                ]:
                    for key, value in values.items():
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
                    telemetry=telemetry,
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
                    with telemetry.rest():
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
                        execution=telemetry.data,
                    )
                )
            except BaseException as exc:
                error = exc
                exc.execution = telemetry.data
                if not isinstance(exc, Exception):
                    exc.gpu_seconds = (time.perf_counter() - started) * self.gpu_count
                    exc.request_accounting = accounting
                    raise
                from llm_optimizer.exceptions import ServerCleanupError

                if isinstance(exc, (ServerCleanupError, optuna.TrialPruned)):
                    exc.gpu_seconds = (time.perf_counter() - started) * self.gpu_count
                    exc.request_accounting = accounting
                    raise
                failure = BenchmarkFailure(f"{type(exc).__name__}: {exc}")
                failure.execution = telemetry.data
                failure.request_accounting = getattr(
                    exc, "request_accounting", accounting
                )
                failure.gpu_seconds = (time.perf_counter() - started) * self.gpu_count
                raise failure from exc
            finally:
                telemetry.data["request_accounting"] = accounting or telemetry.data.get(
                    "request_accounting", {}
                )
                telemetry.finish(error)
                telemetry.data["timings_s"]["allocation_wall"] = (
                    time.perf_counter() - started
                )
                telemetry.data["gpu_seconds"] = (
                    telemetry.data["timings_s"]["allocation_wall"] * self.gpu_count
                )
                telemetry.save()

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(execute).result()
        except BaseException as exc:
            # A keyboard interrupt in the caller is not delivered to the worker.
            # Preserve the existing executor shutdown/wait behavior, then mark
            # the attempt interrupted after its server has been cleaned up.
            if attempts and not getattr(exc, "execution", None):
                telemetry = attempts[0]
                telemetry.finish(exc)
                exc.execution = telemetry.data
                exc.gpu_seconds = telemetry.data.get("gpu_seconds", 0)
                exc.request_accounting = telemetry.data.get("request_accounting", {})
            raise
