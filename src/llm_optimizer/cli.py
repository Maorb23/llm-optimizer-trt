import importlib
import json
import pathlib
import time
import typing as t

import click

import llm_optimizer.args as lo_args
import llm_optimizer.predefined as predefined
from llm_optimizer.cli_utils import (
    collect_gpu_configuration,
    collect_interactive_parameters,
    detect_gpu_type,
    get_gpu_count,
    normalize_gpu_choice,
)
from llm_optimizer.logging import get_logger, setup_logging
from llm_optimizer.optimization import IntRange, parse_int_ranges, satisfies_constraints
from llm_optimizer.performance import (
    PerformanceEstimationParams,
    display_performance_estimation_results,
    run_performance_estimation,
)
from llm_optimizer.predefined.gpus import list_available_gpus_with_lowercase
from llm_optimizer.result_utils import normalize_config
from llm_optimizer.server_utils import (
    get_server_env,
    start_server,
    terminate_process_top_down,
)
from llm_optimizer.utils import InfinityToNullEncoder

setup_logging()
logger = get_logger("main")

PREDEFINED_FRAMEWORKS = list(predefined.SERVER_CONFIGS.keys())


def construct_benchmark_settings(
    combo: list[lo_args.BaseArg], framework: t.Optional[str] = None
) -> dict[str, t.Any]:
    client_args = [arg for arg in combo if arg.scope == lo_args.ArgScope.CLIENT]
    server_args = [arg for arg in combo if arg.scope == lo_args.ArgScope.SERVER]
    client_kv_pairs = lo_args.get_all_kv_pairs(client_args)
    server_cmd_args = predefined.get_server_cmd_args(framework, server_args)
    server_kv_pairs = lo_args.get_all_kv_pairs(server_args)
    server_config = dict(server_kv_pairs)
    client_config = dict(client_kv_pairs)
    return {
        "framework": framework,
        "client_args": client_config,
        "server_args": server_config,
        "server_cmd_args": server_cmd_args,
        "normalized_config": normalize_config(framework, server_config, client_config),
    }


def construct_result_config(
    benchmark_settings: dict[str, t.Any], framework: t.Optional[str]
) -> dict[str, t.Any]:
    """Add framework lifecycle metadata to a serializable configuration."""
    result_config = dict(benchmark_settings)
    if framework == "trtllm":
        engine_args, runtime_args = predefined.trtllm.classify_server_args(
            result_config["server_args"]
        )
        result_config["engine_args"] = engine_args
        result_config["runtime_args"] = runtime_args
    return result_config


def extract_token_lengths(client_params: dict) -> tuple[int, int]:
    """Extract input and output token lengths from client parameters."""
    input_len = client_params.get("random_input_len", 0)
    output_len = client_params.get("random_output_len", 0)

    # If using sharegpt dataset, these might be None
    if input_len == 0 and output_len == 0:
        # Check if using sharegpt
        if client_params.get("dataset_name") == "sharegpt":
            # sharegpt_output_len might be specified
            output_len = client_params.get("sharegpt_output_len", -1)
            input_len = -1  # Variable for sharegpt

    return input_len, output_len


def validate_fixed_workload(
    client_args_sets: list[lo_args.ArgSet], int_ranges: tuple[IntRange, ...] = ()
) -> None:
    """Engine searches must not silently change the benchmark workload."""
    variable = [
        arg_set.name
        for arg_set in client_args_sets
        if len(arg_set.get_all_possible_arg_values()) > 1
    ]
    variable.extend(
        range_.name for range_ in int_ranges if range_.scope == lo_args.ArgScope.CLIENT
    )
    if variable:
        names = ", ".join(str(name) for name in variable)
        raise click.UsageError(
            "Workload parameters must be fixed while optimizing engine parameters: "
            f"{names}"
        )


def find_best_throughput_configs(
    all_results: list, constraints: t.Optional[list] = None
) -> dict:
    """Find configurations with best input/output throughput."""
    best_configs: dict[str, t.Any] = {
        "best_input_throughput": None,
        "best_output_throughput": None,
        "best_input_throughput_constrained": None,
        "best_output_throughput_constrained": None,
    }

    # Find best configurations
    for result in all_results:
        if "results" not in result:
            continue

        metrics = result["results"]
        if metrics.get("benchmark_valid") is False or metrics.get("failure_rate", 0) != 0:
            continue
        input_tp = metrics.get("input_throughput", 0)
        output_tp = metrics.get("output_throughput", 0)

        # Update unconstrained bests
        if (
            best_configs["best_input_throughput"] is None
            or input_tp > best_configs["best_input_throughput"]["throughput"]
        ):
            best_configs["best_input_throughput"] = {
                "throughput": input_tp,
                "config": result["config"],
                "cmd": result.get("cmd", ""),
            }

        if (
            best_configs["best_output_throughput"] is None
            or output_tp > best_configs["best_output_throughput"]["throughput"]
        ):
            best_configs["best_output_throughput"] = {
                "throughput": output_tp,
                "config": result["config"],
                "cmd": result.get("cmd", ""),
            }

        # Update constrained bests if constraints are satisfied
        if constraints and satisfies_constraints(result, constraints):
            if (
                best_configs["best_input_throughput_constrained"] is None
                or input_tp
                > best_configs["best_input_throughput_constrained"]["throughput"]
            ):
                best_configs["best_input_throughput_constrained"] = {
                    "throughput": input_tp,
                    "config": result["config"],
                    "cmd": result.get("cmd", ""),
                }

            if (
                best_configs["best_output_throughput_constrained"] is None
                or output_tp
                > best_configs["best_output_throughput_constrained"]["throughput"]
            ):
                best_configs["best_output_throughput_constrained"] = {
                    "throughput": output_tp,
                    "config": result["config"],
                    "cmd": result.get("cmd", ""),
                }

    return best_configs


def get_config_id(client_params: dict, server_params: dict) -> str:
    """Create a descriptive ID for a configuration."""
    client_param_strs = [f"{k}-{v}" for k, v in sorted(client_params.items())]
    server_param_strs = [f"{k}-{v}" for k, v in sorted(server_params.items())]

    config_id_parts = []
    if client_param_strs:
        config_id_parts.append("client_" + "-".join(client_param_strs))
    if server_param_strs:
        config_id_parts.append("server_" + "-".join(server_param_strs))

    return "_".join(config_id_parts) or "default"


def _load_jsonl(path: pathlib.Path) -> list[dict[str, t.Any]]:
    results: list[dict[str, t.Any]] = []
    with path.open() as stream:
        for line_num, line in enumerate(stream, 1):
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(
                    f"Could not parse line {line_num} in {path}: {line.strip()}"
                )
    return results


def _write_combined_results(
    output_json: pathlib.Path,
    output_jsonl: pathlib.Path,
    gpu_type: str,
    gpus: int,
    model: t.Optional[str],
    constraints_for_viz: list,
    parsed_constraints: list,
    optimization_metadata: t.Optional[dict[str, t.Any]] = None,
    workload_metadata: t.Optional[dict[str, t.Any]] = None,
) -> None:
    all_results = _load_jsonl(output_jsonl) if output_jsonl.exists() else []
    input_len = output_len = None
    if all_results:
        client = all_results[0].get("config", {}).get("client_args", {})
        input_len, output_len = extract_token_lengths(client)
    enhanced_results: dict[str, t.Any] = {
        "metadata": {
            "gpu_type": gpu_type,
            "gpu_count": gpus,
            "model_tag": model,
            "total_tests": len(all_results),
            "constraints": constraints_for_viz,
            "input_len": input_len,
            "output_len": output_len,
            **(workload_metadata or {}),
        },
        "best_configurations": find_best_throughput_configs(
            all_results, parsed_constraints
        ),
        "test_results": all_results,
    }
    if optimization_metadata is not None:
        enhanced_results["optimization"] = optimization_metadata
    with output_json.open("w") as stream:
        json.dump(enhanced_results, stream, indent=2, cls=InfinityToNullEncoder)


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
) -> tuple[dict[str, t.Any], t.Optional[float]]:
    """Run and persist one configuration, always cleaning up its server."""
    bench_client = importlib.import_module("llm_optimizer.bench_client")
    from llm_optimizer.optimization import extract_objective

    client_params = benchmark_settings["client_args"]
    server_params = benchmark_settings["server_args"]
    full_server_cmd = f"{server_cmd} {' '.join(benchmark_settings['server_cmd_args'])}"
    server_process = None
    try:
        server_env = (
            get_server_env(framework, gpus, server_params) if managed_server else {}
        )
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
        benchmark_result = bench_client.run_benchmark(benchmark_args)
        if not isinstance(benchmark_result, dict):
            raise RuntimeError("Benchmark did not return a result")

        feasible = satisfies_constraints(
            benchmark_result, parsed_constraints or []
        )
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
        if output_jsonl_path:
            with output_jsonl_path.open("a") as stream:
                stream.write(json.dumps(result, cls=InfinityToNullEncoder) + "\n")
            logger.info(f"Appended result to {output_jsonl_path}")
        elif output_file_path:
            with output_file_path.open("w") as stream:
                json.dump(result, stream, indent=2, cls=InfinityToNullEncoder)
            logger.info(f"Benchmark results saved to {output_file_path}")
        if objective_error and objective is not None:
            raise ValueError(objective_error)
        return result, objective_value
    finally:
        if server_process:
            terminate_process_top_down(server_process)


def _trial_summary(trial: t.Any) -> dict[str, t.Any]:
    return {
        "number": trial.number,
        "objective_value": trial.value,
        "config_id": trial.user_attrs.get("config_id"),
        "slo_feasible": trial.user_attrs.get("slo_feasible", True),
    }


def _run_optuna_search(
    *,
    server_args_sets: list[lo_args.ArgSet],
    client_args_sets: list[lo_args.ArgSet],
    int_ranges: tuple[IntRange, ...],
    framework: str,
    model: str,
    server_cmd: str,
    managed_server: bool,
    gpus: int,
    gpu_type: str,
    host: str,
    port: t.Optional[int],
    ready_url: str,
    mute_server: bool,
    rest: int,
    output_json: str,
    continue_flag: bool,
    dry_run: bool,
    parsed_constraints: list,
    constraints_for_viz: list,
    constraints_text: t.Optional[str],
    trials_budget: int,
    timeout: t.Optional[float],
    objective: str,
    seed: int,
    storage: t.Optional[str],
    study_name: t.Optional[str],
    sampler_name: str,
    sampler_kwargs_text: str,
    pruner_name: str,
    pruner_kwargs_text: str,
    workload_manifest: t.Optional[dict[str, t.Any]] = None,
    workload_metadata: t.Optional[dict[str, t.Any]] = None,
) -> None:
    from llm_optimizer.optimization import (
        OptunaSearchSpace,
        canonical_signature,
        constraint_violations,
        default_storage_url,
        extract_objective,
        objective_direction,
        require_optuna,
        storage_kind,
    )

    direction = objective_direction(objective)
    output_path = pathlib.Path(output_json)
    jsonl_path = output_path.with_suffix(".jsonl")
    storage_url = storage or default_storage_url(output_path)
    try:
        space = OptunaSearchSpace(
            tuple(server_args_sets + client_args_sets), int_ranges
        )
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc
    signature_payload = {
        "model": model,
        "server_command": server_cmd,
        "framework": framework,
        "search_space": space.canonical(),
        "objective": objective,
        "direction": direction,
        "constraints": constraints_text or "",
        "workload_sha256": (workload_metadata or {}).get("workload_sha256"),
        "sampler": sampler_name,
        "sampler_kwargs": sampler_kwargs_text,
        "pruner": pruner_name,
        "pruner_kwargs": pruner_kwargs_text,
        "gpu_count": gpus,
        "host": host,
        "port": port,
    }
    signature, canonical_payload = canonical_signature(signature_payload)
    resolved_study_name = study_name or f"llm-optimizer-{framework}-{signature[:12]}"

    if dry_run:
        click.echo("Optuna dry run mode enabled.")
        click.echo(f"Framework: {framework}")
        click.echo(f"Search-space size: {space.cardinality}")
        click.echo(f"Objective: {objective} ({direction})")
        click.echo(f"Sampler: {sampler_name}")
        click.echo(f"Pruner: {pruner_name}")
        click.echo(f"Trial budget: {trials_budget}")
        click.echo(f"Timeout: {timeout if timeout is not None else 'none'}")
        click.echo(f"Storage: {storage_kind(storage_url)}")
        click.echo(f"Study name: {resolved_study_name}")
        return

    try:
        optuna = require_optuna()
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc

    if not continue_flag and (output_path.exists() or jsonl_path.exists()):
        raise click.ClickException(
            f"Output files already exist. Use --continue to resume: {output_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_results = _load_jsonl(jsonl_path) if jsonl_path.exists() else []
    if continue_flag and output_path.exists() and not jsonl_path.exists():
        try:
            existing_combined = json.loads(output_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise click.ClickException(
                "Cannot validate the existing Optuna output file."
            ) from exc
        if (
            existing_combined.get("optimization", {}).get("study_signature")
            != signature
        ):
            raise click.ClickException(
                "Existing output does not match this Optuna study signature."
            )
        if existing_combined.get("metadata", {}).get("total_tests", 0):
            raise click.ClickException(
                "Cannot resume successful Optuna runs without their JSONL records."
            )
    for result in existing_results:
        recorded_signature = result.get("optimization", {}).get("study_signature")
        if recorded_signature != signature:
            raise click.ClickException(
                "Existing JSONL records do not match this Optuna study signature."
            )

    def constraints_func(frozen_trial: t.Any) -> list[float]:
        return t.cast(
            list[float],
            frozen_trial.user_attrs.get(
                "constraint_values", [float("inf")] * len(parsed_constraints)
            ),
        )

    def build_optuna_component(
        component_name: str,
        kwargs_text: str,
        namespace: t.Any,
        default_name: str,
    ) -> t.Any:
        name = component_name or default_name
        component_cls = getattr(namespace, name, None)
        if component_cls is None and "." in name:
            module_name, class_name = name.rsplit(".", 1)
            component_cls = getattr(importlib.import_module(module_name), class_name)
        if component_cls is None or not callable(component_cls):
            raise click.UsageError(f"Unknown Optuna component: {name}")
        try:
            kwargs = json.loads(kwargs_text) if kwargs_text else {}
        except json.JSONDecodeError as exc:
            raise click.UsageError(
                f"Invalid JSON for Optuna component '{name}': {exc}"
            ) from exc
        if not isinstance(kwargs, dict):
            raise click.UsageError(
                f"Optuna component kwargs for '{name}' must be a JSON object."
            )
        return component_cls(**kwargs)

    sampler_kwargs: dict[str, t.Any] = {} if sampler_name.endswith("GridSampler") else {"seed": seed}
    if sampler_kwargs_text:
        try:
            parsed_sampler_kwargs = json.loads(sampler_kwargs_text)
        except json.JSONDecodeError as exc:
            raise click.UsageError(f"Invalid JSON for Optuna sampler: {exc}") from exc
        if not isinstance(parsed_sampler_kwargs, dict):
            raise click.UsageError("Optuna sampler kwargs must be a JSON object.")
        sampler_kwargs.update(parsed_sampler_kwargs)
    if parsed_constraints and "constraints_func" not in sampler_kwargs:
        sampler_kwargs["constraints_func"] = constraints_func
    sampler_cls = getattr(optuna.samplers, sampler_name, None)
    if sampler_cls is None and "." in sampler_name:
        module_name, class_name = sampler_name.rsplit(".", 1)
        sampler_cls = getattr(importlib.import_module(module_name), class_name)
    if sampler_cls is None or not callable(sampler_cls):
        raise click.UsageError(f"Unknown Optuna sampler: {sampler_name}")
    if sampler_cls.__name__ == "GridSampler":
        if "search_space" in sampler_kwargs:
            raise click.UsageError(
                "GridSampler search_space is generated from the benchmark search space"
            )
        sampler_kwargs["search_space"] = space.grid_search_space()
        if parsed_constraints and "constraints_func" in sampler_kwargs:
            sampler_kwargs.pop("constraints_func")
    sampler = sampler_cls(**sampler_kwargs)
    pruner = build_optuna_component(
        pruner_name, pruner_kwargs_text, optuna.pruners, "NopPruner"
    )
    study = optuna.create_study(
        study_name=resolved_study_name,
        storage=storage_url,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=continue_flag,
    )
    existing_signature = study.user_attrs.get("llm_optimizer_signature")
    if existing_signature and existing_signature != signature:
        raise click.ClickException(
            "The existing Optuna study signature does not match this benchmark. "
            "Choose another study/output or restore the original arguments."
        )
    if not existing_signature:
        study.set_user_attr("llm_optimizer_signature", signature)
        study.set_user_attr("llm_optimizer_signature_payload", canonical_payload)

    result_by_id = {
        get_config_id(
            result.get("config", {}).get("client_args", {}),
            result.get("config", {}).get("server_args", {}),
        ): result
        for result in existing_results
    }

    # Reconcile the append-before-tell crash window.
    for frozen in study.get_trials(deepcopy=False):
        config_id = frozen.user_attrs.get("config_id")
        if (
            frozen.state == optuna.trial.TrialState.RUNNING
            and config_id in result_by_id
        ):
            result = result_by_id[config_id]
            value = extract_objective(result.get("results", {}), objective)
            violations = constraint_violations(
                result.get("results", {}), parsed_constraints
            )
            resumed_trial = optuna.trial.Trial(study, frozen._trial_id)
            resumed_trial.set_user_attr("constraint_values", violations)
            resumed_trial.set_user_attr(
                "slo_feasible", all(item <= 0 for item in violations)
            )
            study.tell(resumed_trial, value)

    evaluated_ids = set(result_by_id)
    for frozen in study.get_trials(deepcopy=False):
        if frozen.state in (
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.FAIL,
        ):
            config_id = frozen.user_attrs.get("config_id")
            if config_id:
                evaluated_ids.add(config_id)

    started = time.monotonic()
    attempts = 0
    max_attempts = max(100, trials_budget * 20, min(space.cardinality * 10, 10000))
    stop_reason = "trial_budget"
    effective_trials_budget = (
        space.cardinality if sampler.__class__.__name__ == "GridSampler" else trials_budget
    )
    while len(evaluated_ids) < effective_trials_budget:
        if len(evaluated_ids) >= space.cardinality:
            stop_reason = "search_space_exhausted"
            break
        if timeout is not None and time.monotonic() - started >= timeout:
            stop_reason = "timeout"
            break
        if attempts >= max_attempts:
            stop_reason = "sampling_attempt_limit"
            break
        attempts += 1
        trial = study.ask()
        combo = space.suggest(trial)
        settings = construct_benchmark_settings(combo, framework)
        config_id = get_config_id(settings["client_args"], settings["server_args"])
        trial.set_user_attr("config_id", config_id)
        if config_id in evaluated_ids:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            continue

        logger.info(
            f"Starting Optuna trial {trial.number} "
            f"({len(evaluated_ids) + 1}/{min(trials_budget, space.cardinality)}): "
            f"{config_id}"
        )
        try:
            result, benchmark_value = run_single_configuration(
                settings,
                framework=framework,
                model=model,
                server_cmd=server_cmd,
                managed_server=managed_server,
                gpus=gpus,
                gpu_type=gpu_type,
                host=host,
                port=port,
                ready_url=ready_url,
                mute_server=mute_server,
                constraints_for_viz=constraints_for_viz,
                output_jsonl_path=jsonl_path,
                objective=objective,
                trial_number=trial.number,
                study_signature=signature,
                parsed_constraints=parsed_constraints,
                workload_manifest=workload_manifest,
                workload_metadata=workload_metadata,
            )
            violations = constraint_violations(result["results"], parsed_constraints)
            feasible = all(item <= 0 for item in violations)
            trial.set_user_attr("constraint_values", violations)
            trial.set_user_attr("slo_feasible", feasible)
            assert benchmark_value is not None
            try:
                study.tell(trial, benchmark_value)
            except RuntimeError as exc:
                # Optuna's GridSampler marks the final trial complete and then
                # calls Study.stop(), which is only legal inside optimize().
                if "Study.stop" not in str(exc):
                    raise
        except KeyboardInterrupt:
            trial.set_user_attr("slo_feasible", False)
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            raise
        except Exception as exc:
            logger.error(f"Optuna trial {trial.number} failed for {config_id}: {exc}")
            trial.set_user_attr("slo_feasible", False)
            trial.set_user_attr(
                "constraint_values", [float("inf")] * len(parsed_constraints)
            )
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
        evaluated_ids.add(config_id)
        if len(evaluated_ids) < min(effective_trials_budget, space.cardinality) and rest:
            logger.info(f"Resting for {rest} seconds before the next run.")
            time.sleep(rest)

    frozen_trials = study.get_trials(deepcopy=False)
    state_counts: dict[str, int] = {}
    for frozen in frozen_trials:
        state_counts[frozen.state.name] = state_counts.get(frozen.state.name, 0) + 1
    completed = [
        trial
        for trial in frozen_trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    key = (lambda trial: trial.value) if direction == "maximize" else (
        lambda trial: -trial.value
    )
    unconstrained = max(completed, key=key) if completed else None
    feasible_trials = [
        trial for trial in completed if trial.user_attrs.get("slo_feasible", True)
    ]
    best_feasible = max(feasible_trials, key=key) if feasible_trials else None
    metadata = {
        "strategy": "optuna",
        "study_name": resolved_study_name,
        "study_signature": signature,
        "sampler": sampler.__class__.__name__,
        "sampler_kwargs": {
            key: value for key, value in sampler_kwargs.items() if key != "search_space"
        },
        "pruner": pruner.__class__.__name__,
        "objective": objective,
        "direction": direction,
        "budget": trials_budget,
        "timeout": timeout,
        "seed": seed,
        "search_space_size": space.cardinality,
        "trial_state_counts": state_counts,
        "stop_reason": stop_reason,
        "best_feasible_trial": (
            _trial_summary(best_feasible) if best_feasible else None
        ),
        "best_unconstrained_trial": (
            _trial_summary(unconstrained) if unconstrained else None
        ),
        "storage_kind": storage_kind(storage_url),
    }
    _write_combined_results(
        output_path,
        jsonl_path,
        gpu_type,
        gpus,
        model,
        constraints_for_viz,
        parsed_constraints,
        metadata,
        workload_metadata,
    )
    logger.info(f"Optuna benchmark results saved to {output_path}")


@click.group(invoke_without_command=True)
@click.option("--server-cmd", type=str, help="The command to start the server.")
@click.option("--model", type=str, help="The model to use.")
@click.option(
    "--framework",
    type=click.Choice(PREDEFINED_FRAMEWORKS),
    help="The framework to use.",
)
@click.option(
    "--server-args", type=str, help="Arguments for the server.", multiple=True
)
@click.option(
    "--client-args", type=str, help="Arguments for the client.", multiple=True
)
@click.option("--gpus", type=int, help="The number of GPUs to use.")
@click.option("--dry-run", is_flag=True, help="A dry run will not run the command.")
@click.option(
    "--output-dir", default="results", help="Directory to store output files."
)
@click.option(
    "--output-json",
    type=str,
    default=None,
    help="Path to output a single JSON file with all results.",
)
@click.option(
    "--continue",
    "-c",
    "continue_flag",
    is_flag=True,
    help="Skip configs that already have output files.",
)
@click.option(
    "--rest", type=int, default=10, help="Rest time in seconds between benchmark runs."
)
@click.option("--mute-server", is_flag=True, help="Suppress server process stdout.")
@click.option(
    "--ready-endpoint",
    default=None,
    help="Endpoint to check if server is ready (e.g., /health, /readyz).",
)
@click.option(
    "--host", type=str, default="127.0.0.1", help="Server host to connect to."
)
@click.option("--port", type=int, default=None, help="Server port to connect to.")
@click.option(
    "--constraints", type=str, help="SLO constraints (e.g., 'ttft<300ms;itl<8.5ms')"
)
@click.option(
    "--search-strategy",
    type=click.Choice(["grid", "optuna"]),
    default="grid",
    show_default=True,
    help="Configuration search driver.",
)
@click.option(
    "--optuna-trials",
    type=click.IntRange(min=1),
    default=30,
    show_default=True,
    help="Unique configurations to attempt in the study.",
)
@click.option(
    "--optuna-timeout",
    type=click.FloatRange(min=0, min_open=True),
    help="Optional optimization wall-clock limit in seconds.",
)
@click.option(
    "--optuna-objective",
    default="output_throughput",
    show_default=True,
    help="Benchmark metric to optimize.",
)
@click.option(
    "--optuna-seed",
    type=int,
    default=1,
    show_default=True,
    help="TPE sampler seed.",
)
@click.option(
    "--optuna-sampler",
    default="TPESampler",
    show_default=True,
    help="Optuna sampler class name or import path.",
)
@click.option(
    "--optuna-sampler-kwargs",
    default="",
    help="JSON object of keyword arguments passed to the sampler.",
)
@click.option(
    "--optuna-pruner",
    default="NopPruner",
    show_default=True,
    help="Optuna pruner class name or import path.",
)
@click.option(
    "--optuna-pruner-kwargs",
    default="",
    help="JSON object of keyword arguments passed to the pruner.",
)
@click.option(
    "--optuna-storage", type=str, help="Optuna/SQLAlchemy RDB storage URL."
)
@click.option(
    "--optuna-study-name", type=str, help="Study name; derived when omitted."
)
@click.option(
    "--optuna-server-args",
    multiple=True,
    help="Optuna-only server integer ranges: name=int(low,high[,step]).",
)
@click.option(
    "--optuna-client-args",
    multiple=True,
    help="Optuna-only client integer ranges: name=int(low,high[,step]).",
)
@click.pass_context
def cli(
    ctx,
    server_cmd,
    model,
    framework,
    server_args,
    client_args,
    gpus,
    dry_run,
    output_dir,
    output_json,
    continue_flag,
    rest,
    mute_server,
    ready_endpoint,
    host,
    port,
    constraints,
    search_strategy,
    optuna_trials,
    optuna_timeout,
    optuna_objective,
    optuna_seed,
    optuna_sampler,
    optuna_sampler_kwargs,
    optuna_pruner,
    optuna_pruner_kwargs,
    optuna_storage,
    optuna_study_name,
    optuna_server_args,
    optuna_client_args,
):
    """A CLI tool to optimize LLM performance."""
    if ctx.invoked_subcommand is None:
        # If no subcommand is provided, run the main benchmark command
        benchmark(
            server_cmd,
            model,
            framework,
            server_args,
            client_args,
            gpus,
            dry_run,
            output_dir,
            output_json,
            continue_flag,
            rest,
            mute_server,
            ready_endpoint,
            host,
            port,
            constraints,
            search_strategy,
            optuna_trials,
            optuna_timeout,
            optuna_objective,
            optuna_seed,
            optuna_sampler,
            optuna_sampler_kwargs,
            optuna_pruner,
            optuna_pruner_kwargs,
            optuna_storage,
            optuna_study_name,
            optuna_server_args,
            optuna_client_args,
        )


def benchmark(
    server_cmd,
    model,
    framework,
    server_args,
    client_args,
    gpus,
    dry_run,
    output_dir,
    output_json,
    continue_flag,
    rest,
    mute_server,
    ready_endpoint,
    host,
    port,
    constraints,
    search_strategy="grid",
    optuna_trials=30,
    optuna_timeout=None,
    optuna_objective="output_throughput",
    optuna_seed=1,
    optuna_sampler="TPESampler",
    optuna_sampler_kwargs="",
    optuna_pruner="NopPruner",
    optuna_pruner_kwargs="",
    optuna_storage=None,
    optuna_study_name=None,
    optuna_server_args=(),
    optuna_client_args=(),
):
    """A CLI tool to optimize LLM performance."""
    from llm_optimizer.performance import parse_slo_constraints
    from llm_optimizer.visualization.visualize import (
        convert_constraints_for_visualization,
    )

    # Parse constraints if provided
    parsed_constraints = []
    constraints_for_viz = []
    if constraints:
        try:
            parsed_constraints = parse_slo_constraints(constraints)
            constraints_for_viz = convert_constraints_for_visualization(
                parsed_constraints
            )
            logger.info(
                f"Parsed {len(parsed_constraints)} constraint(s): {constraints}"
            )
        except ValueError as e:
            if search_strategy == "optuna":
                raise click.UsageError(f"Invalid Optuna constraints: {e}") from e
            logger.error(f"Error parsing constraints: {e}")
            # Continue without constraints rather than failing

    managed_server = not server_cmd
    if search_strategy == "optuna":
        from llm_optimizer.optimization import objective_direction

        if not output_json:
            raise click.UsageError("--output-json is required for Optuna search.")
        if not framework or not model:
            raise click.UsageError(
                "Optuna search requires one --framework and one --model."
            )
        try:
            objective_direction(optuna_objective)
        except ValueError as exc:
            raise click.UsageError(str(exc)) from exc

    workload_manifest = None
    workload_metadata = None
    if managed_server:
        if not model or not framework:
            raise click.UsageError(
                "If --server-cmd is not provided, both --model and "
                "--framework are required."
            )

        framework_spec = predefined.get_framework_spec(framework)
        if port is None:
            port = framework_spec.default_port
        if ready_endpoint is None:
            ready_endpoint = framework_spec.ready_endpoint

        tmpl = framework_spec.server_command
        server_cmd = tmpl.format(
            model=model,
            host=host,
            port=port,
        )

    if ready_endpoint is None:
        ready_endpoint = "/health"

    if gpus is None:
        gpus = get_gpu_count()

    # Detect GPU type
    gpu_type = detect_gpu_type() or "unknown"
    logger.info(f"Detected GPU type: {gpu_type}, Count: {gpus}")

    if dry_run and search_strategy == "grid":
        click.echo("Dry run mode enabled.")

    server_args = ";".join(server_args)
    client_args = ";".join(client_args)

    server_configs = None
    if framework:
        server_configs = predefined.SERVER_CONFIGS[framework]

    server_args_sets = lo_args.parse_args_str(
        server_args,
        scope=lo_args.ArgScope.SERVER,
        configs=server_configs,
        strict=False,
    )

    client_args_sets = lo_args.parse_args_str(
        client_args,
        scope=lo_args.ArgScope.CLIENT,
        configs=predefined.CLIENT_CONFIGS,
        strict=True,
    )

    if search_strategy != "optuna" and (optuna_server_args or optuna_client_args):
        raise click.UsageError(
            "--optuna-server-args and --optuna-client-args require "
            "--search-strategy optuna."
        )
    int_ranges: tuple[IntRange, ...] = ()
    if search_strategy == "optuna":
        try:
            int_ranges = parse_int_ranges(
                optuna_server_args,
                lo_args.ArgScope.SERVER,
                server_configs or {},
            ) + parse_int_ranges(
                optuna_client_args,
                lo_args.ArgScope.CLIENT,
                predefined.CLIENT_CONFIGS,
            )
        except ValueError as exc:
            raise click.UsageError(str(exc)) from exc

    if search_strategy == "optuna":
        # Fail on a missing optional dependency before importing the heavyweight
        # benchmark client (which also depends on transformers).
        from llm_optimizer.optimization import require_optuna

        try:
            require_optuna()
        except RuntimeError as exc:
            raise click.ClickException(str(exc)) from exc

    if not dry_run and framework and model:
        bench_client = importlib.import_module("llm_optimizer.bench_client")

        if hasattr(bench_client, "prepare_workload_manifest"):
            validate_fixed_workload(client_args_sets, int_ranges)
            fixed_client_args = {
                key: value
                for arg_set in client_args_sets
                for key, value in lo_args.get_all_kv_pairs(
                    [arg_set.get_all_possible_arg_values()[0]]
                ).items()
            }
            workload_args = {
                "backend": predefined.get_framework_spec(framework).benchmark_backend,
                "framework": framework,
                "model": model,
                "host": host,
                "port": port,
                "dataset_name": "sharegpt",
                "num_prompts": 1000,
                "request_rate": float("inf"),
                "seed": 1,
            }
            workload_args.update(fixed_client_args)
            fixed_server_args = {
                key: value
                for arg_set in server_args_sets
                for key, value in lo_args.get_all_kv_pairs(
                    [arg_set.get_all_possible_arg_values()[0]]
                ).items()
            }
            workload_args["model_revision"] = next(
                (fixed_server_args.get(key) for key in ("revision", "hf_revision") if key in fixed_server_args),
                None,
            )
            workload_args["tokenizer_revision"] = fixed_server_args.get(
                "tokenizer_revision"
            )
            workload_manifest, workload_metadata = bench_client.prepare_workload_manifest(
                workload_args
            )
            manifest_path = (
                pathlib.Path(output_json).with_suffix(".workload.json")
                if output_json
                else pathlib.Path(output_dir) / "workload_manifest.json"
            )
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(workload_manifest, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    ready_url = f"http://{host}:{port}{ready_endpoint}"

    if search_strategy == "optuna":
        _run_optuna_search(
            server_args_sets=server_args_sets,
            client_args_sets=client_args_sets,
            int_ranges=int_ranges,
            framework=framework,
            model=model,
            server_cmd=server_cmd,
            managed_server=managed_server,
            gpus=gpus,
            gpu_type=gpu_type,
            host=host,
            port=port,
            ready_url=ready_url,
            mute_server=mute_server,
            rest=rest,
            output_json=output_json,
            continue_flag=continue_flag,
            dry_run=dry_run,
            parsed_constraints=parsed_constraints,
            constraints_for_viz=constraints_for_viz,
            constraints_text=constraints,
            trials_budget=optuna_trials,
            timeout=optuna_timeout,
            objective=optuna_objective,
            seed=optuna_seed,
            sampler_name=optuna_sampler,
            sampler_kwargs_text=optuna_sampler_kwargs,
            pruner_name=optuna_pruner,
            pruner_kwargs_text=optuna_pruner_kwargs,
            storage=optuna_storage,
            study_name=optuna_study_name,
            workload_manifest=workload_manifest,
            workload_metadata=workload_metadata,
        )
        return

    all_combinations = lo_args.get_all_arg_combinations(
        client_args_sets=client_args_sets,
        server_args_sets=server_args_sets,
    )
    total_configs = len(all_combinations)
    logger.info(f"Generated {total_configs} configuration(s) to run.")

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_jsonl_path = None
    if output_json:
        output_jsonl_path = pathlib.Path(output_json).with_suffix(".jsonl")

        # Check if output files already exist (unless continue flag is set)
        if not continue_flag:
            if pathlib.Path(output_json).exists():
                logger.error(
                    f"Output file {output_json} already exists! "
                    "Use --continue to resume or choose a different output file."
                )
                return
            if output_jsonl_path.exists():
                logger.error(
                    f"JSONL file {output_jsonl_path} already exists! "
                    "Use --continue to resume or choose a different output file."
                )
                return

    completed_config_ids = set()
    if continue_flag and output_jsonl_path and output_jsonl_path.exists():
        logger.info(
            f"Found existing JSONL file, loading completed runs: {output_jsonl_path}"
        )

        # Validate that existing results are for the same model
        existing_model = None
        with open(output_jsonl_path) as f:
            for line_num, line in enumerate(f, 1):
                try:
                    result = json.loads(line)
                    result_model = result.get("metadata", {}).get("model_tag")
                    current_model = model

                    if existing_model is None:
                        existing_model = result_model
                    elif result_model != existing_model:
                        logger.warning(
                            f"Inconsistent models in existing results: "
                            f"{existing_model} vs {result_model}"
                        )

                    if result_model != current_model:
                        logger.error(
                            f"Model mismatch! Existing results for '{existing_model}' "
                            f"but current model is '{current_model}'. "
                            "Cannot append to existing results."
                        )
                        return

                    client_params = result.get("config", {}).get("client_args", {})
                    server_params = result.get("config", {}).get("server_args", {})
                    config_id = get_config_id(client_params, server_params)
                    completed_config_ids.add(config_id)
                except json.JSONDecodeError:
                    logger.warning(
                        f"Could not parse line {line_num} in {output_jsonl_path}: "
                        f"{line.strip()}"
                    )
        logger.info(
            f"Loaded {len(completed_config_ids)} completed runs for model "
            f"{existing_model}"
        )

    for idx, combo in enumerate(all_combinations):
        benchmark_settings = construct_benchmark_settings(combo, framework)

        client_params = benchmark_settings["client_args"]
        server_params = benchmark_settings["server_args"]

        config_id = get_config_id(client_params, server_params)
        output_file_path = output_dir / f"{config_id}.json"

        logger.info("-" * 80)
        logger.info(f"Starting run {idx + 1}/{total_configs}: {config_id}")

        if continue_flag:
            if output_jsonl_path:
                if config_id in completed_config_ids:
                    logger.info(
                        f"Skipping as config_id '{config_id}' found in "
                        f"{output_jsonl_path}"
                    )
                    continue
            elif output_file_path.exists():
                logger.info(
                    f"Skipping as output file already exists: {output_file_path}"
                )
                continue

        if dry_run:
            print(benchmark_settings)
            continue

        try:
            run_single_configuration(
                benchmark_settings,
                framework=framework,
                model=model,
                server_cmd=server_cmd,
                managed_server=managed_server,
                gpus=gpus,
                gpu_type=gpu_type,
                host=host,
                port=port,
                ready_url=ready_url,
                mute_server=mute_server,
                constraints_for_viz=constraints_for_viz,
                output_jsonl_path=output_jsonl_path,
                output_file_path=output_file_path,
                parsed_constraints=parsed_constraints,
                workload_manifest=workload_manifest,
                workload_metadata=workload_metadata,
            )
        except Exception as e:
            logger.error(f"Error during run for config {config_id}: {e}")
        finally:
            if idx < total_configs - 1:
                logger.info(f"Resting for {rest} seconds before the next run.")
                time.sleep(rest)

    if output_jsonl_path and output_jsonl_path.exists():
        all_results = []
        with open(output_jsonl_path) as f:
            for line in f:
                try:
                    all_results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # Already warned about this

        # Find best throughput configurations
        best_configs = find_best_throughput_configs(all_results, parsed_constraints)

        # Extract input/output lengths from first result (they should be consistent)
        input_len = output_len = None
        if all_results:
            first_result = all_results[0]
            first_client_params = first_result.get("config", {}).get("client_args", {})
            input_len, output_len = extract_token_lengths(first_client_params)

        # Create enhanced result structure
        enhanced_results = {
            "metadata": {
                "gpu_type": gpu_type,
                "gpu_count": gpus,
                "model_tag": model,
                "total_tests": len(all_results),
                "constraints": constraints_for_viz,
                "input_len": input_len,
                "output_len": output_len,
                **(workload_metadata or {}),
            },
            "best_configurations": best_configs,
            "test_results": all_results,
        }

        with open(output_json, "w") as f:
            json.dump(enhanced_results, f, indent=2, cls=InfinityToNullEncoder)
        logger.info(
            f"All benchmark results saved to {output_json} with enhanced metadata"
        )

    logger.info("-" * 80)
    logger.info("All benchmark runs completed.")

    # Auto-generate HTML visualization if output JSON exists
    if output_json and pathlib.Path(output_json).exists():
        try:
            from llm_optimizer.visualization.visualize import ParetoLLMOptimizer

            # Create optimizer instance with default config
            config_path = (
                pathlib.Path(__file__).parent
                / "visualization"
                / "visualization_config.json"
            )
            optimizer = ParetoLLMOptimizer(str(config_path))

            # Generate HTML with same base name as JSON
            json_path = pathlib.Path(output_json)
            html_file = json_path.with_suffix(".html")

            logger.info("Generating visualization dashboard...")
            optimizer.generate_dashboard(output_json, output_file=str(html_file))
            logger.info(f"Visualization dashboard saved to {html_file}")

        except Exception as e:
            logger.warning(f"Could not generate visualization dashboard: {e}")
            # Don't fail the benchmark if visualization fails


@cli.command()
@click.option(
    "--data-file",
    type=str,
    required=True,
    help="Path to the JSON data file to visualize",
)
@click.option(
    "--config", type=str, default=None, help="Path to visualization config file"
)
@click.option(
    "-o",
    "--output",
    type=str,
    default=None,
    help="Output HTML file path (default: pareto_llm_dashboard.html)",
)
@click.option("--serve", is_flag=True, help="Start HTTP server after generating HTML")
@click.option(
    "--port",
    type=int,
    default=8080,
    help="Port to run the dashboard server (used with --serve)",
)
def visualize(data_file, config, output, serve, port):
    """Generate and open visualization dashboard from benchmark results."""
    try:
        from llm_optimizer.visualization.visualize import ParetoLLMOptimizer

        # Determine config file path
        if config is None:
            config_path = (
                pathlib.Path(__file__).parent
                / "visualization"
                / "visualization_config.json"
            )
        else:
            config_path = pathlib.Path(config)

        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            return

        # Create optimizer instance
        optimizer = ParetoLLMOptimizer(str(config_path))

        data_files = [
            pathlib.Path(name.strip()) for name in data_file.split(",") if name.strip()
        ]
        missing_files = [str(path) for path in data_files if not path.exists()]
        if missing_files:
            logger.error(f"Data file(s) not found: {', '.join(missing_files)}")
            return

        # Generate dashboard
        logger.info(f"Generating dashboard from {data_file}...")
        html_file = optimizer.generate_dashboard(data_file, output_file=output)
        logger.info(f"Dashboard generated: {html_file}")

        # Start server only if requested
        if serve:
            logger.info(f"Starting server on port {port}...")
            optimizer.start_server(port=port)
        else:
            logger.info(f"HTML dashboard ready at: {html_file}")
            logger.info("Use --serve to start a local server and open in browser")

    except Exception as e:
        logger.error(f"Failed to generate visualization: {e}")
        import traceback

        traceback.print_exc()


@cli.command("dynamo-check")
@click.option("--model", required=True, help="Hugging Face model ID or model path.")
@click.option(
    "--framework",
    required=True,
    type=click.Choice(["vllm", "sglang", "trtllm"]),
)
@click.option(
    "--server-args",
    multiple=True,
    help="Native backend parameters; lists and composite grids are supported.",
)
@click.option("--node-count", type=click.IntRange(min=1), required=True)
@click.option("--gpus-per-node", type=click.IntRange(min=1), required=True)
@click.option("--namespace", default="dynamo", show_default=True)
@click.option("--name", default=None, help="DynamoGraphDeployment name.")
@click.option("--image", default=None, help="Override the Dynamo runtime image.")
@click.option("--hf-secret", default="hf-token-secret", show_default=True)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=pathlib.Path),
    default=None,
    help="Write the JSON validation report and generated manifests to this path.",
)
@click.option(
    "--check-cluster",
    is_flag=True,
    help="Validate each manifest with kubectl --dry-run=server.",
)
def dynamo_check(
    model,
    framework,
    server_args,
    node_count,
    gpus_per_node,
    namespace,
    name,
    image,
    hf_secret,
    output,
    check_cluster,
):
    """Validate parameter grids and generate NVIDIA Dynamo Kubernetes manifests."""
    from llm_optimizer.dynamo import (
        build_dynamo_manifest,
        kubernetes_server_dry_run,
        validate_dynamo_config,
    )

    try:
        parsed = lo_args.parse_args_str(
            ";".join(server_args),
            scope=lo_args.ArgScope.SERVER,
            configs=predefined.SERVER_CONFIGS[framework],
            strict=True,
        )
    except (TypeError, ValueError) as exc:
        raise click.UsageError(str(exc)) from exc

    combinations = lo_args.get_all_arg_combinations([], parsed)
    report: dict[str, t.Any] = {
        "model": model,
        "framework": framework,
        "topology": {
            "node_count": node_count,
            "gpus_per_node": gpus_per_node,
        },
        "configurations": [],
    }
    all_valid = True
    for index, combination in enumerate(combinations, 1):
        native_args = dict(lo_args.get_all_kv_pairs(list(combination)))
        validation = validate_dynamo_config(
            framework, native_args, node_count, gpus_per_node
        )
        entry: dict[str, t.Any] = {
            "index": index,
            "server_args": native_args,
            **validation.to_dict(),
        }
        if validation.valid:
            manifest_name = name
            if name and len(combinations) > 1:
                manifest_name = f"{name}-{index}"
            manifest = build_dynamo_manifest(
                framework,
                model,
                native_args,
                node_count,
                gpus_per_node,
                namespace=namespace,
                name=manifest_name,
                image=image,
                hf_secret=hf_secret or None,
            )
            if len(combinations) > 1 and not name:
                base_name = manifest["metadata"]["name"]
                suffix = f"-{index}"
                manifest["metadata"]["name"] = (
                    base_name[: 63 - len(suffix)].rstrip("-") + suffix
                )
            entry["manifest"] = manifest
            if check_cluster:
                cluster_valid, cluster_output = kubernetes_server_dry_run(manifest)
                entry["cluster_validation"] = {
                    "valid": cluster_valid,
                    "output": cluster_output,
                }
                if not cluster_valid:
                    entry["valid"] = False
        all_valid = all_valid and bool(entry["valid"])
        report["configurations"].append(entry)

    report["valid"] = all_valid
    report["mode"] = "static"
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
        click.echo(f"Wrote Dynamo report to {output}")
    else:
        click.echo(rendered)
    if not all_valid:
        raise click.exceptions.Exit(1)


@cli.command("dynamo-optimize")
@click.option("--model", required=True, help="Hugging Face model ID or model path.")
@click.option(
    "--framework",
    required=True,
    type=click.Choice(["vllm", "sglang", "trtllm"]),
)
@click.option("--node-count", type=click.IntRange(min=1), required=True)
@click.option("--gpus-per-node", type=click.IntRange(min=1), required=True)
@click.option("--gpu-system", required=True, help="For example h100_sxm or h200_sxm.")
@click.option("--backend-version", default=None, help="Backend version modeled by AIC.")
@click.option(
    "--input-len", type=click.IntRange(min=1), default=4000, show_default=True
)
@click.option(
    "--output-len", type=click.IntRange(min=1), default=1000, show_default=True
)
@click.option("--ttft", type=click.FloatRange(min=0, min_open=True), default=2000.0)
@click.option("--tpot", type=click.FloatRange(min=0, min_open=True), default=30.0)
@click.option("--namespace", default="dynamo", show_default=True)
@click.option(
    "--artifacts-dir",
    type=click.Path(file_okay=False, path_type=pathlib.Path),
    default=None,
    help="Directory for ranked CSVs, Dynamo manifests, and AIPerf artifacts.",
)
@click.option(
    "--aiconfigurator-command",
    default="aiconfigurator",
    show_default=True,
    help="AI Configurator executable, including an absolute path if needed.",
)
@click.option(
    "--timeout", type=click.IntRange(min=1), default=600, show_default=True
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=pathlib.Path),
    default=None,
    help="Write the AI Configurator JSON report to this path.",
)
@click.option(
    "--check-cluster",
    is_flag=True,
    help="Validate AI Configurator-generated manifests with Kubernetes.",
)
def dynamo_optimize(
    model,
    framework,
    node_count,
    gpus_per_node,
    gpu_system,
    backend_version,
    input_len,
    output_len,
    ttft,
    tpot,
    namespace,
    artifacts_dir,
    aiconfigurator_command,
    timeout,
    output,
    check_cluster,
):
    """Run NVIDIA AI Configurator's GPU-free Dynamo sizing workflow."""
    from llm_optimizer.dynamo import (
        AIConfiguratorError,
        AIConfiguratorRequest,
        kubernetes_file_server_dry_run,
        run_ai_configurator,
    )

    if artifacts_dir is None:
        if output:
            artifacts_dir = output.parent / f"{output.stem}-artifacts"
        else:
            artifacts_dir = pathlib.Path("results") / f"{framework}-aiconfigurator"
    request = AIConfiguratorRequest(
        model=model,
        framework=framework,
        gpu_system=gpu_system,
        total_gpus=node_count * gpus_per_node,
        input_length=input_len,
        output_length=output_len,
        ttft_ms=ttft,
        tpot_ms=tpot,
        backend_version=backend_version,
        namespace=namespace,
    )
    try:
        result = run_ai_configurator(
            request,
            str(artifacts_dir),
            executable=aiconfigurator_command,
            timeout_seconds=timeout,
        )
    except (AIConfiguratorError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    cluster_checks = []
    valid = True
    if check_cluster:
        for artifact in result["artifacts"]:
            if not artifact.endswith("k8s_deploy.yaml"):
                continue
            cluster_valid, cluster_output = kubernetes_file_server_dry_run(
                str(artifacts_dir / artifact)
            )
            cluster_checks.append(
                {
                    "artifact": artifact,
                    "valid": cluster_valid,
                    "output": cluster_output,
                }
            )
            valid = valid and cluster_valid
    report = {
        "valid": valid,
        "mode": "ai-configurator",
        "model": model,
        "framework": framework,
        "topology": {
            "node_count": node_count,
            "gpus_per_node": gpus_per_node,
            "total_gpus": node_count * gpus_per_node,
        },
        "ai_configurator": result,
    }
    if check_cluster:
        report["cluster_validation"] = cluster_checks
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
        click.echo(f"Wrote AI Configurator report to {output}")
    else:
        click.echo(rendered)
    if not valid:
        raise click.exceptions.Exit(1)


@cli.command("estimate")
@click.option(
    "--model",
    type=str,
    required=False,  # Made optional for interactive mode
    help="HuggingFace model ID (e.g., 'meta-llama/Meta-Llama-3-8B')",
)
@click.option(
    "--input-len", type=int, required=False, help="Input sequence length in tokens"
)
@click.option(
    "--output-len", type=int, required=False, help="Output sequence length to generate"
)
@click.option(
    "--gpu",
    type=click.Choice(list_available_gpus_with_lowercase(), case_sensitive=False),
    help="GPU model (auto-detected if not specified, case-insensitive)",
)
@click.option(
    "--num-gpus", type=int, help="Number of GPUs (auto-detected if not specified)"
)
@click.option(
    "--precision",
    type=click.Choice(["fp16", "bf16", "fp8"]),
    default=None,
    help="Model precision (auto-inferred from model config if not specified)",
)
@click.option(
    "--framework",
    type=click.Choice(["sglang", "vllm", "trtllm", "both"]),
    default="both",
    help="Framework to optimize for",
)
@click.option(
    "--constraints", type=str, help="SLO constraints (e.g., 'ttft<300ms;itl<8.5ms')"
)
@click.option(
    "--target",
    type=click.Choice(["throughput", "latency"]),
    default="throughput",
    help="Optimization target",
)
@click.option("--interactive", is_flag=True, help="Run in interactive mode")
@click.option(
    "--dataset",
    type=click.Choice(["random", "sharegpt"]),
    default="random",
    help="Dataset to use for benchmarking (default: random)",
)
def estimate_performance(
    model,
    input_len,
    output_len,
    gpu,
    num_gpus,
    precision,
    framework,
    constraints,
    target,
    interactive,
    dataset,
):
    """Estimate LLM performance and suggest optimal configurations."""

    # Normalize GPU input if provided via CLI
    if gpu:
        gpu = normalize_gpu_choice(gpu)

    # Validate that required parameters are provided either via CLI or interactive mode
    if not interactive and (not model or input_len is None or output_len is None):
        click.echo(
            "Error: --model, --input-len, and --output-len are required "
            "when not using --interactive mode"
        )
        click.echo(
            "Use --interactive for guided input or provide all required parameters"
        )
        return

    try:
        # Parameter collection phase
        if interactive:
            # Collect interactive parameters
            interactive_params = collect_interactive_parameters()

            # Update parameters with interactive input
            model = model or interactive_params["model"]
            input_len = (
                input_len if input_len is not None else interactive_params["input_len"]
            )
            output_len = (
                output_len
                if output_len is not None
                else interactive_params["output_len"]
            )
            target = target or interactive_params["target"]
            constraints = constraints or interactive_params["constraints"]
            precision = precision or interactive_params["precision"]
            framework = framework or interactive_params["framework"]
            dataset = dataset or interactive_params["dataset"]

        # GPU configuration (both modes may need this)
        if interactive or not gpu or not num_gpus:
            gpu, num_gpus = collect_gpu_configuration(
                interactive=interactive, gpu=gpu, num_gpus=num_gpus
            )

        # Build parameters for common estimation function
        params = PerformanceEstimationParams(
            model=model,
            input_len=input_len,
            output_len=output_len,
            gpu=gpu,
            num_gpus=num_gpus,
            precision=precision,
            framework=framework,
            constraints=constraints,
            target=target,
            dataset=dataset,
        )

        # Run common estimation function
        updated_params, result = run_performance_estimation(params)

        # Display results consistently
        display_performance_estimation_results(updated_params, result)

    except Exception as e:
        click.echo(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    cli()
