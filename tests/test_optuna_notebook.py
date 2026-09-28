"""CPU checks for the notebook's search contract and native benchmark adapter."""

import ast
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
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from click.testing import CliRunner
from scipy.stats import spearmanr
from scipy.stats import t as student_t

optuna = pytest.importorskip("optuna")
optuna.logging.set_verbosity(optuna.logging.ERROR)


@pytest.mark.parametrize("target", ["e2e", "pareto_latency"])
def test_notebook_wandb_logs_study_and_objective_vectors(lab, target):
    run = MagicMock(id="test-run", url="https://example.invalid/test-run")
    run.summary = {}
    sdk = SimpleNamespace(init=MagicMock(return_value=run))
    lab.update(WANDB_ENABLED=True, WANDB_MODE="offline")
    with patch.dict("sys.modules", {"wandb": sdk}):
        study = lab["run_study"](
            "wandb-vector-" + target,
            "vllm",
            "random",
            concurrency=1,
            target=target,
            trials=2,
        )
    config = sdk.init.call_args.kwargs["config"]
    assert config["phase"] == "A"
    assert config["workload"]["num_prompts"] == lab["PHASE_REQUESTS"]["A"]
    assert config["environment"]["measurement"] == "SYNTHETIC"
    assert sdk.init.call_args.kwargs["mode"] == "offline"
    assert run.log.call_count == 2
    for call, trial in zip(run.log.call_args_list, study.trials):
        payload = call.args[0]
        assert call.kwargs["step"] == trial.number
        assert payload["trial/state"] == trial.state.name
        assert payload["trial/config"] == trial.user_attrs["config"]
        for metric, value in zip(config["metrics"], trial.values):
            assert payload[f"objective/{metric}"] == value
    assert study.user_attrs["wandb"]["id"] == "test-run"
    expected = study.best_trials
    assert {t["number"] for t in run.summary["best_or_pareto_trials"]} == {
        t.number for t in expected
    }
    assert run.summary["interrupted"] is False
    run.finish.assert_called_once()


def test_notebook_wandb_disabled_does_not_initialize(lab):
    sdk = SimpleNamespace(init=MagicMock())
    with patch.dict("sys.modules", {"wandb": sdk}):
        lab["run_study"]("wandb-disabled", "vllm", "random", concurrency=1, trials=1)
    sdk.init.assert_not_called()


def test_notebook_wandb_failures_are_isolated(lab):
    lab["WANDB_ENABLED"] = True
    sdk = SimpleNamespace(init=MagicMock(side_effect=RuntimeError("secret")))
    with (
        patch.dict("sys.modules", {"wandb": sdk}),
        pytest.warns(UserWarning, match="initialization failed"),
    ):
        study = lab["run_study"](
            "wandb-init-failure", "vllm", "random", concurrency=1, trials=1
        )
    assert study.trials[0].state == optuna.trial.TrialState.COMPLETE
    run = MagicMock(id="fake", url=None)
    run.log.side_effect = RuntimeError("secret")
    run.summary.update.side_effect = RuntimeError("secret")
    run.finish.side_effect = RuntimeError("secret")
    sdk.init = MagicMock(return_value=run)
    with patch.dict("sys.modules", {"wandb": sdk}), pytest.warns(UserWarning):
        study = lab["run_study"](
            "wandb-log-failure", "vllm", "random", concurrency=1, trials=1
        )
    assert study.trials[0].state == optuna.trial.TrialState.COMPLETE
    run.finish.assert_called_once()


@pytest.mark.parametrize("failure", ["failed", "pruned", "interrupted"])
def test_notebook_wandb_failure_pruning_and_shutdown(lab, failure):
    run = MagicMock(id="fake", url=None)
    run.summary = {}
    sdk = SimpleNamespace(init=MagicMock(return_value=run))
    lab["WANDB_ENABLED"] = True
    benchmark = lab["Simulator"]()
    if failure == "failed":
        exc = lab["BenchmarkFailure"]("partial requests")
        exc.request_accounting = {"completed": 2, "failed": 1, "total_requests": 3}
        benchmark.measure = MagicMock(side_effect=exc)
    elif failure == "interrupted":
        benchmark.measure = MagicMock(side_effect=KeyboardInterrupt())
    kwargs = {}
    if failure == "pruned":
        kwargs["pruner"] = optuna.pruners.ThresholdPruner(upper=0)
    with patch.dict("sys.modules", {"wandb": sdk}):
        if failure == "interrupted":
            with pytest.raises(KeyboardInterrupt):
                lab["run_study"](
                    "wandb-interrupted",
                    "vllm",
                    "random",
                    concurrency=1,
                    trials=1,
                    benchmark=benchmark,
                )
        else:
            study = lab["run_study"](
                "wandb-" + failure,
                "vllm",
                "random",
                concurrency=1,
                trials=1,
                benchmark=benchmark,
                **kwargs,
            )
            payload = run.log.call_args.args[0]
            assert payload["trial/state"] == (
                "FAIL" if failure == "failed" else "PRUNED"
            )
            assert study.trials[0].values is None or failure == "pruned"
            if failure == "failed":
                assert payload["requests/failed"] == 1
                assert payload["trial/failure_reason"] == "partial requests"
            else:
                assert payload["trial/intermediate_values"]
                assert payload["trial/pruned_at_fraction"] < 1
    assert run.summary["interrupted"] == (failure == "interrupted")
    run.finish.assert_called_once()


@pytest.fixture
def lab(tmp_path):
    """Load definitions, without plotting, expensive searches, or optional notebooks."""
    path = Path(__file__).resolve().parents[1] / "docs"
    notebook = json.loads(
        (path / "optuna_inference_optimization_lab.ipynb").read_text()
    )
    namespace = {
        "__name__": __name__,
        "display": lambda value: None,
        "hashlib": hashlib,
        "importlib": importlib,
        "json": json,
        "math": math,
        "platform": platform,
        "random": random,
        "shlex": shlex,
        "time": time,
        "warnings": warnings,
        "Counter": Counter,
        "closing": closing,
        "dataclass": dataclass,
        "Path": Path,
        "np": np,
        "optuna": optuna,
        "TrialState": optuna.trial.TrialState,
        "student_t": student_t,
        "spearmanr": spearmanr,
        "RUN_DIR": tmp_path,
        "STORAGE": "sqlite:///" + str(tmp_path / "study.sqlite3"),
    }
    constants = {
        "FRAMEWORKS",
        "CONCURRENCIES",
        "SEEDS",
        "QUICK_DEMO",
        "SAMPLE_BUDGETS",
        "DEMO_SAMPLE_BUDGETS",
        "ACTIVE_SAMPLE_BUDGETS",
        "TTFT_LIMITS_MS",
        "FAILURE_RATE_LIMIT",
        "BASE_WORKLOAD",
        "ENVIRONMENT",
        "SAMPLER_NAMES",
        "RUN_GP",
        "ARM_AVAILABILITY",
        "PREFILL_BUDGETS_1",
        "TOKEN_BUDGETS_128",
        "CAPACITIES_128",
        "SPACES",
        "FLOAT_PARAMS",
        "PHASE_MODELS",
        "PHASE_B_TRIALS",
        "EXPANDED_CHOICES",
        "PHASE_REQUESTS",
        "VALIDATION_REQUESTS",
        "VALIDATION_REPEATS",
        "E2E_LIMITS_MS",
        "SEARCH_TRIALS",
        "DEMO_TRIALS",
        "ACTIVE_SEEDS",
        "PARETO_SAMPLERS",
        "RUN_REAL_GPU",
        "REAL_FRAMEWORKS",
        "STUDIES",
        "SKIPPED_ARMS",
        "GROUP_RUNS",
        "MODEL_SNAPSHOTS",
        "REAL_PREFLIGHT",
        "VALIDATION_REQUIRE_SLO",
        "CONSTRAINED_STUDIES",
        "calibration",
        "WANDB_ENABLED",
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_MODE",
        "WANDB_GROUP",
    }
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        definitions = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
            or (
                isinstance(node, ast.Assign)
                and all(
                    isinstance(t, ast.Name) and t.id in constants for t in node.targets
                )
            )
        ]
        exec(
            compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"),
            namespace,
        )
    return namespace


@pytest.mark.parametrize("concurrency,expected", [(1, 16), (128, 64)])
def test_phase_a_numerical_support_and_rendering(lab, concurrency, expected):
    # A single long prompt can exercise the sampled prefill budgets.
    assert lab["BASE_WORKLOAD"]["random_input_len"] >= max(lab["PREFILL_BUDGETS_1"])
    for framework in lab["FRAMEWORKS"]:
        assert lab["grid_size"](concurrency, framework) == expected
        space = lab["SPACES"][concurrency][framework]
        values = {key: choices[0] for key, choices in space.items()}
        trial = optuna.trial.FixedTrial(values)
        rendered = lab["suggest_config"](trial, framework, concurrency)
        assert trial.params == values
        features = lab["engine_features"](framework, rendered)
        assert features[0] == (1 if concurrency == 1 else 8)
        assert features[3] is (concurrency != 1)
        assert all(
            not isinstance(d, optuna.distributions.CategoricalDistribution)
            for d in trial.distributions.values()
        )
    with pytest.raises(ValueError, match="regular"):
        lab["suggest_numeric"](optuna.trial.FixedTrial({"x": 8}), "x", [8, 16, 32])
    with pytest.raises(ValueError, match="concurrency"):
        lab["suggest_config"](optuna.trial.FixedTrial({}), "vllm", 32)


@pytest.mark.parametrize("concurrency,direction", [(1, "MINIMIZE"), (128, "MINIMIZE")])
def test_runner_separates_workload_direction_and_identity(lab, concurrency, direction):
    study = lab["run_study"](
        f"runner-{concurrency}",
        "vllm",
        "random",
        concurrency=concurrency,
        trials=3,
    )
    assert study.directions[0].name == direction
    contract = study.user_attrs["contract"]
    assert contract["workload"]["max_concurrency"] == concurrency
    scores = lab["convergence"](study)[2]
    assert np.all(np.diff(scores) <= 0)
    config = study.trials[0].user_attrs["config"]
    assert lab["config_fingerprint"]("vllm", 1, config) != lab["config_fingerprint"](
        "vllm",
        128,
        config,
    )
    candidates = lab["shortlist"]([study], k=1)
    row = lab["validate_candidates"](candidates, repeats=2)[0]
    assert row["concurrency"] == concurrency
    assert row["successful_repeats"] == 2


def test_equivalent_float_grid_points_share_configuration_identity(lab):
    space = lab["SPACES"][128]["vllm"]
    configs = []
    for value in [0.85, 0.8500000000000001]:
        sampled = {name: choices[0] for name, choices in space.items()}
        sampled["gpu_memory_utilization"] = value
        config = lab["suggest_config"](optuna.trial.FixedTrial(sampled), "vllm", 128)
        configs.append(config)
    assert configs[0] == configs[1]
    assert lab["config_fingerprint"]("vllm", 128, configs[0]) == lab[
        "config_fingerprint"
    ]("vllm", 128, configs[1])


def test_qmc_shared_initial_point_then_sobol(lab):
    from scipy.stats import qmc

    seed = 17
    study, budget = lab["create_phase_a_study"]("qmc", "vllm", 1, seed)
    assert budget == lab["trial_budget"]("A", 1)
    space = lab["SPACES"][1]["vllm"]
    for _ in range(5):
        trial = study.ask()
        lab["suggest_config"](trial, "vllm", 1)
        study.tell(trial, 100.0)
    assert study.trials[0].params == {
        name: values[len(values) // 2] for name, values in space.items()
    }
    # Optuna 4.9 preserves the first trial's distribution order, not sorted names.
    names = list(study.trials[0].distributions)
    points = qmc.Sobol(d=len(names), scramble=True, seed=seed).random_base2(2)
    for trial, point in zip(study.trials[1:], points):
        expected = {
            name: space[name][min(int(u * len(space[name])), len(space[name]) - 1)]
            for name, u in zip(names, point)
        }
        assert trial.params == expected


@pytest.mark.parametrize("concurrency", [1, 128])
def test_grid_exhausts_expected_space(lab, concurrency):
    study = lab["run_study"](
        f"grid-{concurrency}",
        "vllm",
        "grid",
        concurrency=concurrency,
    )
    assert len(study.trials) == lab["grid_size"](concurrency, "vllm")
    assert len({t.user_attrs["config_id"] for t in study.trials}) == len(study.trials)


def test_native_nested_serialization(lab):
    import yaml

    for graph_on in [0, 1]:
        cfg = lab["render_engine_config"](
            "vllm",
            1,
            {"max_num_batched_tokens": 2048, "cuda_graph_on": graph_on},
        )
        native = lab["serialize_engine_config"]("vllm", 1, cfg)
        assert json.loads(native["compilation_config"]) == cfg["compilation_config"]
        cfg = lab["render_engine_config"](
            "trtllm",
            1,
            {"max_num_tokens": 2048, "cuda_graph_on": graph_on},
        )
        native = lab["serialize_engine_config"]("trtllm", 1, cfg)
        parsed = yaml.safe_load(Path(native["config"]).read_text())
        assert parsed["cuda_graph_config"] == cfg["cuda_graph_config"]
        assert parsed["kv_cache_config"] == cfg["kv_cache_config"]
    native = lab["serialize_engine_config"](
        "sglang", 128, {"disable_cuda_graph": False}
    )
    assert "disable_cuda_graph" not in native
    with pytest.raises(lab["BenchmarkFailure"], match="False booleans"):
        lab["serialize_engine_config"]("vllm", 1, {"enable_chunked_prefill": False})


@pytest.mark.parametrize("framework", ["vllm", "sglang", "trtllm"])
@pytest.mark.parametrize("concurrency", [1, 128])
@pytest.mark.parametrize("space_kind", ["baseline", "expanded"])
def test_measured_adapter_propagates_regime_and_cleans_up(
    lab, framework, concurrency, space_kind, tmp_path
):
    import llm_optimizer.bench_client as client
    import llm_optimizer.cli as cli_module

    seen = []
    expected = lab["PHASE_REQUESTS"]["B" if space_kind == "expanded" else "A"]

    def prepare(args):
        seen.append(args)
        return {"requests": [], "sha256": "test"}, {"workload_sha256": "test"}

    environment = {
        "framework_version": "mock",
        "cuda_version": "mock",
        "driver_version": "mock",
        "model_revision": "pinned-mock",
        "native_config_verified": True,
    }
    with patch.object(client, "prepare_workload_manifest", side_effect=prepare):
        benchmark = lab["RepositoryBenchmark"](
            framework,
            str(tmp_path),
            "mock-gpu",
            concurrency,
            rest_seconds=0,
            environment=environment,
            phase="B" if space_kind == "expanded" else "A",
        )
    assert seen[0]["max_concurrency"] == concurrency
    sampled = {
        key: values[0] for key, values in lab["SPACES"][concurrency][framework].items()
    }
    if space_kind == "expanded":
        cfg = lab["suggest_config"](
            optuna.trial.FixedTrial(expanded_values(lab, framework, concurrency)),
            framework,
            concurrency,
            space_kind,
        )
    else:
        cfg = lab["render_engine_config"](framework, concurrency, sampled)
    metrics = {
        "output_throughput": 1000.0,
        "p95_ttft_ms": 80.0,
        "p95_e2e_latency_ms": 400.0,
        "total_requests": expected,
        "completed": expected,
        "failed": 0,
        "benchmark_valid": True,
    }
    with (
        patch.object(cli_module, "start_server", return_value=object()) as start,
        patch.object(cli_module, "terminate_process_top_down") as terminate,
        patch.object(client, "run_benchmark", return_value=metrics) as run,
    ):
        result = benchmark.measure(framework, cfg, concurrency)
        assert result.source == "MEASURED_ALLOCATION_WALL_PROXY"
        assert run.call_args.args[0]["max_concurrency"] == concurrency
        assert start.call_count == terminate.call_count == 1
        argv = shlex.split(start.call_args.args[0])
        if framework == "vllm":
            flag = next(a for a in argv if a.startswith("--compilation-config="))
            assert json.loads(flag.split("=", 1)[1]) == cfg["compilation_config"]
            if space_kind == "expanded":
                assert "--no-async-scheduling" in argv
                assert "--no-enable-prefix-caching" in argv
                assert "--performance-mode=balanced" in argv
                assert "--attention-backend=auto" in argv
        elif framework == "sglang" and space_kind == "expanded":
            offset = argv.index("--cuda-graph-bs")
            assert argv[offset + 1 : offset + 1 + len(cfg["cuda_graph_bs"])] == [
                str(s) for s in cfg["cuda_graph_bs"]
            ]
        elif framework == "trtllm" and space_kind == "expanded":
            import yaml

            flag = next(a for a in argv if a.startswith("--config="))
            native = yaml.safe_load(Path(flag.split("=", 1)[1]).read_text())
            assert native["enable_chunked_prefill"] is False
            assert native["kv_cache_config"]["enable_block_reuse"] is False
            assert native["scheduler_config"] == cfg["scheduler_config"]
            assert native["attn_backend"] == cfg["attn_backend"]
    with (
        patch.object(cli_module, "start_server", return_value=object()),
        patch.object(cli_module, "terminate_process_top_down") as terminate,
        patch.object(client, "run_benchmark", side_effect=RuntimeError("mock OOM")),
    ):
        with pytest.raises(lab["BenchmarkFailure"], match="mock OOM") as failed:
            benchmark.measure(framework, cfg, concurrency)
        assert failed.value.gpu_seconds > 0
        terminate.assert_called_once()
    with pytest.raises(ValueError, match="differs"):
        benchmark.measure(framework, cfg, 128 if concurrency == 1 else 1)
    with patch.object(cli_module, "detect_gpu_type", return_value="mock-gpu"):
        command = lab["cli_dry_run_command"](framework, concurrency)
        result = CliRunner().invoke(cli_module.cli, command[1:])
    assert result.exit_code == 0, result.output
    assert "Optuna dry run mode enabled." in result.output


def expanded_values(lab, framework, concurrency, graphs=True):
    values = {
        key: choices[0]
        for key, choices in lab["SPACES"][concurrency][framework].items()
    }
    values.update(
        {key: choices[0] for key, choices in lab["EXPANDED_CHOICES"][framework].items()}
    )
    memory = {
        "vllm": "gpu_memory_utilization",
        "sglang": "mem_fraction_static",
        "trtllm": "kv_cache_free_gpu_memory_fraction",
    }[framework]
    values[memory] = lab["SPACES"][128][framework][memory][0]
    values.update(
        graph_mode="PIECEWISE" if graphs else "NONE",
        graph_enabled=graphs,
        graph_capture_profile="dense",
        graph_padding=0,
        graph_padding_off=1,
        parallel_prefill=1,
        max_num_partial_prefills=3,
        schedule_conservativeness=0.75,
        max_prefill_tokens=16384,
    )
    return values


@pytest.mark.parametrize("framework", ["vllm", "sglang", "trtllm"])
@pytest.mark.parametrize("concurrency", [1, 128])
@pytest.mark.parametrize("graphs", [False, True])
def test_expanded_coverage_and_conditional_graphs(lab, framework, concurrency, graphs):
    trial = optuna.trial.FixedTrial(
        expanded_values(lab, framework, concurrency, graphs)
    )
    cfg = lab["suggest_config"](trial, framework, concurrency, "expanded")
    assert "cuda_graph_on" not in trial.params
    assert ("graph_capture_profile" in trial.params) is graphs
    for name in lab["EXPANDED_CHOICES"][framework]:
        assert name in trial.params
    capacity, _, _, enabled = lab["engine_features"](framework, cfg)
    assert enabled is graphs
    if framework == "vllm":
        assert cfg["max_num_partial_prefills"] == 3
        if graphs:
            assert cfg["compilation_config"]["cudagraph_capture_sizes"] == list(
                range(1, capacity + 1)
            )
    elif framework == "sglang":
        assert cfg["schedule_conservativeness"] == 0.75
        assert cfg["max_prefill_tokens"] == 16384
    else:
        assert cfg["max_num_tokens"] >= lab["BASE_WORKLOAD"]["random_input_len"]
        assert cfg["kv_cache_config"]["enable_block_reuse"] is False


def test_phase_model_isolation_and_filtered_choices(lab):
    cfg = {"identical": True}
    assert lab["config_fingerprint"]("vllm", 1, cfg, "A") != lab["config_fingerprint"](
        "vllm", 1, cfg, "B"
    )
    studies = [
        lab["run_study"](
            "model-" + phase, "vllm", "random", concurrency=1, trials=2, phase=phase
        )
        for phase in ["A", "B"]
    ]
    assert studies[0].user_attrs["contract"]["target_model"] == "Qwen/Qwen3.5-0.8B"
    assert studies[1].user_attrs["contract"]["target_model"] == "Qwen/Qwen3.5-4B"
    rows = lab["validate_candidates"](lab["shortlist"](studies, k=1), repeats=2)
    assert {r["phase"] for r in rows} == {"A", "B"}
    with pytest.raises(ValueError, match="phase/model"):
        lab["run_study"](
            "wrong-model",
            "vllm",
            phase="B",
            benchmark=lab["Simulator"](phase="A"),
            trials=1,
        )
    filtered = {k: [v[0]] for k, v in lab["EXPANDED_CHOICES"]["vllm"].items()}
    study = lab["run_study"](
        "filtered",
        "vllm",
        "random",
        concurrency=1,
        space_kind="expanded",
        choices=filtered,
        trials=2,
    )
    assert study.user_attrs["contract"]["phase"] == "B"
    assert study.user_attrs["contract"]["expanded_choices"] == filtered
    for trial in study.trials:
        for name, values in filtered.items():
            assert trial.params[name] in values
    with pytest.raises(ValueError, match="every"):
        lab["expanded_choices"]("vllm", {"attention_backend": ["auto"]})


@pytest.mark.parametrize("concurrency,target", [(1, "ttft"), (128, "throughput")])
def test_pruning_reports_correct_metric_and_closes_generator(lab, concurrency, target):
    class Streaming:
        supports_checkpoints = True
        closed = False

        def stream(self, framework, config, regime):
            try:
                assert regime == concurrency
                yield (
                    1,
                    0.25,
                    lab["Measurement"](500, 150, 10, 10, 0, 1, 1, "mock", 400),
                    1,
                )
            finally:
                self.closed = True

    benchmark = Streaming()
    kwargs = {"upper": 100} if target == "ttft" else {"lower": 600}
    study = lab["run_study"](
        f"prune-{concurrency}",
        "vllm",
        "random",
        concurrency=concurrency,
        trials=1,
        benchmark=benchmark,
        target=target,
        pruner=optuna.pruners.ThresholdPruner(**kwargs),
    )
    assert study.trials[0].state == optuna.trial.TrialState.PRUNED
    assert study.trials[0].intermediate_values[1] == (150 if target == "ttft" else 500)
    assert benchmark.closed


def test_pareto_and_constraint_selection(lab):
    study = lab["run_study"](
        "pareto", "vllm", "nsga2", concurrency=1, target="pareto", trials=9
    )
    assert [d.name for d in study.directions] == ["MINIMIZE", "MINIMIZE"]
    assert {
        t.number for t in lab["pareto_front"](lab["completed"](study), study.directions)
    } == {t.number for t in study.best_trials}
    scalar = lab["run_study"](
        "constrained",
        "vllm",
        "tpe",
        concurrency=128,
        target="throughput",
        constrained=True,
        trials=3,
    )
    best = lab["choose_single"](scalar, True)
    assert best is None or best.user_attrs["slo_feasible"]
    with pytest.raises(ValueError, match="constraint-aware"):
        lab["make_sampler"]("qmc", "vllm", 128, constrained=True)


OBJECTIVE_CASES = [
    (phase, concurrency, target, sampler)
    for phase in ["A", "B"]
    for concurrency, targets in [
        (1, ["e2e", "ttft", "pareto_latency"]),
        (128, ["e2e", "throughput", "pareto_throughput"]),
    ]
    for target in targets
    for sampler in (
        ["grid", "random", "qmc", "tpe", "tpe_multivariate", "gp", "nsga2"]
        if target.startswith("pareto")
        else ["grid", "random", "qmc", "tpe", "tpe_multivariate", "gp"]
    )
]


@pytest.mark.parametrize("phase,concurrency,target,sampler", OBJECTIVE_CASES)
def test_every_objective_and_sampler_contract(lab, phase, concurrency, target, sampler):
    if sampler == "gp" and importlib.util.find_spec("torch") is None:
        pytest.skip("Optional GPSampler dependency torch is unavailable")
    study = lab["run_study"](
        f"matrix-{phase}-{concurrency}-{target}-{sampler}",
        "vllm",
        sampler,
        concurrency=concurrency,
        phase=phase,
        target=target,
        space_kind="baseline" if phase == "A" else "expanded",
        trials=2,
    )
    contract = study.user_attrs["contract"]
    _, metrics, directions = lab["objective_spec"](target, concurrency)
    assert [d.name for d in study.directions] == [d.upper() for d in directions]
    assert contract["workload"]["num_prompts"] == lab["PHASE_REQUESTS"][phase]
    assert contract["workload"]["max_concurrency"] == concurrency
    assert contract["metrics"] == metrics
    for trial in study.trials:
        assert trial.state == optuna.trial.TrialState.COMPLETE
        assert trial.values == [trial.user_attrs["measurement"][m] for m in metrics]
    if target.startswith("pareto"):
        assert {
            t.number for t in lab["pareto_front"](study.trials, study.directions)
        } == {t.number for t in study.best_trials}
        with pytest.raises(ValueError, match="scalar objectives"):
            lab["run_study"](
                "bad-prune",
                "vllm",
                concurrency=concurrency,
                target=target,
                trials=1,
                pruner=optuna.pruners.MedianPruner(),
            )
    if sampler == "grid" and phase == "B":
        assert not lab["summarize"](study)["grid_exhausted"]
        assert contract["grid_declared_points"] > len(study.trials)
        assert isinstance(study.sampler._all_grids, lab["CartesianPoints"])


def test_lazy_grid_is_bijective_and_exhausts_small_cartesian_menu(lab):
    menu = {"x": [0, 1, 2], "y": [False, True]}
    points = lab["CartesianPoints"](menu, 42)
    assert len({points[i] for i in range(len(points))}) == 6
    assert all(points[i] == lab["CartesianPoints"](menu, 42)[i] for i in range(6))
    sampler = lab["LazyGridSampler"](menu, 42)
    study = optuna.create_study(sampler=sampler, directions=["minimize", "minimize"])
    study.optimize(
        lambda t: (t.suggest_int("x", 0, 2), t.suggest_categorical("y", [False, True]))
    )
    assert sampler.is_exhausted(study)
    assert len(study.trials) == 6


@pytest.mark.parametrize("phase", ["A", "B"])
@pytest.mark.parametrize(
    "target,metric,value,kwargs",
    [
        ("e2e", "p95_e2e_latency_ms", 400, {"upper": 300}),
        ("ttft", "p95_ttft_ms", 150, {"upper": 100}),
        ("throughput", "output_throughput", 500, {"lower": 600}),
    ],
)
def test_pruning_matches_objective_not_concurrency(
    lab, phase, target, metric, value, kwargs
):
    concurrency = 128 if target == "throughput" else 1

    class Streaming:
        supports_checkpoints = True
        closed = False

        def stream(self, framework, config, concurrency):
            try:
                yield (
                    1,
                    0.25,
                    lab["Measurement"](500, 150, 10, 10, 0, 1, 1, "mock", 400),
                    1,
                )
            finally:
                self.closed = True

    benchmark = Streaming()
    study = lab["run_study"](
        f"prune-{phase}-{target}",
        "vllm",
        "random",
        concurrency=concurrency,
        phase=phase,
        target=target,
        benchmark=benchmark,
        trials=1,
        space_kind="baseline" if phase == "A" else "expanded",
        pruner=optuna.pruners.ThresholdPruner(**kwargs),
    )
    assert study.trials[0].state == optuna.trial.TrialState.PRUNED
    assert study.trials[0].intermediate_values[1] == value
    assert study.user_attrs["contract"]["metrics"] == [metric]
    assert benchmark.closed
    menu = lab["pruner_menu"](
        "maximize" if target == "throughput" else "minimize", value
    )
    assert menu["percentile"][1]["percentile"] == (25 if target == "throughput" else 75)
    assert ("lower" if target == "throughput" else "upper") in menu["threshold"][1]


def test_phase_request_counts_and_arbitrary_loaded_concurrency(lab):
    lab["CONCURRENCIES"] = (1, 16, 64)
    lab["PHASE_REQUESTS"] = {"A": 17, "B": 25}
    for phase in ["A", "B"]:
        for concurrency in lab["CONCURRENCIES"]:
            study = lab["run_study"](
                f"counts-{phase}-{concurrency}",
                "vllm",
                "random",
                phase=phase,
                concurrency=concurrency,
                target="e2e",
                trials=1,
            )
            assert (
                study.trials[0].user_attrs["measurement"]["total_requests"]
                == lab["PHASE_REQUESTS"][phase]
            )
            assert (
                study.user_attrs["contract"]["workload"]["max_concurrency"]
                == concurrency
            )
    study = lab["run_study"](
        "count-override", "vllm", "random", concurrency=16, trials=1, num_prompts=43
    )
    assert study.trials[0].user_attrs["measurement"]["total_requests"] == 43
    for count in [0, -1, True, 1.5]:
        with pytest.raises(ValueError, match="num_prompts"):
            lab["workload_for"](16, "A", count)


def controlled_candidates(lab, target, concurrency=1):
    base = lab["run_study"](
        "contract-" + target,
        "vllm",
        "random",
        concurrency=concurrency,
        target=target,
        trials=1,
    )
    study = optuna.create_study(directions=base.directions)
    study.set_user_attr("contract", base.user_attrs["contract"])
    metrics = base.user_attrs["contract"]["metrics"]
    for tag, numbers in [
        (
            "fast_e2e",
            {"output_throughput": 200, "p95_ttft_ms": 80, "p95_e2e_latency_ms": 100},
        ),
        (
            "fast_ttft",
            {"output_throughput": 400, "p95_ttft_ms": 20, "p95_e2e_latency_ms": 200},
        ),
    ]:
        study.add_trial(
            optuna.trial.create_trial(
                values=[numbers[m] for m in metrics],
                user_attrs={"config": {"tag": tag}, "slo_feasible": True},
            )
        )
    return lab["shortlist"]([study])


@pytest.mark.parametrize(
    "target,concurrency,expected",
    [
        ("e2e", 1, "fast_e2e"),
        ("ttft", 1, "fast_ttft"),
        ("throughput", 128, "fast_ttft"),
        ("pareto_latency", 1, None),
        ("pareto_throughput", 128, None),
    ],
)
def test_independent_validation_selects_matching_scalar_or_front(
    lab, target, concurrency, expected
):
    seen = []

    def factory(framework, concurrency, repeat, phase, count, seed):
        seen.append((repeat, phase, count, seed))

        class Benchmark:
            def measure(self, framework, config, concurrency):
                fast = config["tag"] == "fast_e2e"
                return lab["Measurement"](
                    200 if fast else 400,
                    80 if fast else 20,
                    count,
                    count,
                    0,
                    1,
                    1,
                    "mock",
                    100 if fast else 200,
                )

        return Benchmark()

    candidates = controlled_candidates(lab, target, concurrency)
    rows = lab["validate_candidates"](candidates, repeats=3, benchmark_factory=factory)
    selected = next(iter(lab["validation_recommendations"](rows).values()))
    if expected:
        assert selected["config"]["tag"] == expected
    else:
        assert {row["config"]["tag"] for row in selected} == {"fast_e2e", "fast_ttft"}
    assert all(item[2] == lab["VALIDATION_REQUESTS"]["A"] for item in seen)
    assert {item[3] for item in seen} == {50000, 50001, 50002}
    assert len({row["task_id"] for row in rows}) == 1
    assert all(
        len(row["statistics"]["p95_e2e_latency_ms"]["replicates"]) == 3 for row in rows
    )


def test_failed_trial_and_validation_repeats_cannot_win(lab):
    count = lab["PHASE_REQUESTS"]["A"]

    class Partial:
        supports_checkpoints = False

        def measure(self, framework, config, concurrency):
            return lab["Measurement"](99999, 1, count - 1, count, 1, 2, 2, "mock", 1)

    study = lab["run_study"](
        "partial", "vllm", "random", target="e2e", trials=1, benchmark=Partial()
    )
    trial = study.trials[0]
    assert trial.state == optuna.trial.TrialState.FAIL
    assert "Incomplete" in trial.user_attrs["failure_reason"]
    assert trial.user_attrs["request_accounting"]["failed"] == 1
    assert trial.user_attrs["gpu_seconds"] == 2

    def factory(framework, concurrency, repeat, phase, count, seed):
        class Benchmark:
            def measure(self, framework, config, concurrency):
                failed = int(config["tag"] == "fast_e2e" and repeat == 1)
                return lab["Measurement"](
                    200,
                    20,
                    count - failed,
                    count,
                    failed,
                    2,
                    2,
                    "mock",
                    1 if config["tag"] == "fast_e2e" else 200,
                )

        return Benchmark()

    rows = lab["validate_candidates"](
        controlled_candidates(lab, "e2e"), repeats=2, benchmark_factory=factory
    )
    selected = next(iter(lab["validation_recommendations"](rows).values()))
    assert selected["config"]["tag"] == "fast_ttft"
    failed = next(row for row in rows if row["config"]["tag"] == "fast_e2e")
    assert not failed["validation_complete"]
    assert failed["failures"][0]["request_accounting"]["failed"] == 1


def test_objective_running_cells_are_separate_and_complete(lab):
    path = (
        Path(__file__).resolve().parents[1]
        / "docs/optuna_inference_optimization_lab.ipynb"
    )
    notebook = json.loads(path.read_text())
    calls = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        local = [
            node
            for node in ast.walk(ast.parse("".join(cell["source"])))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "run_objective_group"
        ]
        assert len(local) <= 1
        for node in local:
            calls.append(
                (ast.literal_eval(node.args[0]), ast.literal_eval(node.args[1]))
            )
    assert len(calls) == 12
    for phase in ["A", "B"]:
        assert Counter(target for p, target in calls if p == phase) == Counter(
            e2e=2, ttft=1, throughput=1, pareto_latency=1, pareto_throughput=1
        )


def test_group_dispatches_all_scalar_samplers_and_records_optional_skips(lab):
    lab["FRAMEWORKS"] = ("vllm",)
    lab["ACTIVE_SEEDS"] = [11]
    lab["ARM_AVAILABILITY"]["gp"]["enabled"] = False
    group = lab["run_objective_group"]("A", "e2e", [1])
    assert {key[4] for key in group} == set(lab["SAMPLER_NAMES"]) - {"gp"}
    assert lab["SKIPPED_ARMS"][0]["sampler"] == "gp"
    assert all(
        study.user_attrs["contract"]["target"] == "e2e" for study in group.values()
    )
    again = lab["run_objective_group"]("A", "e2e", [1])
    assert {s.study_name for s in group.values()}.isdisjoint(
        s.study_name for s in again.values()
    )


def test_pruning_requires_fresh_task_specific_calibration(lab):
    lab["FRAMEWORKS"] = ("vllm",)
    lab["CONCURRENCIES"] = (1,)
    with pytest.raises(ValueError, match="Calibrate"):
        lab["run_pruning_objective"]("e2e")
    cal = lab["calibrate_pruning"]("A", 1, "vllm", "e2e", samples=12)
    assert cal["metric"] == "p95_e2e_latency_ms"
    assert cal["workload"]["num_prompts"] == lab["PHASE_REQUESTS"]["A"]
    lab["calibration"][("A", 1, "vllm", "e2e")] = cal
    lab["PHASE_REQUESTS"]["A"] += 1
    with pytest.raises(ValueError, match="recalibrate"):
        lab["run_pruning_objective"]("e2e")
    assert lab["artifact_value"](cal)["workload"]["request_rate"] == "inf"


def test_constraint_and_objective_identity_do_not_merge(lab):
    scalar = lab["run_study"](
        "unconstrained-identity", "vllm", "tpe", target="e2e", trials=1
    )
    constrained = lab["run_study"](
        "constrained-identity", "vllm", "tpe", target="e2e", constrained=True, trials=1
    )
    assert lab["task_identity"](scalar.user_attrs["contract"]) != lab["task_identity"](
        constrained.user_attrs["contract"]
    )


def test_validation_simulator_uses_reserved_noise_and_request_seeds(lab):
    search = lab["benchmark_for"]("A", "vllm", 1, "search")
    validation = lab["benchmark_for"](
        "A", "vllm", 1, "holdout", num_prompts=1000, workload_seed=50000
    )
    next_repeat = lab["benchmark_for"](
        "A", "vllm", 1, "holdout-1", num_prompts=1000, workload_seed=50001, repeat=1
    )
    assert search.rng.bit_generator.state != validation.rng.bit_generator.state
    assert validation.rng.bit_generator.state != next_repeat.rng.bit_generator.state
    assert validation.workload_seed == 50000
    assert validation.num_prompts == 1000
