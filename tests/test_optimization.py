"""Tests for finite Optuna search helpers and dry-run behavior."""

import json
import sys
import types
from pathlib import Path

import pytest
from click.testing import CliRunner

import llm_optimizer.args as lo_args
from llm_optimizer.cli import cli, construct_benchmark_settings
from llm_optimizer.optimization import (
    OptunaSearchSpace,
    constraint_violations,
    extract_objective,
    objective_direction,
    parse_int_ranges,
    satisfies_constraints,
)
from llm_optimizer.performance import parse_slo_constraints
from llm_optimizer.predefined import CLIENT_CONFIGS, SERVER_CONFIGS


class FakeTrial:
    def __init__(self, indexes):
        self.indexes = indexes
        self.names = []

    def suggest_categorical(self, name, choices):
        self.names.append(name)
        index = self.indexes[name]
        assert index in choices
        return index


def _sets(value, scope):
    return lo_args.parse_args_str(value, scope=scope, configs=None, strict=False)


def test_search_space_preserves_scopes_ranges_and_composites():
    server = _sets(
        "shared=[1,2];tp*dp=[(1,4),(2,2)]", lo_args.ArgScope.SERVER
    )
    client = _sets(
        "shared=range(3,5);flag:bool=[True,False]", lo_args.ArgScope.CLIENT
    )
    space = OptunaSearchSpace(tuple(server + client))
    trial = FakeTrial(
        {
            "server:shared": 1,
            "server:tp*dp": 1,
            "client:shared": 0,
            "client:flag": 1,
        }
    )
    selected = space.suggest(trial)
    assert space.cardinality == 16
    assert trial.names == [
        "server:shared",
        "server:tp*dp",
        "client:shared",
        "client:flag",
    ]
    assert dict(lo_args.get_all_kv_pairs(selected)) == {
        "shared": 3,
        "tp": 2,
        "dp": 2,
        "flag": False,
    }


def test_fixed_arguments_are_not_suggested():
    space = OptunaSearchSpace(tuple(_sets("fixed=7", lo_args.ArgScope.SERVER)))
    trial = FakeTrial({})
    selected = space.suggest(trial)
    assert trial.names == []
    assert dict(lo_args.get_all_kv_pairs(selected)) == {"fixed": 7}


def test_grid_search_space_contains_all_categorical_and_integer_values():
    ranges = parse_int_ranges(
        ["max_num_seqs=int(16,32,8)"],
        lo_args.ArgScope.SERVER,
        SERVER_CONFIGS["vllm"],
    )
    space = OptunaSearchSpace(
        tuple(_sets("tensor_parallel_size=[1,2]", lo_args.ArgScope.SERVER)), ranges
    )
    assert space.grid_search_space() == {
        "server:tensor_parallel_size": [0, 1],
        "server:max_num_seqs": [16, 24, 32],
    }


def test_grid_sampler_accepts_generated_search_space():
    optuna = pytest.importorskip("optuna")
    space = OptunaSearchSpace(tuple(_sets("x=[1,2]", lo_args.ArgScope.SERVER)))
    sampler = optuna.samplers.GridSampler(space.grid_search_space())
    study = optuna.create_study(sampler=sampler, direction="maximize")
    trial = study.ask()
    selected = space.suggest(trial)
    assert dict(lo_args.get_all_kv_pairs(selected))["x"] in {1, 2}


def test_grid_sampler_runs_every_engine_configuration(tmp_path, monkeypatch):
    pytest.importorskip("optuna")
    import types
    import sys
    import llm_optimizer.cli as cli_module

    fake_client = types.ModuleType("llm_optimizer.bench_client")
    fake_client.run_benchmark = lambda args: {"output_throughput": 1.0}
    monkeypatch.setitem(sys.modules, "llm_optimizer.bench_client", fake_client)
    monkeypatch.setattr(cli_module, "start_server", lambda *args: object())
    monkeypatch.setattr(cli_module, "terminate_process_top_down", lambda process: None)
    output = tmp_path / "grid.json"
    result = CliRunner().invoke(
        cli,
        [
            "--framework", "vllm", "--model", "example/model", "--gpus", "1",
            "--server-args", "max_num_seqs=[1,2]",
            "--client-args", "max_concurrency=1",
            "--search-strategy", "optuna", "--optuna-sampler", "GridSampler",
            "--rest", "0", "--output-json", str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(output.read_text())
    assert report["metadata"]["total_tests"] == 2


def test_optuna_prepares_fixed_workload_from_argument_pairs(tmp_path, monkeypatch):
    pytest.importorskip("optuna")
    import types
    import sys
    import llm_optimizer.cli as cli_module

    captured = []
    fake_client = types.ModuleType("llm_optimizer.bench_client")
    fake_client.prepare_workload_manifest = lambda args: (
        captured.append(args) or {"sha256": "test-workload"},
        {"workload_sha256": "test-workload"},
    )
    fake_client.run_benchmark = lambda args: {"output_throughput": 1.0}
    monkeypatch.setitem(sys.modules, "llm_optimizer.bench_client", fake_client)
    monkeypatch.setattr(cli_module, "start_server", lambda *args: object())
    monkeypatch.setattr(cli_module, "terminate_process_top_down", lambda process: None)

    result = CliRunner().invoke(
        cli,
        [
            "--framework", "vllm", "--model", "example/model", "--gpus", "1",
            "--server-args", "max_num_seqs=[8,16];revision=abc123",
            "--client-args", "dataset_name=random;random_input_len=128;random_output_len=32;num_prompts=16",
            "--search-strategy", "optuna", "--optuna-trials", "2",
            "--rest", "0", "--output-json", str(tmp_path / "result.json"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured[0]["dataset_name"] == "random"
    assert captured[0]["num_prompts"] == 16
    assert captured[0]["model_revision"] == "abc123"


def test_optuna_integer_ranges_use_suggest_int_and_preserve_scopes():
    server_ranges = parse_int_ranges(
        ["max_num_seqs=int(16,64);max_num_batched_tokens=int(4096,16384,step=4096)"],
        lo_args.ArgScope.SERVER,
        SERVER_CONFIGS["vllm"],
    )
    client_ranges = parse_int_ranges(
        ["max_concurrency=int(8,32,8)"],
        lo_args.ArgScope.CLIENT,
        CLIENT_CONFIGS,
    )
    space = OptunaSearchSpace((), server_ranges + client_ranges)

    class IntegerTrial:
        def __init__(self):
            self.suggestions = []

        def suggest_int(self, name, low, high, *, step):
            self.suggestions.append((name, low, high, step))
            return high

    trial = IntegerTrial()
    selected = space.suggest(trial)
    assert space.cardinality == 49 * 4 * 4
    assert trial.suggestions == [
        ("server:max_num_seqs", 16, 64, 1),
        ("server:max_num_batched_tokens", 4096, 16384, 4096),
        ("client:max_concurrency", 8, 32, 8),
    ]
    assert dict(lo_args.get_all_kv_pairs(selected)) == {
        "max_num_seqs": 64,
        "max_num_batched_tokens": 16384,
        "max_concurrency": 32,
    }


@pytest.mark.parametrize(
    "definition",
    [
        "max_num_seqs=int(64,16)",
        "max_num_seqs=int(16,64,0)",
        "max_num_seqs=int(16,65,8)",
        "max_num_seqs=int(16.0,64)",
        "gpu_memory_utilization=int(1,2)",
    ],
)
def test_invalid_optuna_integer_ranges_fail(definition):
    with pytest.raises(ValueError):
        parse_int_ranges(
            [definition], lo_args.ArgScope.SERVER, SERVER_CONFIGS["vllm"]
        )


def test_optuna_range_cannot_overlap_categorical_argument():
    arg_sets = tuple(_sets("max_num_seqs=[16,32]", lo_args.ArgScope.SERVER))
    ranges = parse_int_ranges(
        ["max_num_seqs=int(16,64)"],
        lo_args.ArgScope.SERVER,
        SERVER_CONFIGS["vllm"],
    )
    with pytest.raises(ValueError, match="also appears"):
        OptunaSearchSpace(arg_sets, ranges)


@pytest.mark.parametrize(
    ("framework", "definition", "native_option"),
    [
        ("vllm", "max_num_seqs=int(16,64)", "--max-num-seqs=64"),
        ("sglang", "max_running_requests=int(16,64)", "--max-running-requests=64"),
        ("trtllm", "max_batch_size=int(16,64)", "--max_batch_size=64"),
    ],
)
def test_integer_ranges_keep_framework_native_spelling(
    framework, definition, native_option
):
    ranges = parse_int_ranges(
        [definition], lo_args.ArgScope.SERVER, SERVER_CONFIGS[framework]
    )

    class HighTrial:
        def suggest_int(self, name, low, high, *, step):
            return high

    combo = OptunaSearchSpace((), ranges).suggest(HighTrial())
    settings = construct_benchmark_settings(combo, framework)
    assert native_option in settings["server_cmd_args"]


def test_objectives_validate_numeric_values_and_direction():
    assert objective_direction("output_throughput") == "maximize"
    assert objective_direction("p95_ttft_ms") == "minimize"
    assert extract_objective({"output_throughput": 12}, "output_throughput") == 12
    with pytest.raises(ValueError, match="missing or non-numeric"):
        extract_objective({}, "output_throughput")


def test_every_constraint_operator_and_seconds_conversion():
    metrics = {"mean_ttft_ms": 1000.0}
    expected = {
        "<": False,
        "<=": True,
        ">": False,
        ">=": True,
        "==": True,
        "!=": False,
    }
    for operator, feasible in expected.items():
        constraints = parse_slo_constraints(f"ttft{operator}1s")
        assert satisfies_constraints(metrics, constraints) is feasible
        assert (constraint_violations(metrics, constraints)[0] <= 0) is feasible


def test_missing_constraint_metric_is_infeasible():
    constraints = parse_slo_constraints("itl:p95<10ms")
    assert not satisfies_constraints({}, constraints)


def test_optuna_dry_run_creates_no_files(tmp_path: Path):
    output = tmp_path / "nested" / "study.json"
    result = CliRunner().invoke(
        cli,
        [
            "--framework",
            "vllm",
            "--model",
            "example/model",
            "--gpus",
            "1",
            "--server-args",
            "max_num_seqs=[1,2]",
            "--search-strategy",
            "optuna",
            "--output-json",
            str(output),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Search-space size: 2" in result.output
    assert "Objective: output_throughput (maximize)" in result.output
    assert not output.parent.exists()


def test_optuna_integer_dry_run_and_grid_rejection(tmp_path: Path):
    output = tmp_path / "nested" / "study.json"
    args = [
        "--framework", "vllm", "--model", "example/model", "--gpus", "1",
        "--server-args", "tensor_parallel_size=1",
        "--optuna-server-args", "max_num_seqs=int(16,64)",
        "--output-json", str(output), "--dry-run",
    ]
    grid = CliRunner().invoke(cli, args)
    assert grid.exit_code != 0
    assert "require --search-strategy optuna" in grid.output
    optuna = CliRunner().invoke(cli, args + ["--search-strategy", "optuna"])
    assert optuna.exit_code == 0, optuna.output
    assert "Search-space size: 49" in optuna.output
    assert not output.parent.exists()


def test_missing_optional_dependency_is_actionable(tmp_path, monkeypatch):
    def missing():
        raise RuntimeError("Install it with: pip install -e '.[optuna]'")

    monkeypatch.setattr("llm_optimizer.optimization.require_optuna", missing)
    output = tmp_path / "nested" / "study.json"
    result = CliRunner().invoke(
        cli,
        [
            "--framework",
            "vllm",
            "--model",
            "example/model",
            "--gpus",
            "1",
            "--search-strategy",
            "optuna",
            "--output-json",
            str(output),
        ],
    )
    assert result.exit_code != 0
    assert ".[optuna]" in result.output
    assert not output.parent.exists()


def test_sqlite_search_resume_and_signature_validation(tmp_path, monkeypatch):
    pytest.importorskip("optuna")
    import llm_optimizer.cli as cli_module

    fake_client = types.ModuleType("llm_optimizer.bench_client")
    fake_client.run_benchmark = lambda args: {
        "output_throughput": float(args["max_concurrency"]),
        "mean_ttft_ms": 2.0,
    }
    monkeypatch.setitem(sys.modules, "llm_optimizer.bench_client", fake_client)
    monkeypatch.setattr(cli_module, "start_server", lambda *args: object())
    terminated = []
    monkeypatch.setattr(
        cli_module,
        "terminate_process_top_down",
        lambda process: terminated.append(process),
    )
    output = tmp_path / "study.json"
    arguments = [
        "--framework",
        "vllm",
        "--model",
        "example/model",
        "--gpus",
        "1",
        "--server-args",
        "max_num_seqs=[1,2]",
        "--client-args",
        "max_concurrency=[1,2]",
        "--search-strategy",
        "optuna",
        "--optuna-trials",
        "4",
        "--rest",
        "0",
        "--output-json",
        str(output),
    ]
    first = CliRunner().invoke(cli, arguments)
    assert first.exit_code == 0, first.output
    combined = json.loads(output.read_text())
    assert combined["metadata"]["total_tests"] == 4
    assert combined["optimization"]["stop_reason"] == "trial_budget"
    assert len(terminated) == 4
    assert output.with_suffix(".optuna.db").exists()

    resumed = CliRunner().invoke(cli, arguments + ["--continue"])
    assert resumed.exit_code == 0, resumed.output
    assert len(terminated) == 4

    changed = list(arguments)
    changed[changed.index("max_concurrency=[1,2]")] = "max_concurrency=[1,3]"
    mismatch = CliRunner().invoke(cli, changed + ["--continue"])
    assert mismatch.exit_code != 0
    assert "signature" in mismatch.output


def test_integer_range_trials_persist_and_resume(tmp_path, monkeypatch):
    optuna = pytest.importorskip("optuna")
    import llm_optimizer.cli as cli_module

    fake_client = types.ModuleType("llm_optimizer.bench_client")
    fake_client.run_benchmark = lambda args: {
        "output_throughput": float(args["max_concurrency"]),
        "mean_ttft_ms": 2.0,
    }
    monkeypatch.setitem(sys.modules, "llm_optimizer.bench_client", fake_client)
    monkeypatch.setattr(cli_module, "start_server", lambda *args: object())
    terminated = []
    monkeypatch.setattr(
        cli_module,
        "terminate_process_top_down",
        lambda process: terminated.append(process),
    )
    output = tmp_path / "integer.json"
    arguments = [
        "--framework", "vllm", "--model", "example/model", "--gpus", "1",
        "--server-args", "tensor_parallel_size=1",
        "--optuna-server-args", "max_num_seqs=int(1,3)",
        "--optuna-client-args", "max_concurrency=int(1,2)",
        "--search-strategy", "optuna", "--optuna-trials", "4",
        "--rest", "0", "--output-json", str(output),
    ]
    first = CliRunner().invoke(cli, arguments)
    assert first.exit_code == 0, first.output
    report = json.loads(output.read_text())
    assert report["optimization"]["search_space_size"] == 6
    assert report["metadata"]["total_tests"] == 4
    assert len(terminated) == 4
    study = optuna.load_study(
        study_name=report["optimization"]["study_name"],
        storage=f"sqlite:///{output.with_suffix('.optuna.db')}",
    )
    complete = [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    assert len(complete) == 4
    assert all(
        isinstance(trial.distributions["server:max_num_seqs"],
                   optuna.distributions.IntDistribution)
        for trial in complete
    )
    assert all(
        isinstance(trial.distributions["client:max_concurrency"],
                   optuna.distributions.IntDistribution)
        for trial in complete
    )

    resumed = CliRunner().invoke(cli, arguments + ["--continue"])
    assert resumed.exit_code == 0, resumed.output
    assert len(terminated) == 4
    changed = list(arguments)
    changed[changed.index("max_num_seqs=int(1,3)")] = "max_num_seqs=int(1,4)"
    mismatch = CliRunner().invoke(cli, changed + ["--continue"])
    assert mismatch.exit_code != 0
    assert "signature" in mismatch.output
