"""Tests for finite Optuna search helpers and dry-run behavior."""

import json
import sys
import types
from pathlib import Path

import pytest
from click.testing import CliRunner

import llm_optimizer.args as lo_args
from llm_optimizer.cli import cli
from llm_optimizer.optimization import (
    OptunaSearchSpace,
    constraint_violations,
    extract_objective,
    objective_direction,
    satisfies_constraints,
)
from llm_optimizer.performance import parse_slo_constraints


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
