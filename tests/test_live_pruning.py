"""Production pruning decisions must cancel trials without persisting candidates."""

import json
import sys
import types

import optuna
import pytest
from click.testing import CliRunner

from llm_optimizer.cli import cli
from llm_optimizer.optimization import checkpoint_schedule


@pytest.mark.parametrize(
    "count,minimum,expected",
    [
        (256, 1, [64, 128, 256]),
        (100, 60, [60, 100]),
        (10, 20, [10]),
    ],
)
def test_schedule(count, minimum, expected):
    assert checkpoint_schedule(count, minimum) == expected


@pytest.mark.parametrize("failed", [False, True])
def test_live_checkpoint_stops_and_cleans_up(tmp_path, monkeypatch, failed):
    import llm_optimizer.cli as module

    cleaned = []

    def benchmark(args):
        assert args["_checkpoint_requests"] == [4, 8, 16]
        args["_checkpoint_callback"](
            {
                "completed": 4,
                "total_requests": 5 if failed else 4,
                "failed": int(failed),
                "failure_rate": 0.2 if failed else 0,
                "benchmark_valid": not failed,
                "output_throughput": 1.0,
            }
        )
        raise AssertionError("A stopped trial must not finish the benchmark")

    monkeypatch.setitem(
        sys.modules,
        "llm_optimizer.bench_client",
        types.SimpleNamespace(run_benchmark=benchmark),
    )
    monkeypatch.setattr(module, "start_server", lambda *args: object())
    monkeypatch.setattr(module, "terminate_process_top_down", cleaned.append)
    monkeypatch.setattr(optuna.trial.Trial, "should_prune", lambda self: True)
    output = tmp_path / "result.json"
    args = [
        "--framework",
        "vllm",
        "--model",
        "example/model",
        "--gpus",
        "1",
        "--server-args",
        "max_num_seqs=[1,2]",
        "--client-args",
        "num_prompts=16",
        "--search-strategy",
        "optuna",
        "--optuna-sampler",
        "GridSampler",
        "--optuna-pruner",
        "MedianPruner",
        "--rest",
        "0",
        "--output-json",
        str(output),
    ]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.exception
    study = optuna.load_study(
        storage=f"sqlite:///{output.with_suffix('.optuna.db')}",
        study_name=json.loads(output.read_text())["optimization"]["study_name"],
    )
    expected = (
        optuna.trial.TrialState.FAIL if failed else optuna.trial.TrialState.PRUNED
    )
    assert all(trial.state == expected for trial in study.trials)
    assert len(study.trials) == len(cleaned) == 2
    assert json.loads(output.read_text())["test_results"] == []
    assert all(t.user_attrs["gpu_seconds"] >= 0 for t in study.trials)


def test_pruned_configurations_are_not_retried_on_resume(tmp_path, monkeypatch):
    import llm_optimizer.cli as module

    calls = []

    def benchmark(args):
        calls.append(args)
        args["_checkpoint_callback"](
            {
                "completed": 4,
                "total_requests": 4,
                "failed": 0,
                "failure_rate": 0,
                "benchmark_valid": True,
                "output_throughput": 1.0,
            }
        )

    monkeypatch.setitem(
        sys.modules,
        "llm_optimizer.bench_client",
        types.SimpleNamespace(run_benchmark=benchmark),
    )
    monkeypatch.setattr(module, "start_server", lambda *args: object())
    monkeypatch.setattr(module, "terminate_process_top_down", lambda *args: None)
    monkeypatch.setattr(optuna.trial.Trial, "should_prune", lambda self: True)
    args = [
        "--framework",
        "vllm",
        "--model",
        "example/model",
        "--gpus",
        "1",
        "--client-args",
        "num_prompts=16",
        "--search-strategy",
        "optuna",
        "--optuna-pruner",
        "MedianPruner",
        "--rest",
        "0",
        "--output-json",
        str(tmp_path / "resume.json"),
    ]
    runner = CliRunner()
    assert runner.invoke(cli, args).exit_code == 0
    result = runner.invoke(cli, args + ["--continue"])
    assert result.exit_code == 0, result.exception
    assert len(calls) == 1
