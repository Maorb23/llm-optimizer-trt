import json
import sys
import types

import optuna
import pytest
from click.testing import CliRunner

from llm_optimizer.cli import cli
from llm_optimizer.optimization import extract_objective, summarize_study
from llm_optimizer.remote.contracts import OptimizationRequest
from llm_optimizer.remote.worker import optimizer_command


def test_full_front_and_per_objective_importance():
    study = optuna.create_study(directions=["maximize", "minimize"])
    # Different objectives depend on different parameters.
    for x in range(5):
        for y in range(5):
            study.add_trial(
                optuna.trial.create_trial(
                    params={"x": x, "y": y},
                    distributions={
                        "x": optuna.distributions.IntDistribution(0, 4),
                        "y": optuna.distributions.IntDistribution(0, 4),
                    },
                    values=[float(x), float(y)],
                    user_attrs={"slo_feasible": x != 4},
                )
            )
    summary = summarize_study(study, ["output_throughput", "mean_e2e_latency_ms"])
    assert len(summary["pareto_front"]) == 1
    assert summary["pareto_front"][0]["params"] == {"x": 4, "y": 0}
    assert summary["pareto_front"][0]["slo_feasible"] is False
    importance = summary["importance_per_objective"]
    # Optional sklearn absence must leave an explicit unavailable result.
    if importance["output_throughput"]["status"] == "available":
        assert (
            importance["output_throughput"]["values"]["x"]
            > importance["output_throughput"]["values"]["y"]
        )
        assert (
            importance["mean_e2e_latency_ms"]["values"]["y"]
            > importance["mean_e2e_latency_ms"]["values"]["x"]
        )
    else:
        assert importance["output_throughput"]["reason"]
    tradeoff = optuna.create_study(directions=["maximize", "minimize"])
    for pair in [(100, 10), (150, 15), (200, 30), (90, 40)]:
        tradeoff.add_trial(optuna.trial.create_trial(values=pair))
    assert (
        len(
            summarize_study(tradeoff, ["output_throughput", "mean_e2e_latency_ms"])[
                "pareto_front"
            ]
        )
        == 3
    )


def test_cli_runs_genuine_multi_objective_and_preserves_resume(tmp_path, monkeypatch):
    import llm_optimizer.cli as cli_module

    client = types.ModuleType("llm_optimizer.bench_client")
    client.run_benchmark = lambda args: {
        "output_throughput": args["max_concurrency"] * 100,
        "mean_e2e_latency_ms": args["max_concurrency"] * 10,
    }
    monkeypatch.setitem(sys.modules, "llm_optimizer.bench_client", client)
    monkeypatch.setattr(cli_module, "start_server", lambda *args: object())
    monkeypatch.setattr(cli_module, "terminate_process_top_down", lambda *args: None)
    output = tmp_path / "multi.json"
    args = [
        "--framework",
        "vllm",
        "--model",
        "org/model",
        "--gpus",
        "1",
        "--server-args",
        "max_num_seqs=[1,2,3]",
        "--client-args",
        "max_concurrency=4",
        "--search-strategy",
        "optuna",
        "--optuna-sampler",
        "GridSampler",
        "--optuna-objective",
        "output_throughput,mean_e2e_latency_ms",
        "--rest",
        "0",
        "--output-json",
        str(output),
    ]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, (result.output, result.exception)
    data = json.loads(output.read_text())
    summary = data["optimization"]["study_summary"]
    assert summary["directions"] == ["maximize", "minimize"]
    assert summary["completed_trials"] == 3
    assert len(summary["pareto_front"]) == 3  # equal vectors must all survive
    assert set(summary["importance_per_objective"]) == {
        "output_throughput",
        "mean_e2e_latency_ms",
    }
    study = optuna.load_study(
        study_name=data["optimization"]["study_name"],
        storage=f"sqlite:///{output.with_suffix('.optuna.db')}",
    )
    assert len(study.directions) == 2
    assert all(len(t.values) == 2 for t in study.trials)
    result = CliRunner().invoke(cli, [*args, "--continue"])
    assert result.exit_code == 0, (result.output, result.exception)
    assert (
        json.loads(output.read_text())["optimization"]["study_summary"][
            "completed_trials"
        ]
        == 3
    )


def test_remote_objectives_propagation_and_validation(tmp_path):
    request = {
        "model": "org/model",
        "frameworks": ["vllm"],
        "gpu_type": "gpu-h100-sxm",
        "images": {"vllm": "image"},
        "bucket": "bucket",
        "search_strategy": "optuna",
        "optuna": {"objectives": ["output_throughput", "mean_e2e_latency_ms"]},
    }
    command = optimizer_command(
        {"request": request, "framework": "vllm"}, tmp_path / "results.json"
    )
    assert (
        command[command.index("--optuna-objective") + 1]
        == "output_throughput,mean_e2e_latency_ms"
    )
    request["optuna"]["objective"] = "output_throughput"
    with pytest.raises(ValueError, match="omit objective"):
        OptimizationRequest.parse(request)
    assert extract_objective(
        {"output_throughput": 5, "mean_e2e_latency_ms": 3},
        "output_throughput,mean_e2e_latency_ms",
    ) == [5, 3]


def test_failed_worker_retains_successful_jsonl_measurements(tmp_path, monkeypatch):
    import huggingface_hub

    import llm_optimizer.remote.worker as worker

    class Store:
        def __init__(self):
            self.files = {}

        def upload(self, path, run_id, framework):
            self.files[path.name] = path.read_bytes()
            return f"s3://bucket/{run_id}/{framework}/{path.name}"

        def download(self, *args):
            return False

    def preflight(command, **kwargs):
        return types.SimpleNamespace(stdout="NVIDIA H100\n", returncode=0)

    class FailedProcess:
        returncode = 1

        def __init__(self, command, **kwargs):
            row = {
                "config": {"server_args": {"max_num_seqs": 1}},
                "results": {"output_throughput": 100, "mean_e2e_latency_ms": 10},
            }
            (tmp_path / "results.jsonl").write_text(json.dumps(row) + "\n{incomplete")

        def poll(self):
            return 1

    monkeypatch.setattr(worker.subprocess, "run", preflight)
    monkeypatch.setattr(worker, "_version", lambda package: "1.0")
    monkeypatch.setattr(
        huggingface_hub.HfApi,
        "model_info",
        lambda *args, **kwargs: types.SimpleNamespace(sha="abc123"),
    )
    request = {
        "model": "org/model",
        "frameworks": ["vllm"],
        "gpu_type": "gpu-h100-sxm",
        "images": {"vllm": "image"},
        "bucket": "bucket",
        "dry_run": False,
        "search_strategy": "optuna",
        "optuna": {"objectives": ["output_throughput", "mean_e2e_latency_ms"]},
    }
    store = Store()
    manifest = worker.run_worker(
        {"run_id": "remote", "framework": "vllm", "request": request},
        store=store,
        work_dir=tmp_path,
        runner=FailedProcess,
    )
    assert manifest["status"] == "failed"
    assert len(manifest["test_results"]) == 1
    assert manifest["test_results"][0]["results"]["output_throughput"] == 100
    assert "results.jsonl" in store.files
