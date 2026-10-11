"""CLI artifacts and parity with the unchanged reference notebook."""

import copy
import json
from pathlib import Path
from unittest.mock import patch

import optuna
import pytest
from click.testing import CliRunner
from test_optuna_notebook import lab as reference_lab_fixture

from llm_optimizer.cli import cli
from llm_optimizer.experiment.config import DEFAULTS, ExperimentConfig
from llm_optimizer.experiment.workflow import create_lab, execute

lab = reference_lab_fixture


@pytest.fixture
def config_path(tmp_path):
    c = copy.deepcopy(DEFAULTS)
    c.update(
        phases=["A"],
        frameworks=["vllm"],
        concurrencies=[1],
        sampler_names=["random"],
        pareto_samplers=["random"],
        objectives={"1": ["e2e", "pareto_latency"], "128": ["e2e"]},
        validation_repeats=2,
    )
    c["demo_trials"]["A"]["1"] = 4
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(c))
    return path


@pytest.mark.parametrize(
    "field,value",
    [
        ("concurrencies", [0]),
        ("validation_repeats", True),
        ("sampler_names", ["unknown"]),
        ("objectives", {"1": ["throughput"]}),
        ("run_real_gpu", "yes"),
        ("unknown", 1),
    ],
)
def test_invalid_config(config_path, field, value):
    c = json.loads(config_path.read_text())
    c[field] = value
    config_path.write_text(json.dumps(c))
    with pytest.raises(ValueError):
        ExperimentConfig.load(config_path)


def test_cli_artifacts_resume_and_frozen_config(config_path, tmp_path):
    run_dir = tmp_path / "run"
    runner = CliRunner()
    args = ["--config", str(config_path), "--run-dir", str(run_dir)]
    for command in ["run", "run", "validate", "report"]:
        result = runner.invoke(cli, ["experiment", command, *args])
        assert result.exit_code == 0, result.output
        if command == "run":
            current = json.loads((run_dir / "results.json").read_text())
            assert all(len(s["trials"]) == 4 for s in current["studies"])
    artifact = json.loads((run_dir / "results.json").read_text())
    assert {p.name for p in run_dir.iterdir()} >= {
        "config.json",
        "environment.json",
        "studies.sqlite3",
        "experiment.log",
        "results.json",
    }
    assert len(artifact["studies"]) == 2
    assert all(len(s["trials"]) == 4 for s in artifact["studies"])
    assert artifact["validation"] and artifact["recommendations"]
    for s in artifact["studies"]:
        for t in s["trials"]:
            assert (
                t["attrs"]["measurement"]["total_requests"]
                == s["contract"]["workload"]["num_prompts"]
            )
            assert t["distributions"]
    c = json.loads(config_path.read_text())
    c["seeds"] = [99]
    config_path.write_text(json.dumps(c))
    result = runner.invoke(cli, ["experiment", "run", *args])
    assert result.exit_code != 0
    assert "frozen config" in result.output


@pytest.mark.parametrize(
    "phase,framework,concurrency,target",
    [
        ("A", "vllm", 1, "e2e"),
        ("A", "sglang", 16, "pareto_throughput"),
        ("A", "trtllm", 1, "pareto_latency"),
        ("B", "vllm", 1, "ttft"),
        ("B", "sglang", 128, "throughput"),
        ("B", "trtllm", 128, "pareto_throughput"),
    ],
)
def test_notebook_search_validation_and_export_parity(
    lab, config_path, tmp_path, phase, framework, concurrency, target
):
    c = ExperimentConfig.load(config_path)
    c.values["concurrencies"] = [1, 16, 128]
    run_dir = tmp_path / "extracted"
    run_dir.mkdir()
    extracted = create_lab(c, run_dir)
    lab["CONCURRENCIES"] = [1, 16, 128]
    lab["WANDB_ENABLED"] = False
    lab["RUN_REAL_GPU"] = False
    kwargs = {
        "concurrency": concurrency,
        "phase": phase,
        "target": target,
        "seed": 11,
        "trials": 4,
        "space_kind": "baseline" if phase == "A" else "expanded",
    }
    reference_kwargs = {**kwargs, "trials": kwargs["trials"] + 1}
    # The reference subtracts its queued warm-start before it has executed.
    # Compare identical executed budgets while keeping the source unchanged.
    reference = lab["run_study"]("parity", framework, "random", **reference_kwargs)
    study = extracted.run_study("parity", framework, "random", **kwargs)
    reference_contract = {**reference.user_attrs["contract"], "budget_trials": 4}
    assert study.user_attrs["contract"] == reference_contract
    assert [(t.params, t.values, t.state) for t in study.trials] == [
        (t.params, t.values, t.state) for t in reference.trials
    ]
    assert [
        t.number
        for t in extracted.pareto_front(extracted.completed(study), study.directions)
    ] == [
        t.number
        for t in lab["pareto_front"](lab["completed"](reference), reference.directions)
    ]
    reference.set_user_attr("contract", reference_contract)
    expected = lab["validate_candidates"](lab["shortlist"]([reference]), repeats=2)
    actual = extracted.validate_candidates(extracted.shortlist([study]), repeats=2)
    assert actual == expected
    assert extracted.validation_recommendations(actual) == lab[
        "validation_recommendations"
    ](expected)
    from llm_optimizer.experiment.workflow import export_results

    artifact = export_results(extracted, [study], actual, [])
    saved = json.loads((run_dir / "results.json").read_text())
    assert saved == extracted.artifact_value(artifact)
    assert saved["studies"][0]["contract"] == reference.user_attrs["contract"]
    for a, b in zip(saved["studies"][0]["trials"], reference.trials):
        assert a["attrs"]["measurement"] == b.user_attrs["measurement"]


def test_real_gpu_routes_repository_adapter_and_skips_pruning(config_path, tmp_path):
    import llm_optimizer.bench_client as client
    import llm_optimizer.cli as cli_module

    c = json.loads(config_path.read_text())
    c.update(
        run_real_gpu=True,
        real_frameworks=["vllm"],
        rest_seconds=0,
        pruning=True,
        model_snapshots={"A": str(tmp_path)},
    )
    c["real_preflight"] = {
        "A": {
            "vllm": {
                "gpu_type": "mock-gpu",
                "framework_version": "mock",
                "cuda_version": "mock",
                "driver_version": "mock",
                "model_revision": "pinned-mock",
                "native_config_verified": True,
            }
        }
    }
    config_path.write_text(json.dumps(c))
    count = c["phase_requests"]["A"]
    metrics = {
        "output_throughput": 1000,
        "p95_ttft_ms": 80,
        "p95_e2e_latency_ms": 400,
        "total_requests": count,
        "completed": count,
        "failed": 0,
        "benchmark_valid": True,
    }
    with (
        patch.object(
            client,
            "prepare_workload_manifest",
            return_value=(
                {"requests": [], "sha256": "mock"},
                {"workload_sha256": "mock"},
            ),
        ),
        patch.object(
            cli_module,
            "run_single_configuration",
            return_value=({"results": metrics}, None),
        ) as benchmark,
    ):
        result_path = execute("run", config_path, tmp_path / "gpu-run")
    assert benchmark.called
    assert benchmark.call_args.kwargs["managed_server"] is True
    assert benchmark.call_args.kwargs["workload_manifest"]["requests"] == []
    results = json.loads(result_path.read_text())
    assert results["studies"][0]["contract"]["environment"]["measurement"] == "MEASURED"
    assert any(
        "Real pruning skipped" in arm["reason"] for arm in results["skipped_arms"]
    )


def test_running_trials_are_not_silently_resumed(config_path, tmp_path):
    run_dir = tmp_path / "run"
    execute("run", config_path, run_dir)
    study = optuna.load_study(
        study_name="A-vllm-c1-e2e-random-s11",
        storage=f"sqlite:///{run_dir / 'studies.sqlite3'}",
    )
    study.ask()
    with pytest.raises(RuntimeError, match="RUNNING"):
        execute("run", config_path, run_dir)


def test_launcher_uses_shell_commands():
    notebook = json.loads(
        (
            Path(__file__).resolve().parents[1] / "docs/optuna_experiment_cli.ipynb"
        ).read_text()
    )
    sources = [
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"
    ]
    for command in ("run", "validate", "report"):
        assert any(
            s.startswith(f"!llm-optimizer experiment {command} ") for s in sources
        )
    assert any("results.json" in s and "DataFrame" in s for s in sources)
    assert not any("register_magic" in s for s in sources)


def test_one_trial_executes_queued_candidate(config_path, tmp_path):
    c = json.loads(config_path.read_text())
    c["demo_trials"]["A"]["1"] = 1
    c["objectives"]["1"] = ["e2e"]
    config_path.write_text(json.dumps(c))
    result = execute("run", config_path, tmp_path / "single")
    study = json.loads(result.read_text())["studies"][0]
    assert len(study["trials"]) == 1
    assert study["trials"][0]["state"] == "COMPLETE"


def test_documented_config_matches_installed_defaults():
    path = Path(__file__).resolve().parents[1] / "docs/optuna_experiment.json"
    assert json.loads(path.read_text()) == DEFAULTS
