import json
from pathlib import Path

import optuna
import pytest

from llm_optimizer.experiment.gp_policy import GPFailurePolicy


class Failure(RuntimeError):
    pass


def test_rejected_proposals_do_not_launch_or_consume_execution_budget(tmp_path):
    study = optuna.create_study()
    for value in [0, 0, 1]:
        study.enqueue_trial({"value": value})
    policy = GPFailurePolicy(study, tmp_path, {"model": "test"})
    launches = []

    def objective(trial):
        config = {"value": trial.suggest_int("value", 0, 1)}
        trial.set_user_attr("config", config)
        policy.check(trial, config)
        launches.append(config["value"])
        if config["value"] == 0:
            trial.set_user_attr("failure_reason", "invalid configuration")
            raise Failure("invalid configuration")
        return 1

    callbacks = []
    policy.optimize(objective, 2, Failure, [lambda s, t: callbacks.append(t.number)])
    assert launches == [0, 1]
    assert callbacks == [0, 2]
    data = json.loads(policy.path.read_text())
    assert len(data["failures"]) == len(data["rejections"]) == 1
    assert data["rejections"][0]["server_launched"] is False
    resumed = GPFailurePolicy(study, tmp_path, {"model": "test"})
    assert resumed.blocked == policy.blocked
    assert len(resumed.data["failures"]) == 1


def test_bounded_blocked_proposals_stop_without_fallback(tmp_path):
    study = optuna.create_study()
    for _ in range(4):
        study.enqueue_trial({"value": 0})
    policy = GPFailurePolicy(study, tmp_path, {"model": "test"}, max_rejections=3)
    launches = []

    def objective(trial):
        config = {"value": trial.suggest_int("value", 0, 1)}
        trial.set_user_attr("config", config)
        policy.check(trial, config)
        launches.append(0)
        raise Failure("bad")

    policy.optimize(objective, 5, Failure, [])
    assert launches == [0]
    assert len(policy.data["rejections"]) == 3
    assert "no fallback" in study.user_attrs["gp_stop_reason"]
    assert len(policy.data["stops"]) == 1


def test_yaml_contents_and_fixed_settings_define_identity(tmp_path):
    pytest.importorskip("yaml")
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("cache: 0.8\n")
    second.write_text("cache: 0.8\n")
    policy = GPFailurePolicy(
        optuna.create_study(), tmp_path, {"model": "test"}, fixed_server={"tp_size": 1}
    )
    assert policy.fingerprint({"config": str(first)}) == policy.fingerprint(
        {"config": str(second)}
    )
    assert policy.fingerprint({"tp_size": 2}) != policy.fingerprint({})


def test_cli_gp_stops_after_blocked_proposals_and_logs_failures(tmp_path, monkeypatch):
    import sys
    import types

    from click.testing import CliRunner

    import llm_optimizer.cli as cli_module

    class StaticGP(optuna.samplers.GPSampler):
        def __init__(self, **kwargs):
            pass

        def infer_relative_search_space(self, study, trial):
            return {}

        def sample_relative(self, study, trial, search_space):
            return {}

        def sample_independent(self, study, trial, name, distribution):
            return distribution.choices[0]

        def before_trial(self, study, trial):
            pass

        def after_trial(self, study, trial, state, values):
            pass

    monkeypatch.setattr(optuna.samplers, "GPSampler", StaticGP)
    fake_client = types.ModuleType("llm_optimizer.bench_client")

    def fail(args):
        raise RuntimeError("bad config")

    fake_client.run_benchmark = fail
    monkeypatch.setitem(sys.modules, "llm_optimizer.bench_client", fake_client)
    launches = []
    monkeypatch.setattr(
        cli_module, "start_server", lambda *a: launches.append(a) or object()
    )
    monkeypatch.setattr(cli_module, "terminate_process_top_down", lambda p: None)
    output = tmp_path / "gp.json"
    result = CliRunner().invoke(
        cli_module.cli,
        [
            "--framework",
            "vllm",
            "--model",
            "test/model",
            "--gpus",
            "1",
            "--server-args",
            "max_num_seqs=[1,2]",
            "--client-args",
            "max_concurrency=1",
            "--search-strategy",
            "optuna",
            "--optuna-sampler",
            "GPSampler",
            "--optuna-trials",
            "2",
            "--rest",
            "0",
            "--output-json",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(launches) == 1
    report = json.loads(output.read_text())
    assert report["optimization"]["stop_reason"] == "gp_blocked_proposal_limit"
    audit = json.loads(Path(report["optimization"]["gp_failure_log"]).read_text())
    assert len(audit["failures"]) == 1
    assert len(audit["rejections"]) == 10
