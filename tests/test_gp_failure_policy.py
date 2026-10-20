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


@pytest.mark.parametrize("pruned", [False, True])
def test_cli_gp_stops_after_blocked_proposals_and_logs_failures(
    tmp_path, monkeypatch, pruned
):
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
        if pruned:
            raise optuna.TrialPruned()
        raise RuntimeError("bad config")

    fake_client.run_benchmark = fail
    monkeypatch.setitem(sys.modules, "llm_optimizer.bench_client", fake_client)
    launches = []
    monkeypatch.setattr(
        cli_module, "start_server", lambda *a: launches.append(a) or object()
    )
    monkeypatch.setattr(cli_module, "terminate_process_top_down", lambda p: None)
    output = tmp_path / "gp.json"
    args = [
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
    ]
    result = CliRunner().invoke(cli_module.cli, args)
    assert result.exit_code == 0, result.output
    assert len(launches) == 1
    report = json.loads(output.read_text())
    assert report["optimization"]["stop_reason"] == "gp_blocked_proposal_limit"
    audit = json.loads(Path(report["optimization"]["gp_failure_log"]).read_text())
    assert len(audit["pruned" if pruned else "failures"]) == 1
    assert len(audit["rejections"]) == 10
    assert audit["rejections"][0]["original_trial_number"] == 0
    resumed = CliRunner().invoke(cli_module.cli, [*args, "--continue"])
    assert resumed.exit_code == 0, resumed.output
    assert len(launches) == 1
    report = json.loads(output.read_text())
    assert report["optimization"]["stop_reason"] == "gp_blocked_proposal_limit"
    audit = json.loads(Path(report["optimization"]["gp_failure_log"]).read_text())
    assert len(audit["rejections"]) == 20


def test_pruned_duplicates_are_failed_proposals_not_benchmarks(tmp_path):
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=11))
    for value in [0, 0, 1]:
        study.enqueue_trial({"value": value})
    policy = GPFailurePolicy(study, tmp_path, {"model": "test"})
    launches, callbacks = [], []

    def objective(trial):
        config = {"value": trial.suggest_int("value", 0, 1)}
        policy.check(trial, config)
        launches.append(config["value"])
        if config["value"] == 0:
            trial.report(10, 400)
            raise optuna.TrialPruned()
        return 1

    policy.optimize(objective, 2, Failure, [lambda s, t: callbacks.append(t.number)])
    assert launches == [0, 1]
    assert callbacks == [0, 2]
    assert [t.state for t in study.trials] == [
        optuna.trial.TrialState.PRUNED,
        optuna.trial.TrialState.FAIL,
        optuna.trial.TrialState.COMPLETE,
    ]
    rejected = study.trials[1]
    assert rejected.user_attrs["gp_original_trial_number"] == 0
    assert rejected.user_attrs["benchmark_executed"] is False
    assert not rejected.intermediate_values
    assert rejected.values is None
    assert len(policy.data["pruned"]) == 1
    assert policy.data["rejections"][0]["original_state"] == "PRUNED"
    assert not study.trials[0].user_attrs.get("gp_non_config_failure")


@pytest.mark.parametrize("restore_audit", [True, False])
def test_pruned_history_restored_from_sqlite_and_audit(tmp_path, restore_audit):
    storage = f"sqlite:///{tmp_path / 'study.db'}"
    study = optuna.create_study(study_name="resume", storage=storage)
    policy = GPFailurePolicy(study, tmp_path, {"model": "test"})

    def prune(trial):
        policy.check(trial, {"value": 0})
        raise optuna.TrialPruned()

    policy.optimize(prune, 1, Failure, [])
    if not restore_audit:
        policy.path.unlink()
    resumed_study = optuna.load_study(study_name="resume", storage=storage)
    resumed = GPFailurePolicy(resumed_study, tmp_path, {"model": "test"})
    trial = resumed_study.ask()
    from llm_optimizer.experiment.gp_policy import BlockedGPProposal

    with pytest.raises(BlockedGPProposal):
        resumed.check(trial, {"value": 0})
    assert resumed.data["rejections"][-1]["original_trial_number"] == 0
    assert len(resumed.data["pruned"]) == 1


@pytest.mark.parametrize(
    "field",
    [
        "model",
        "draft_revision",
        "workload",
        "objective",
        "search_space",
        "pruning",
    ],
)
def test_changed_experiment_scope_cannot_reuse_rejection_history(tmp_path, field):
    study = optuna.create_study(study_name="identity")
    scope = dict.fromkeys(
        ("model", "draft_revision", "workload", "objective", "search_space", "pruning"),
        "original",
    )
    GPFailurePolicy(study, tmp_path, {"model": "test"}, scope=scope)
    with pytest.raises(ValueError, match="experiment scope"):
        GPFailurePolicy(
            study, tmp_path, {"model": "test"}, scope={**scope, field: "changed"}
        )


def test_budget_changes_do_not_change_scoped_identity(tmp_path):
    study = optuna.create_study()
    study.set_user_attr(
        "contract", {"workload": {"num_prompts": 1000}, "budget_trials": 2}
    )
    policy = GPFailurePolicy(study, tmp_path, {"model": "test"})
    fingerprint = policy.fingerprint({"x": 1})
    study.set_user_attr(
        "contract", {"workload": {"num_prompts": 1000}, "budget_trials": 20}
    )
    resumed = GPFailurePolicy(study, tmp_path, {"model": "test"})
    assert resumed.fingerprint({"x": 1}) == fingerprint


def test_pruned_rejection_limit_keeps_sampler_and_execution_budget(tmp_path):
    sampler = optuna.samplers.RandomSampler(seed=11)
    study = optuna.create_study(sampler=sampler)
    for _ in range(5):
        study.enqueue_trial({"value": 0})
    policy = GPFailurePolicy(study, tmp_path, {"model": "test"}, max_rejections=3)
    launches = []

    def objective(trial):
        config = {"value": trial.suggest_int("value", 0, 1)}
        policy.check(trial, config)
        launches.append(trial.number)
        raise optuna.TrialPruned()

    policy.optimize(objective, 2, Failure, [])
    assert launches == [0]
    assert study.sampler is sampler
    assert len(policy.data["rejections"]) == 3
    assert "3 consecutive blocked proposals" in study.user_attrs["gp_stop_reason"]
    assert all(t.state == optuna.trial.TrialState.FAIL for t in study.trials[1:4])
    assert study.trials[4].state == optuna.trial.TrialState.WAITING


def test_pruner_parameters_are_scoped_even_without_explicit_settings(tmp_path):
    study = optuna.create_study(pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))
    GPFailurePolicy(study, tmp_path, {"model": "test"})
    study.pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)
    with pytest.raises(ValueError, match="experiment scope"):
        GPFailurePolicy(study, tmp_path, {"model": "test"})


@pytest.mark.parametrize("surface", ["notebook", "shared"])
def test_entry_points_reject_pruned_duplicates_and_resume(
    tmp_path, monkeypatch, surface
):
    import ast

    from llm_optimizer.experiment.config import DEFAULTS, ExperimentConfig
    from llm_optimizer.experiment.workflow import create_lab
    from llm_optimizer.optimization import checkpoint_schedule

    run_dir = tmp_path / surface
    run_dir.mkdir()
    lab = create_lab(ExperimentConfig(DEFAULTS), run_dir)
    lab.WANDB_ENABLED = False
    lab.RUN_REAL_GPU = False
    lab.make_sampler = lambda *a, **k: optuna.samplers.RandomSampler(seed=11)
    lab.suggest_config = lambda *a, **k: {"max_num_tokens": 2048}
    launches = []

    class PruningBenchmark:
        supports_checkpoints = False

        def measure(self, *args):
            launches.append(args)
            raise optuna.TrialPruned()

    if surface == "shared":
        run = lab.run_study
    else:
        notebook = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "docs"
                / "optuna_inference_optimization_lab_prune.ipynb"
            ).read_text()
        )
        namespace = dict(vars(lab))
        # Reuse instance algorithms; execute the notebook's real run_study definition.
        namespace.update(
            {
                name: getattr(lab, name)
                for name in dir(lab)
                if callable(getattr(lab, name)) and not name.startswith("__")
            }
        )
        namespace.update(
            checkpoint_schedule=checkpoint_schedule,
            optuna=optuna,
            TrialState=optuna.trial.TrialState,
            json=json,
            math=__import__("math"),
            time=__import__("time"),
            Path=Path,
            closing=__import__("contextlib").closing,
            RepositoryBenchmark=type("UnusedRealBenchmark", (), {}),
        )
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            for node in ast.parse("".join(cell["source"])).body:
                if isinstance(node, ast.FunctionDef) and node.name == "run_study":
                    exec(
                        compile(
                            ast.Module(body=[node], type_ignores=[]), "notebook", "exec"
                        ),
                        namespace,
                    )
        run = namespace["run_study"]

    kwargs = {"concurrency": 1, "trials": 2, "benchmark": PruningBenchmark()}
    study = run("gp-pruned", "trtllm", "gp", **kwargs)
    assert len(launches) == 1
    assert len(study.trials) == 11
    assert study.trials[0].state == optuna.trial.TrialState.PRUNED
    assert all(t.state == optuna.trial.TrialState.FAIL for t in study.trials[1:])
    resumed = run("gp-pruned", "trtllm", "gp", **kwargs)
    assert len(launches) == 1
    assert len(resumed.trials) == 21
    assert all(
        t.user_attrs["gp_original_trial_number"] == 0 for t in resumed.trials[1:]
    )


def test_legacy_pruned_non_config_flag_does_not_hide_pruning_history(tmp_path):
    study = optuna.create_study()
    study.add_trial(
        optuna.trial.create_trial(
            state=optuna.trial.TrialState.PRUNED,
            user_attrs={"config": {"x": 1}, "gp_non_config_failure": True},
            intermediate_values={400: 10.0},
        )
    )
    policy = GPFailurePolicy(study, tmp_path, {"model": "test"})
    from llm_optimizer.experiment.gp_policy import BlockedGPProposal

    with pytest.raises(BlockedGPProposal):
        policy.check(study.ask(), {"x": 1})
    assert policy.data["rejections"][0]["original_trial_number"] == 0


def test_scope_change_rejected_even_if_audit_file_is_missing(tmp_path):
    study = optuna.create_study()
    policy = GPFailurePolicy(
        study, tmp_path, {"model": "test"}, scope={"objective": "latency"}
    )
    policy.path.unlink()
    with pytest.raises(ValueError, match="experiment scope"):
        GPFailurePolicy(
            study, tmp_path, {"model": "test"}, scope={"objective": "throughput"}
        )


def test_legacy_failure_audit_is_migrated_without_losing_history(tmp_path):
    study = optuna.create_study(study_name="legacy")
    label = __import__("hashlib").sha256(study.study_name.encode()).hexdigest()[:16]
    path = tmp_path / f"gp-failures-{label}.json"
    path.write_text(
        json.dumps(
            {
                "study": study.study_name,
                "identity": {"model": "test"},
                "failures": [
                    {
                        "trial_number": 3,
                        "effective_config": {"x": 1},
                        "fingerprint": "legacy-fingerprint",
                        "reason": "failed",
                    }
                ],
                "rejections": [],
                "stops": [],
            }
        )
    )
    policy = GPFailurePolicy(study, tmp_path, {"model": "test"})
    from llm_optimizer.experiment.gp_policy import BlockedGPProposal

    with pytest.raises(BlockedGPProposal):
        policy.check(study.ask(), {"x": 1})
    assert policy.data["rejections"][0]["original_trial_number"] == 3
    assert "scope" in json.loads(path.read_text())
