"""Verify the relocated notebook and package entry point on CPU."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import optuna
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src/llm_optimizer/experiment/phase1_experiments.py"
NOTEBOOK = ROOT / "docs/phase1_experiments.ipynb"


def configuration(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("phase1_configuration", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for key, value in dict(RUN_ROOT=tmp_path, TRIALS=2,
                           REQUESTS_BY_CONCURRENCY={1: 32, 8: 32, 128: 32},
                           PRUNING_MIN_COMPLETED_REQUESTS={1: 12, 8: 12, 128: 12},
                           SAMPLER_NAMES=["tpe"], PARETO_SAMPLERS=["tpe"],
                           SEEDS=[11], CONCURRENCIES=[1, 8], WANDB_ENABLED=False).items():
        monkeypatch.setattr(module, key, value)
    return module


def test_notebook_and_module_agree():
    script = SCRIPT.read_text()
    notebook = json.loads(NOTEBOOK.read_text())
    for cell in notebook["cells"]:
        text = "".join(cell["source"])
        assert "notebooks/phase1_experiments" not in text
        if cell["cell_type"] == "code":
            compile(text, str(NOTEBOOK), "exec")
            if text.startswith(("from pathlib", "CAMPAIGN", "PREFILL_BUDGETS",
                                "# Independent Phase B", "def make_runner")):
                assert text in script
    assert "python -m llm_optimizer.experiment.phase1_experiments" in NOTEBOOK.read_text()


def test_search_resume_and_gpu_lock(tmp_path, monkeypatch):
    module = configuration(tmp_path, monkeypatch)
    module.EXTRA_SPACES["sglang"]["schedule_conservativeness"] = [0.6, 0.8]
    runner = module.make_runner()
    first = runner.run_objective_group("B", "throughput", [8])
    study = next(iter(first.values()))
    assert len(study.trials) == 2
    assert all(t.params["schedule_conservativeness"] in [0.6, 0.8] for t in study.trials)
    resumed = module.make_runner().run_objective_group("B", "throughput", [8])
    assert len(next(iter(resumed.values())).trials) == 2
    abandoned = study.ask()
    runner.run_objective_group("B", "throughput", [8])
    assert study.trials[abandoned.number].state == optuna.trial.TrialState.FAIL
    with runner.lock(), pytest.raises(RuntimeError, match="owns"):
        runner.run_objective_group("B", "throughput", [8])


@pytest.mark.parametrize("framework", ["vllm", "sglang", "trtllm"])
@pytest.mark.parametrize("weights,cache", [
    ("bf16", "bf16"), ("bf16", "fp8"), ("fp8", "bf16"), ("fp8", "fp8"),
])
def test_precision_serialization(tmp_path, monkeypatch, framework, weights, cache):
    module = configuration(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "FRAMEWORKS", [framework])
    runner = module.make_runner()
    menu = runner.lab.sampling_menu(framework, 8, "expanded")
    assert "kv_cache_dtype" not in menu
    params = {key: values[0] for key, values in menu.items()}
    params.update(weight_precision=weights, kv_cache_precision=cache)
    config = runner.lab.suggest_config(
        optuna.trial.FixedTrial(params), framework, 8, "expanded"
    )
    native = runner.lab.serialize_engine_config(framework, 8, config)
    if framework == "trtllm":
        document = yaml.safe_load(Path(native["config"]).read_text())
        assert document["dtype"] == "bfloat16"
        assert document["quant_config"] == {
            "quant_algo": "FP8" if weights == "fp8" else None,
            "kv_cache_quant_algo": "FP8" if cache == "fp8" else None,
        }
        assert ("calib_config" in document) == (weights == "fp8")
    else:
        assert native["dtype"] == "bfloat16"
        assert native.get("quantization") == ("fp8" if weights == "fp8" else None)
        fp8_cache = "fp8_e4m3" if framework == "sglang" else "fp8"
        assert native["kv_cache_dtype"] == (fp8_cache if cache == "fp8" else "auto")


def test_module_batch_launch_and_resume(tmp_path):
    command = [sys.executable, "-m", "llm_optimizer.experiment.phase1_experiments",
               "--smoke", "--run-root", str(tmp_path)]
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    for _ in range(2):
        result = subprocess.run(command, cwd=ROOT, env=environment, timeout=60,
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
    queue = json.loads(next(tmp_path.glob("*/queue.json")).read_text())
    assert queue["completed"] == [0]
    assert queue["attempts"] == {"0": 1}


@pytest.mark.parametrize("enabled,mode,expected_calls", [
    (True, "online", 1), (True, "offline", 0), (False, "online", 0),
])
def test_notebook_wandb_login(monkeypatch, enabled, mode, expected_calls):
    from types import SimpleNamespace
    from unittest.mock import Mock

    login = Mock()
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(login=login))
    notebook = json.loads(NOTEBOOK.read_text())
    cell = next("".join(c["source"]) for c in notebook["cells"]
                if c["cell_type"] == "code" and "wandb.login()" in "".join(c["source"]))
    exec(compile(cell, str(NOTEBOOK), "exec"),
         {"WANDB_ENABLED": enabled, "WANDB_MODE": mode})
    assert login.call_count == expected_calls


def test_phase1_wandb_logs_parameters_metrics_and_summary(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    module = configuration(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "WANDB_ENABLED", True)
    monkeypatch.setattr(module, "WANDB_PROJECT", "test-project")
    monkeypatch.setattr(module, "WANDB_ENTITY", "test-team")
    monkeypatch.setattr(module, "WANDB_GROUP", "test-campaign")
    module.MODEL_PROFILES["sglang"] = {"model_revision": "a" * 40,
        "draft_revision": "b" * 40, "speculative_decoding": {"algorithm": "DSPARK"}}
    fake_run = SimpleNamespace(id="test-run", url="https://wandb.test/run",
                               log=Mock(), finish=Mock(), summary={})
    initialize = Mock(return_value=fake_run)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=initialize))
    runner = module.make_runner()
    study = next(iter(runner.run_objective_group("B", "throughput", [8]).values()))
    kwargs = initialize.call_args.kwargs
    assert kwargs["config"]["environment"]["draft_revision"] == "b" * 40
    assert kwargs["project"] == "test-project"
    assert kwargs["entity"] == "test-team"
    assert kwargs["mode"] == "online"
    assert kwargs["group"] == "test-campaign"
    assert kwargs["name"] == study.study_name
    assert "SYNTHETIC" in kwargs["tags"]
    assert len(fake_run.log.call_args_list) == 2
    for call in fake_run.log.call_args_list:
        payload = call.args[0]
        assert "parameters/weight_precision" in payload
        assert "parameters/kv_cache_precision" in payload
        assert "objective/output_throughput" in payload
        assert "benchmark/p95_e2e_latency_ms" in payload
    assert fake_run.summary["best_or_pareto_trials"]
    fake_run.finish.assert_called_once()
    assert study.user_attrs["wandb"]["id"] == "test-run"


def test_sampler_selection_and_scaled_requests(tmp_path, monkeypatch):
    module = configuration(tmp_path, monkeypatch)
    counts = {1: 32, 8: 62, 16: 125, 32: 250, 128: 1000}
    monkeypatch.setattr(module, "REQUESTS_BY_CONCURRENCY", counts)
    monkeypatch.setattr(module, "PRUNING_MIN_COMPLETED_REQUESTS",
                        {c: max(1, int(n * 0.4)) for c, n in counts.items()})
    monkeypatch.setattr(module, "CONCURRENCIES", list(counts))
    monkeypatch.setattr(module, "SAMPLER_NAMES", ["tpe", "tpe_multivariate"])
    monkeypatch.setattr(module, "TRIALS", 1)
    runner = module.make_runner()
    for c, count in counts.items():
        target = "e2e" if c == 1 else "throughput"
        studies = runner.run_objective_group("B", target, [c])
        assert len(studies) == 2
        for study in studies.values():
            contract = study.user_attrs["contract"]
            assert contract["workload"]["num_prompts"] == count
            assert contract["budget_trials"] == 1
            assert study.trials[0].user_attrs["measurement"]["total_requests"] == count
            if contract["sampler"] == "tpe_multivariate":
                assert contract["sampler_settings"]["multivariate"] is True
        # Explicit validation request budgets must not be replaced by the search map.
        assert runner.lab.workload_for(c, "B", num_prompts=1000)["num_prompts"] == 1000


def test_default_controls_and_fresh_runner(tmp_path, monkeypatch):
    module = configuration(tmp_path, monkeypatch)
    first = module.get_runner()
    module.REQUESTS_BY_CONCURRENCY[8] = 62
    second = module.get_runner()
    assert second.lab.workload_for(8, "B")["num_prompts"] == 62
    assert second.run_dir != first.run_dir
    source = SCRIPT.read_text()
    assert 'TRIALS = 20' in source
    assert 'SAMPLER_NAMES = ["gp", "tpe", "tpe_multivariate"]' in source


def test_missing_gp_is_reported_without_substitution(tmp_path, monkeypatch):
    module = configuration(tmp_path, monkeypatch)
    runner = module.make_runner()
    runner.lab.RUN_GP = False
    with pytest.warns(UserWarning, match="torch"):
        studies = runner.run_objective_group("B", "throughput", [8], sampler_names=["gp", "tpe"])
    assert len(studies) == 1
    assert next(iter(studies.values())).user_attrs["contract"]["sampler"] == "tpe"
    skipped = json.loads((runner.run_dir / "skipped-arms.json").read_text())
    assert skipped[0]["sampler"] == "gp"


def test_median_checkpoints_scale_with_request_budget(tmp_path, monkeypatch):
    module = configuration(tmp_path, monkeypatch)
    module.REQUESTS_BY_CONCURRENCY[8] = 62
    module.PRUNING_MIN_COMPLETED_REQUESTS[8] = 24
    runner = module.make_runner()
    study = next(iter(runner.run_objective_group("B", "throughput", [8], "median").values()))
    assert study.user_attrs["contract"]["checkpoint_requests"] == [24, 31, 62]
    assert study.user_attrs["contract"]["pruner_settings"]["n_warmup_steps"] == 24


@pytest.mark.parametrize("target,concurrency", [("pareto_latency", 1), ("pareto_throughput", 8)])
@pytest.mark.parametrize("constrained", [False, True])
def test_pareto_and_constraint_search(tmp_path, monkeypatch, target, concurrency, constrained):
    module = configuration(tmp_path, monkeypatch)
    runner = module.make_runner()
    studies = runner.run_objective_group("B", target, [concurrency], constrained=constrained)
    study = next(iter(studies.values()))
    assert len(study.directions) == 2
    assert study.user_attrs["contract"]["constrained"] is constrained
    assert all(len(t.values) == 2 for t in study.trials)
    from llm_optimizer.experiment.phase1 import plot_group
    assert study.study_name in plot_group(studies)


@pytest.mark.parametrize("pruner", ["median", "percentile", "halving", "hyperband", "threshold"])
def test_pruners_and_unsupported_combinations(tmp_path, monkeypatch, pruner):
    module = configuration(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "PRUNING_SAMPLES", 10)
    runner = module.make_runner()
    with pytest.raises(ValueError, match="Pareto"):
        runner.run_objective_group("B", "pareto_latency", [1], pruner)
    with pytest.raises(ValueError, match="Constrained"):
        runner.run_objective_group("B", "e2e", [1], pruner, constrained=True)
    study = next(iter(runner.run_objective_group("B", "throughput", [8], pruner).values()))
    assert study.user_attrs["contract"]["checkpoint_requests"] == [12, 16, 32]
    assert study.user_attrs["contract"]["budget_trials"] == 2
    if pruner == "threshold":
        report = json.loads(next(runner.run_dir.glob("*-calibration.json")).read_text())
        assert report["workload"]["num_prompts"] == 32
        assert report["checkpoint_requests"] == [12, 16, 32]
        assert len(report["rank_correlations"]) == 2
        assert 0 <= report["false_prune_rate_among_top_10pct"] <= 1
        assert 0 <= report["replayed_time_saved_fraction"] <= 1
        assert report["synthetic"] is True


def test_slo_and_speculative_identity_metadata(tmp_path, monkeypatch):
    module = configuration(tmp_path, monkeypatch)
    original = module.make_runner()
    module.TTFT_LIMITS_MS["1"] = 999
    changed = module.make_runner()
    assert changed.run_dir != original.run_dir
    assert changed.lab.TTFT_LIMITS_MS[1] == 999
    profile = {"model_revision": "a" * 40, "draft_revision": "b" * 40,
               "speculative_decoding": {"algorithm": "DSPARK", "block_size": 16},
               "fixed_server": {"speculative_algorithm": "DSPARK",
                                "speculative_draft_model_path": "/pinned/draft",
                                "speculative_num_draft_tokens": 17}}
    module.MODEL_PROFILES["sglang"] = profile
    speculative = module.make_runner()
    assert speculative.run_dir != changed.run_dir
    assert speculative.lab.REAL_PREFLIGHT["B", "sglang"]["draft_revision"] == "b" * 40
    frozen = json.loads((speculative.run_dir / "config.json").read_text())
    assert frozen["settings"]["real_preflight"]["B"]["sglang"]["fixed_server"] == profile["fixed_server"]
    module.MODEL_PROFILES["sglang"]["draft_revision"] = "c" * 40
    assert module.make_runner().run_dir != speculative.run_dir
    module.MODEL_SETUP_IDENTITY = dict(module.HF_SETUP)
    module.HF_SETUP["block_size"] = 32
    with pytest.raises(ValueError, match="HF_SETUP changed"):
        module.make_runner()


def test_new_notebook_controls_match_script():
    script = SCRIPT.read_text()
    notebook = json.loads(NOTEBOOK.read_text())
    controls = next("".join(c["source"]) for c in notebook["cells"]
                    if "".join(c["source"]).startswith("# Illustrative SLOs"))
    assert controls in script
    for target in ("pareto_latency", "pareto_throughput"):
        assert any(f'run_objective_group("B", "{target}"' in "".join(c["source"])
                   for c in notebook["cells"])


def test_calibration_repository_callback_and_cache(tmp_path, monkeypatch):
    from llm_optimizer.experiment.lab import RepositoryBenchmark
    module = configuration(tmp_path, monkeypatch)
    runner = module.make_runner()
    lab = runner.lab
    fake = object.__new__(RepositoryBenchmark)
    fake.environment = {"measurement": "MEASURED", "requests_sha256": "fixed-manifest"}
    simulation = lab.Simulator(phase="B", num_prompts=32)
    calls = []
    def measure(framework, config, concurrency, checkpoint_callback, checkpoint_requests):
        calls.append(checkpoint_requests)
        rows = list(simulation.stream(framework, config, concurrency, checkpoint_requests))
        for _, _, measurement, _ in rows:
            checkpoint_callback(measurement)
        return rows[-1][2]
    fake.measure = measure
    monkeypatch.setattr(lab, "benchmark_for", lambda *args: fake)
    monkeypatch.setattr(lab, "RUN_REAL_GPU", True)
    report = runner._calibrate("B", "throughput", 8, "sglang", 10)
    assert calls == [[12, 16, 32]] * 10
    assert report["environment"]["requests_sha256"] == "fixed-manifest"
    assert report["synthetic"] is False
    assert runner._calibrate("B", "throughput", 8, "sglang", 10, reuse=True) == report
    assert len(calls) == 10


def test_constraint_sampler_rejection_and_identity(tmp_path, monkeypatch):
    module = configuration(tmp_path, monkeypatch)
    runner = module.make_runner()
    for sampler in ["random", "qmc", "grid"]:
        with pytest.raises(ValueError, match="constraint-aware"):
            runner.run_objective_group("B", "e2e", [1], sampler_names=[sampler], constrained=True)
    module.CONSTRAINED = True
    assert module.make_runner().run_dir != runner.run_dir


def test_phase1_hf_setup_commits_resolved_controls(tmp_path, monkeypatch):
    from llm_optimizer.experiment.lab import Lab
    module = configuration(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "RUN_REAL_GPU", True)
    monkeypatch.setattr(module, "HF_SETUP", {**module.HF_SETUP,
        "target_model_id": "pinned-target", "target_revision": "a" * 40,
        "speculative_algorithm": "DSPARK", "draft_model_id": "draft",
        "draft_revision": "b" * 40})
    def configure(lab, **options):
        assert options == module.HF_SETUP
        assert lab.RUN_REAL_GPU is True
        lab.REAL_PREFLIGHT["B", "sglang"].update(
            model_revision="a" * 40, draft_revision="b" * 40,
            speculative_decoding={"algorithm": "DSPARK"})
        return {"target_model_id": "pinned-target", "target_path": "/resolved/target"}
    monkeypatch.setattr(Lab, "configure_hf_models", configure)
    module.setup_pinned_models()
    monkeypatch.setattr(module, "RUN_REAL_GPU", False)
    runner = module.make_runner()
    assert runner.lab.PHASE_MODELS == {"A": module.MODEL, "B": "pinned-target"}
    assert runner.lab.MODEL_SNAPSHOTS["B"] == "/resolved/target"
    assert runner.lab.REAL_PREFLIGHT["B", "sglang"]["draft_revision"] == "b" * 40

    module.PREFLIGHT["sglang"]["native_config_verified"] = True
    assert module.make_runner().lab.REAL_PREFLIGHT["B", "sglang"]["native_config_verified"] is True
    module.EXPANDED_CHOICES["sglang"]["enable_mixed_chunk"] = [False]
    with pytest.raises(ValueError, match="Expanded choices changed"):
        module.make_runner()
