"""CPU checks for explicit domains, resume identity and exclusive ownership."""
import importlib.util
import json
from pathlib import Path

import optuna
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "notebooks/phase1_experiments.py"


def config(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("phase1_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "RUN_ROOT", tmp_path)
    monkeypatch.setattr(module, "TRIALS", 2)
    monkeypatch.setattr(module, "SEEDS", [11])
    monkeypatch.setattr(module, "CONCURRENCIES", [1, 8])
    monkeypatch.setattr(module, "REQUESTS", 32)
    return module


def test_resume_and_explicit_domains(tmp_path, monkeypatch):
    module = config(tmp_path, monkeypatch)
    module.EXTRA_SPACES["sglang"]["schedule_conservativeness"] = [0.6, 0.8]
    runner = module.make_runner()
    first = runner.run_objective_group("B", "throughput", [8])
    study = next(iter(first.values()))
    assert len(study.trials) == 2
    assert all(t.params["schedule_conservativeness"] in [0.6, 0.8] for t in study.trials)
    assert study.user_attrs["contract"]["sampling_menu"]["schedule_conservativeness"] == [0.6, 0.8]
    resumed = module.make_runner().run_objective_group("B", "throughput", [8])
    assert len(next(iter(resumed.values())).trials) == 2
    module.EXTRA_SPACES["sglang"]["schedule_conservativeness"] = [0.5, 1.0]
    assert module.make_runner().run_dir != runner.run_dir


def test_abandoned_trial_and_lock(tmp_path, monkeypatch):
    module = config(tmp_path, monkeypatch)
    runner = module.make_runner()
    study = next(iter(runner.run_objective_group("A", "ttft", [1]).values()))
    abandoned = study.ask()
    runner.run_objective_group("A", "ttft", [1])
    assert study.trials[abandoned.number].state == optuna.trial.TrialState.FAIL
    with runner.lock(), pytest.raises(RuntimeError, match="owns"):
        runner.run_objective_group("B", "throughput", [8])


@pytest.mark.parametrize("framework", ["vllm", "sglang", "trtllm"])
def test_extended_single_and_pruning(tmp_path, monkeypatch, framework):
    module = config(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "FRAMEWORKS", [framework])
    runner = module.make_runner()
    study = next(iter(runner.run_objective_group("B", "ttft", [1], "median").values()))
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)


def test_notebook_setup_matches_script():
    notebook = json.loads(SCRIPT.with_suffix(".ipynb").read_text())
    script = SCRIPT.read_text()
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            compile(source, "cell", "exec")
            if source.startswith(("from pathlib", "CAMPAIGN", "PREFILL_BUDGETS", "# Independent Phase B", "def make_runner")):
                assert source in script


@pytest.mark.parametrize("framework", ["vllm", "sglang", "trtllm"])
@pytest.mark.parametrize("weights,cache", [
    ("bf16", "bf16"), ("bf16", "fp8"), ("fp8", "bf16"), ("fp8", "fp8"),
])
def test_precision_native_config(tmp_path, monkeypatch, framework, weights, cache):
    import yaml

    module = config(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "FRAMEWORKS", [framework])
    runner = module.make_runner()
    menu = runner.lab.sampling_menu(framework, 8, "expanded")
    assert "kv_cache_dtype" not in menu  # no second conflicting cache axis
    params = {k: v[0] for k, v in menu.items()}
    params.update(weight_precision=weights, kv_cache_precision=cache)
    engine = runner.lab.suggest_config(
        optuna.trial.FixedTrial(params), framework, 8, "expanded"
    )
    assert engine["dtype"] == "bfloat16"
    native = runner.lab.serialize_engine_config(framework, 8, engine)
    if framework == "trtllm":
        document = yaml.safe_load(Path(native["config"]).read_text())
        assert "quant_config" not in native and "dtype" not in native
        assert document["dtype"] == "bfloat16"
        assert document["quant_config"] == {
            "quant_algo": "FP8" if weights == "fp8" else None,
            "kv_cache_quant_algo": "FP8" if cache == "fp8" else None,
        }
        assert document["kv_cache_config"]["dtype"] == ("fp8" if cache == "fp8" else "auto")
        assert ("calib_config" in document) == (weights == "fp8")
    else:
        assert native.get("quantization") == ("fp8" if weights == "fp8" else None)
        expected_cache = ("fp8_e4m3" if framework == "sglang" else "fp8") if cache == "fp8" else "auto"
        assert native["kv_cache_dtype"] == expected_cache
        if framework == "vllm":
            assert native.get("calculate_kv_scales", False) == (cache == "fp8")


def test_precision_identity_and_bf16_baseline(tmp_path, monkeypatch):
    module = config(tmp_path, monkeypatch)
    runner = module.make_runner()
    baseline = runner.lab.render_engine_config("sglang", 8, {
        "max_running_requests": 8, "chunked_prefill_size": 4096, "mem_fraction_static": 0.8,
    })
    assert baseline["dtype"] == "bfloat16"
    assert baseline["kv_cache_dtype"] == "auto"
    assert "quantization" not in baseline
    module.PRECISION_CHOICES["sglang"]["weight_precision"] = ["fp8"]
    assert module.make_runner().run_dir != runner.run_dir
    saved = json.loads((runner.run_dir / "config.json").read_text())
    assert saved["precision_choices"]["sglang"]["weight_precision"] == ["bf16", "fp8"]


def test_real_precision_requires_verified_pairs_and_source_weights(tmp_path, monkeypatch):
    from llm_optimizer.experiment.config import ExperimentConfig

    module = config(tmp_path, monkeypatch)
    # Isolate precision preflight from the already-tested generic GPU preflight.
    monkeypatch.setattr(ExperimentConfig, "_validate", lambda c: None)
    monkeypatch.setattr(module, "RUN_REAL_GPU", True)
    monkeypatch.setattr(module, "MODEL_SNAPSHOT", str(tmp_path))
    model_config = tmp_path / "config.json"
    model_config.write_text('{}')
    with pytest.raises(ValueError, match="verify precision combinations"):
        module.make_runner()
    module.PREFLIGHT["sglang"]["verified_precision_combinations"] = [
        "bf16/bf16", "bf16/fp8", "fp8/bf16", "fp8/fp8",
    ]
    module.make_runner()
    model_config.write_text('{"quantization_config": {"quant_method": "fp8"}}')
    with pytest.raises(ValueError, match="unquantized source weights"):
        module.make_runner()


def test_precision_rejects_fixed_override(tmp_path, monkeypatch):
    module = config(tmp_path, monkeypatch)
    module.PREFLIGHT["sglang"]["fixed_server"] = {"quantization": "fp8"}
    with pytest.raises(ValueError, match="conflict with fixed_server"):
        module.make_runner()
