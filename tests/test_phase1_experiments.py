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
    for key, value in dict(RUN_ROOT=tmp_path, TRIALS=2, REQUESTS=32,
                           SEEDS=[11], CONCURRENCIES=[1, 8]).items():
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
