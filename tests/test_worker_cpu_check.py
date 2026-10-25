"""Exercise the offline image check with runtime distributions mocked."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "check_worker_cpu", ROOT / "docker/check_worker_cpu.py"
)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


@pytest.mark.parametrize("framework", ["vllm", "sglang", "trtllm"])
def test_cpu_check_without_serving_runtime_or_network(monkeypatch, framework):
    original = checker.metadata.version

    def version(name):
        if name == "tensorrt-llm":
            return "1.3.0rc26"
        if name in {"vllm", "sglang", "optuna-integration", "transformers"}:
            return "mock"
        return original(name)

    monkeypatch.setattr(checker.metadata, "version", version)
    integration = ModuleType("optuna_integration")
    integration.__path__ = []
    monkeypatch.setitem(sys.modules, "optuna_integration", integration)
    monkeypatch.setitem(sys.modules, "optuna_integration.wandb", ModuleType("wandb"))
    checker.check(framework, ROOT / "src/llm_optimizer")
