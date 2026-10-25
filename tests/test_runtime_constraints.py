"""CPU checks of the TRT build's runtime preservation guard."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "runtime_constraints",
    Path(__file__).resolve().parents[1] / "docker/runtime_constraints.py",
)
constraints = importlib.util.module_from_spec(spec)
spec.loader.exec_module(constraints)


def test_snapshot_preserves_versions_and_excludes_replaced_packages(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        constraints.metadata,
        "distributions",
        lambda: [
            SimpleNamespace(metadata={"Name": name}, version=version)
            for name, version in [
                ("Torch", "2.9.0+cu130"),
                ("tensorrt_llm", "1.3.0rc26"),
                ("nvidia-ml-py", "13.0.0"),
                ("nvidia-ml-py3", "7.352.0"),
                ("llm-optimizer", "0.0.1"),
            ]
        ],
    )
    prefix = tmp_path / "base"
    constraints.snapshot(prefix)
    assert prefix.with_suffix(".txt").read_text() == (
        "nvidia-ml-py==13.0.0\ntensorrt-llm==1.3.0rc26\ntorch==2.9.0+cu130\n"
    )
    constraints.verify(prefix)


@pytest.mark.parametrize("current", [{"torch": "2.10"}, {}])
def test_verify_rejects_upgrades_and_removals(tmp_path, monkeypatch, current):
    prefix = tmp_path / "base"
    monkeypatch.setattr(constraints, "installed", lambda: {"torch": "2.9"})
    constraints.snapshot(prefix)
    monkeypatch.setattr(constraints, "installed", lambda: current)
    with pytest.raises(RuntimeError, match="Serving base packages changed"):
        constraints.verify(prefix)
