import sys
import types

from llm_optimizer.wandb_tracking import WandbTracker, stable_run_id


class FakeRun:
    def __init__(self):
        self.logged = []
        self.artifacts = []
        self.finished = False

    def log(self, payload, step=None):
        self.logged.append((payload, step))

    def log_artifact(self, artifact):
        self.artifacts.append(artifact)

    def finish(self):
        self.finished = True


def test_disabled_without_project(monkeypatch):
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    assert WandbTracker.start(
        study_name="study", framework="vllm", model="model", gpu="H100",
        run_id="run", objective="throughput",
    ).run is None


def test_mocked_tracking_uses_stable_resume_id_and_never_forwards_key(
    monkeypatch, tmp_path
):
    run = FakeRun()
    calls = []

    class Artifact:
        def __init__(self, **kwargs):
            self.files = []

        def add_file(self, path, name):
            self.files.append(name)

    fake = types.SimpleNamespace(
        init=lambda **kwargs: calls.append(kwargs) or run,
        Settings=lambda **kwargs: kwargs,
        Artifact=Artifact,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setenv("WANDB_PROJECT", "project")
    monkeypatch.setenv("WANDB_API_KEY", "never-record-me")
    monkeypatch.setenv("WANDB_TAGS", "one, two")
    tracker = WandbTracker.start(
        study_name="study", framework="vllm", model="model", gpu="H100",
        run_id="controller-run", objective="output_throughput",
    )
    assert calls[0]["id"] == stable_run_id("controller-run", "study", "vllm")
    assert calls[0]["resume"] == "allow"
    assert "never-record-me" not in repr(calls)
    trial = types.SimpleNamespace(
        number=3, state=types.SimpleNamespace(name="COMPLETE"), value=12.5,
        params={"server:max_num_seqs": 16},
    )
    tracker.trial(trial, {"output_throughput": 12.5, "label": "ignored"})
    result = tmp_path / "results.json"
    result.write_text("{}")
    tracker.artifacts([result, tmp_path / "missing.log"], name="final")
    tracker.finish()
    assert run.logged[0][0]["trial/state"] == "COMPLETE"
    assert run.logged[0][0]["benchmark/output_throughput"] == 12.5
    assert run.artifacts[0].files == ["results.json"]
    assert run.finished


def test_wandb_failures_are_swallowed(monkeypatch):
    monkeypatch.setenv("WANDB_PROJECT", "project")
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        types.SimpleNamespace(
            init=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
            Settings=lambda **kwargs: kwargs,
        ),
    )
    tracker = WandbTracker.start(
        study_name="study", framework="sglang", model="model", gpu="L40S",
        run_id="run", objective="latency",
    )
    assert tracker.run is None
