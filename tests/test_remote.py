import base64
import json
import shlex
import sqlite3
import subprocess

import pytest

from llm_optimizer.remote.adapters import (
    NebiusAdapter,
    NebiusCommandError,
    S3Artifacts,
)
from llm_optimizer.remote.contracts import OptimizationRequest
from llm_optimizer.remote.controller import Controller
from llm_optimizer.remote.worker import _checkpoint, optimizer_command, run_worker


def request(**changes):
    data = {
        "model": "org/model",
        "frameworks": ["vllm", "sglang"],
        "gpu_type": "gpu-h100-sxm",
        "images": {"vllm": "vllm:image", "sglang": "sglang:image"},
        "bucket": "results",
        "dry_run": False,
    }
    data.update(changes)
    return data


class FakeNebius:
    def __init__(self):
        self.submitted = []
        self.states = {}
        self.cancelled = []

    def submit(self, payload):
        self.submitted.append(payload)
        job_id = f"aijob-{len(self.submitted)}"
        self.states[job_id] = "running"
        return job_id

    def status(self, job_id):
        return self.states[job_id]

    def cancel(self, job_id):
        self.cancelled.append(job_id)


class FakeStore:
    manifests = {}
    reports = {}
    progress_data = {}
    execution_logs = {}

    def __init__(self, bucket, prefix):
        pass

    def manifest(self, run_id, framework):
        return self.manifests.get((run_id, framework))

    def progress(self, run_id, framework):
        return self.progress_data.get((run_id, framework))

    def execution_log_tail(self, run_id, framework):
        return self.execution_logs.get((run_id, framework))

    def put_report(self, run_id, report):
        self.reports[run_id] = report


def test_validation_and_complete_optuna_propagation(tmp_path):
    with pytest.raises(ValueError, match="preset"):
        OptimizationRequest.parse(request(gpu_count=2))
    with pytest.raises(ValueError, match="timeout_minutes"):
        OptimizationRequest.parse(request(timeout_minutes=15))
    with pytest.raises(ValueError, match="speculative"):
        OptimizationRequest.parse(request(speculative_heads="eagle"))
    optuna = {
        "trials": 11,
        "timeout": 200,
        "objective": "p95_ttft_ms",
        "seed": 7,
        "sampler": "RandomSampler",
        "sampler_kwargs": {"seed": 7},
        "pruner": "MedianPruner",
        "pruner_kwargs": {"n_startup_trials": 3},
        "storage": "postgresql://db.example/optimizer",
        "study_name": "example",
        "server_args": ["max_num_seqs=int(1,4)"],
        "client_args": ["max_concurrency=int(1,4)"],
    }
    data = request(search_strategy="optuna", optuna=optuna)
    command = optimizer_command(
        {"request": data, "framework": "vllm"}, tmp_path / "results.json"
    )
    for flag in (
        "--optuna-trials",
        "--optuna-timeout",
        "--optuna-objective",
        "--optuna-seed",
        "--optuna-sampler",
        "--optuna-sampler-kwargs",
        "--optuna-pruner",
        "--optuna-pruner-kwargs",
        "--optuna-storage",
        "--optuna-study-name",
        "--optuna-server-args",
        "--optuna-client-args",
    ):
        assert flag in command
    data["optuna_by_framework"] = {
        "sglang": {"trials": 5, "server_args": ["max_running_requests=int(1,4)"]}
    }
    other = optimizer_command(
        {"request": data, "framework": "sglang"}, tmp_path / "sglang.json"
    )
    assert other[other.index("--optuna-trials") + 1] == "5"
    assert "max_running_requests=int(1,4)" in other


def test_trtllm_uses_supported_revision_argument(tmp_path):
    data = request(
        frameworks=["trtllm"],
        images={"trtllm": "trtllm:image"},
        revision="abc123",
    )
    command = optimizer_command(
        {"request": data, "framework": "trtllm"}, tmp_path / "results.json"
    )
    assert "revision=abc123" in command
    assert not any("hf_revision=" in argument for argument in command)


def test_nebius_command_uses_framework_image_and_gpu(monkeypatch):
    seen = []

    def run(command, **kwargs):
        seen.append(command)
        return type("Completed", (), {"stdout": '{"id":"aijob-abc"}'})()

    adapter = NebiusAdapter("project", run)
    adapter.submit({"run_id": "run", "framework": "vllm", "request": request()})
    command = seen[0]
    assert command[command.index("--platform") + 1] == "gpu-h100-sxm"
    assert command[command.index("--image") + 1] == "vllm:image"
    assert command[command.index("--preset") + 1] == "1gpu-16vcpu-96gb"
    assert command[command.index("--name") + 1] == "llm-opt-run-vllm"
    assert command[command.index("--container-command") + 1] == "python3"
    assert command.count("--args") == 1
    assert command[command.index("--args") + 1] == "-m llm_optimizer.remote.worker"


def test_nebius_reads_job_id_from_metadata():
    def run(command, **kwargs):
        return type("Completed", (), {"stdout": '{"metadata":{"id":"aijob-abc"}}'})()

    adapter = NebiusAdapter("project", run)
    assert (
        adapter.submit({"run_id": "run", "framework": "vllm", "request": request()})
        == "aijob-abc"
    )
    assert adapter.find_by_name("run", "vllm") == "aijob-abc"


def test_nebius_debug_job_has_ssh_key_and_sleep_on_failure(tmp_path, monkeypatch):
    key = tmp_path / "id_ed25519.pub"
    key.write_text("ssh-ed25519 example-key")
    monkeypatch.setenv("NEBIUS_DEBUG_SSH_PUBLIC_KEY", str(key))
    seen = []

    def run(command, **kwargs):
        seen.append(command)
        return type("Completed", (), {"stdout": '{"metadata":{"id":"aijob-abc"}}'})()

    NebiusAdapter("project", run).submit(
        {"run_id": "run", "framework": "vllm", "request": request()}
    )
    command = seen[0]
    assert command[command.index("--ssh-key") + 1] == str(key)
    assert command[command.index("--container-command") + 1] == "bash"
    assert command.count("--args") == 1
    args = shlex.split(command[command.index("--args") + 1])
    assert args[0] == "-lc"
    assert "python3 -m llm_optimizer.remote.worker" in args[1]
    assert "sleep 3600" in args[1]


def test_nebius_debug_job_rejects_missing_ssh_key(monkeypatch, tmp_path):
    monkeypatch.setenv("NEBIUS_DEBUG_SSH_PUBLIC_KEY", str(tmp_path / "missing.pub"))

    with pytest.raises(ValueError, match="existing file"):
        NebiusAdapter("project").submit(
            {"run_id": "run", "framework": "vllm", "request": request()}
        )


def test_preemptible_job_is_opt_in_and_worker_payload_stays_compatible():
    seen = []

    def run(command, **kwargs):
        seen.append(command)
        return type("Completed", (), {"stdout": '{"id":"aijob-abc"}'})()

    adapter = NebiusAdapter("project", run)
    payload = {"run_id": "run", "framework": "vllm", "request": request()}
    adapter.submit(payload)
    assert "--preemptible" not in seen[-1]
    payload["request"] = request(preemptible=True)
    adapter.submit(payload)
    assert "--preemptible" in seen[-1]
    encoded = next(
        arg.split("=", 1)[1]
        for arg in seen[-1]
        if arg.startswith("LLM_OPTIMIZER_PAYLOAD_B64=")
    )
    worker_payload = json.loads(base64.b64decode(encoded))
    assert "preemptible" not in worker_payload["request"]
    with pytest.raises(ValueError, match="preemptible"):
        OptimizationRequest.parse(request(preemptible="true"))


def test_nebius_failure_logs_without_persisting_command_secrets(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "example-secret-never-log")

    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(
            1, command, stderr="preset unavailable in project"
        )

    adapter = NebiusAdapter("project", fail)
    with pytest.raises(NebiusCommandError, match="preset unavailable in project"):
        adapter.submit({"run_id": "run", "framework": "vllm", "request": request()})

    controller = Controller(tmp_path / "state.db", adapter, FakeStore)
    run = controller.submit(request(frameworks=["vllm"], images={"vllm": "image"}))
    controller.tick()
    job = controller.status(run["run_id"])["jobs"][0]
    assert job["state"] == "submission_unknown"
    assert "preset unavailable in project" in job["error"]
    assert "preset unavailable in project" in caplog.text
    assert "example-secret-never-log" not in job["error"] + caplog.text
    assert "LLM_OPTIMIZER_PAYLOAD_B64" not in job["error"] + caplog.text


def test_nebius_failure_omits_stderr_containing_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "example-secret-never-log")

    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(
            1, command, stderr="AWS_SECRET_ACCESS_KEY=example-secret-never-log"
        )

    adapter = NebiusAdapter("project", fail)
    controller = Controller(tmp_path / "state.db", adapter, FakeStore)
    run = controller.submit(request(frameworks=["vllm"], images={"vllm": "image"}))
    controller.tick()
    error = controller.status(run["run_id"])["jobs"][0]["error"]
    assert "output omitted" in error
    assert "example-secret-never-log" not in error


def test_existing_credential_error_is_redacted_on_startup(tmp_path):
    path = tmp_path / "state.db"
    controller = Controller(path, FakeNebius(), FakeStore)
    run = controller.submit(request(frameworks=["vllm"], images={"vllm": "image"}))
    with sqlite3.connect(path) as db:
        db.execute(
            "UPDATE jobs SET error=? WHERE run_id=?",
            ("AWS_SECRET_ACCESS_KEY=old-example-secret", run["run_id"]),
        )
    reopened = Controller(path, FakeNebius(), FakeStore)
    error = reopened.status(run["run_id"])["jobs"][0]["error"]
    assert "old-example-secret" not in error
    assert "redacted" in error


def test_restart_partial_failure_idempotency_and_cancel(tmp_path):
    provider = FakeNebius()
    db = tmp_path / "controller.db"
    controller = Controller(db, provider, FakeStore, concurrency=2)
    run = controller.submit(request(), "same")
    run_id = run["run_id"]
    assert controller.submit(request(), "same")["run_id"] == run_id
    controller.tick()
    assert len(provider.submitted) == 2
    FakeStore.progress_data[(run_id, "vllm")] = {"completed_trials": 3}
    restarted = Controller(db, provider, FakeStore, concurrency=2)
    restarted.tick()
    assert len(provider.submitted) == 2
    assert restarted.status(run_id)["jobs"][1]["progress"] == {"completed_trials": 3}
    FakeStore.manifests[(run_id, "vllm")] = {
        "run_id": run_id,
        "framework": "vllm",
        "status": "completed",
        "model": "org/model",
        "gpu_type": "gpu-h100-sxm",
    }
    provider.states["aijob-2"] = "completed"
    provider.states["aijob-1"] = "failed"
    restarted.tick()
    result = restarted.results(run_id)
    assert result["state"] == "partial"
    assert result["manifests"]["vllm"]["status"] == "completed"
    assert FakeStore.reports[run_id]["state"] == "partial"
    other = restarted.submit(request())
    restarted.tick()
    restarted.cancel(other["run_id"], "sglang")
    assert provider.cancelled


def test_single_framework_cancellation_marks_parent_cancelled(tmp_path):
    provider = FakeNebius()
    controller = Controller(tmp_path / "state.db", provider, FakeStore)
    run = controller.submit(request(frameworks=["vllm"], images={"vllm": "image"}))
    controller.tick()

    status = controller.cancel(run["run_id"], "vllm")

    assert status["jobs"][0]["state"] == "cancelled"
    assert status["state"] == "cancelled"
    assert provider.cancelled == ["aijob-1"]


def test_failed_job_reports_uploaded_exception_in_serve_and_status(tmp_path, caplog):
    provider = FakeNebius()
    controller = Controller(tmp_path / "state.db", provider, FakeStore)
    run = controller.submit(request(frameworks=["vllm"], images={"vllm": "image"}))
    controller.tick()
    run_id = run["run_id"]
    FakeStore.manifests[(run_id, "vllm")] = {
        "run_id": run_id,
        "framework": "vllm",
        "status": "failed",
        "error": "RuntimeError: optimizer exited with code 1",
    }
    FakeStore.execution_logs[(run_id, "vllm")] = (
        "Traceback (most recent call last):\n"
        "AttributeError: 'list' object has no attribute 'items'\n"
    )
    provider.states["aijob-1"] = "failed"

    controller.tick()

    error = controller.status(run_id)["jobs"][0]["error"]
    assert "AttributeError: 'list' object has no attribute 'items'" in error
    assert error in caplog.text
    FakeStore.manifests.pop((run_id, "vllm"))
    FakeStore.execution_logs.pop((run_id, "vllm"))


def test_dry_run_and_uncertain_submission_do_not_duplicate(tmp_path):
    provider = FakeNebius()
    controller = Controller(tmp_path / "state.db", provider, FakeStore)
    run = controller.submit(request(dry_run=True))
    controller.tick()
    assert controller.status(run["run_id"])["state"] == "dry_run"

    class Uncertain(FakeNebius):
        def submit(self, payload):
            raise TimeoutError("response lost")

    controller = Controller(tmp_path / "uncertain.db", Uncertain(), FakeStore)
    run = controller.submit(request(frameworks=["vllm"], images={"vllm": "image"}))
    controller.tick()
    Controller(tmp_path / "uncertain.db", Uncertain(), FakeStore).tick()
    assert controller.status(run["run_id"])["jobs"][0]["state"] == "submission_unknown"


def test_uncertain_submission_reconciles_by_nebius_name(tmp_path):
    class LostResponse(FakeNebius):
        def submit(self, payload):
            raise TimeoutError("response lost after create")

        def find_by_name(self, run_id, framework):
            self.states["aijob-found"] = "running"
            return "aijob-found"

    provider = LostResponse()
    controller = Controller(tmp_path / "state.db", provider, FakeStore)
    run = controller.submit(request(frameworks=["vllm"], images={"vllm": "image"}))
    controller.tick()
    assert controller.status(run["run_id"])["jobs"][0]["state"] == "submission_unknown"
    Controller(tmp_path / "state.db", provider, FakeStore).tick()
    job = controller.status(run["run_id"])["jobs"][0]
    assert job["nebius_id"] == "aijob-found"
    assert job["state"] == "running"


def test_cancel_during_uncertain_submission_reconciles_then_cancels(tmp_path):
    class Provider(FakeNebius):
        def find_by_name(self, run_id, framework):
            return "aijob-existing"

    provider = Provider()
    controller = Controller(tmp_path / "state.db", provider, FakeStore)
    run = controller.submit(request(frameworks=["vllm"], images={"vllm": "image"}))
    controller._update(run["run_id"], "vllm", "submission_unknown")
    controller.cancel(run["run_id"], "vllm")
    assert controller.status(run["run_id"])["jobs"][0]["state"] == "cancelling"
    controller.tick()
    assert provider.cancelled == ["aijob-existing"]
    assert controller.status(run["run_id"])["jobs"][0]["state"] == "cancelled"


def test_uncertain_cancellation_does_not_consume_concurrency_slot(tmp_path):
    provider = FakeNebius()
    controller = Controller(tmp_path / "state.db", provider, FakeStore, concurrency=1)
    cancelled = controller.submit(
        request(frameworks=["vllm"], images={"vllm": "image"})
    )
    controller._update(cancelled["run_id"], "vllm", "submission_unknown")
    controller.cancel(cancelled["run_id"], "vllm")
    queued = controller.submit(request(frameworks=["vllm"], images={"vllm": "image"}))

    controller.tick()

    assert controller.status(cancelled["run_id"])["jobs"][0]["state"] == "cancelling"
    assert controller.status(queued["run_id"])["jobs"][0]["state"] == "running"


def test_s3_prefix_and_manifest_integrity(tmp_path):
    class Client:
        def __init__(self):
            self.objects = {}

        def upload_file(self, source, bucket, key):
            self.objects[(bucket, key)] = open(source, "rb").read()

        def get_object(self, Bucket, Key, Range=None):
            import io

            data = self.objects[(Bucket, Key)]
            if Range:
                data = data[-int(Range.removeprefix("bytes=-")) :]
            return {"Body": io.BytesIO(data)}

        def download_file(self, bucket, key, destination):
            with open(destination, "wb") as stream:
                stream.write(self.objects[(bucket, key)])

    client = Client()
    store = S3Artifacts("bucket", "prefix", client=client)
    path = tmp_path / "manifest.json"
    path.write_text('{"run_id":"one","framework":"vllm","status":"completed"}')
    assert (
        store.upload(path, "one", "vllm") == "s3://bucket/prefix/one/vllm/manifest.json"
    )
    assert store.manifest("one", "vllm")["status"] == "completed"
    log = tmp_path / "execution.log"
    log.write_text("AttributeError: test failure\n")
    store.upload(log, "one", "vllm")
    assert store.execution_log_tail("one", "vllm") == "AttributeError: test failure\n"
    target = tmp_path / "copy.json"
    assert store.download(target, "one", "vllm", "manifest.json")
    assert target.read_bytes() == path.read_bytes()


def test_checkpoint_bundles_study_and_trial_records(tmp_path):
    import sqlite3
    import zipfile

    database = tmp_path / "results.optuna.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE trials (number INTEGER)")
        connection.execute("INSERT INTO trials VALUES (3)")
    (tmp_path / "results.jsonl").write_text('{"trial":3}\n')

    class Store:
        def upload(self, path, run_id, framework):
            return f"s3://bucket/{run_id}/{framework}/{path.name}"

    artifacts = _checkpoint(Store(), tmp_path, "run", "vllm")
    assert "checkpoint.zip" in artifacts
    with zipfile.ZipFile(tmp_path / "checkpoint.zip") as archive:
        assert set(archive.namelist()) == {"results.optuna.db", "results.jsonl"}
        assert archive.read("results.jsonl") == b'{"trial":3}\n'


def test_worker_uploads_result_manifest_and_deployment(tmp_path, monkeypatch):
    import types

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
        if command[0] == "nvidia-smi":
            return types.SimpleNamespace(stdout="NVIDIA H100\n", returncode=0)
        assert "--dry-run" in command
        return types.SimpleNamespace(returncode=0)

    class Process:
        returncode = 0

        def __init__(self, command, **kwargs):
            output = command[command.index("--output-json") + 1]
            with open(output, "w") as stream:
                stream.write('{"metadata":{"total_tests":1},"best_configurations":{}}')

        def poll(self):
            return 0

    monkeypatch.setattr(worker.subprocess, "run", preflight)
    monkeypatch.setattr(worker, "_version", lambda package: "1.0")
    monkeypatch.setattr(
        huggingface_hub.HfApi,
        "model_info",
        lambda *args, **kwargs: types.SimpleNamespace(sha="abc123"),
    )
    store = Store()
    payload = {
        "run_id": "run",
        "framework": "vllm",
        "request": request(frameworks=["vllm"], images={"vllm": "vllm:image"}),
    }
    result = run_worker(payload, store=store, work_dir=tmp_path, runner=Process)
    assert result["status"] == "completed"
    assert result["resolved_revision"] == "abc123"
    assert {"results.json", "deployment.json", "manifest.json"} <= set(store.files)
    assert result["sha256"]["results.json"]


def test_scalar_pruning_request_and_minimum_propagation(tmp_path):
    data = request(
        search_strategy="optuna",
        optuna={
            "objectives": ["mean_e2e_latency_ms"],
            "pruner": "MedianPruner",
            "min_completed_requests": 64,
        },
    )
    parsed = OptimizationRequest.parse(data)
    assert parsed.optuna["objectives"] == ["mean_e2e_latency_ms"]
    command = optimizer_command(
        {"request": data, "framework": "vllm"}, tmp_path / "out.json"
    )
    assert command[command.index("--optuna-min-completed-requests") + 1] == "64"
    for minimum in (0, -1, True, 1.5):
        data["optuna"]["min_completed_requests"] = minimum
        with pytest.raises(ValueError, match="positive integer"):
            OptimizationRequest.parse(data)


@pytest.mark.parametrize(
    "changes",
    [
        {
            "optuna": {
                "objectives": ["mean_e2e_latency_ms", "output_throughput"],
                "pruner": "MedianPruner",
            }
        },
        {"constraints": "ttft:mean<=100ms", "optuna": {"pruner": "MedianPruner"}},
        {"optuna": {"pruner": "ThresholdPruner"}},
    ],
)
def test_remote_rejects_unsupported_pruning(changes):
    with pytest.raises(ValueError, match="pruning|Pruning"):
        OptimizationRequest.parse(request(search_strategy="optuna", **changes))
