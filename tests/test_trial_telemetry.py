"""Execution evidence uses CPU-only processes and preserves benchmark contracts."""

import json
import shlex
import sys
import time
from types import SimpleNamespace

import optuna
import pytest

from llm_optimizer import cli, server_utils
from llm_optimizer.exceptions import ServerCleanupError, ServerNotReadyError
from llm_optimizer.trial_telemetry import TrialTelemetry, parse_startup_log


def test_phase_boundaries_and_partial_interruption(tmp_path, monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(
        "llm_optimizer.trial_telemetry.time.perf_counter", lambda: clock[0]
    )
    record = TrialTelemetry(tmp_path)
    with record.phase("launch_to_ready"):
        clock[0] += 4
    record.start("warmup")
    clock[0] += 2
    record.finish(KeyboardInterrupt())
    saved = json.loads(record.path.read_text())
    assert saved["status"] == "interrupted"
    assert saved["timings_s"] == {
        "launch_to_ready": 4,
        "warmup": 2,
        "runner_wall": 6,
        "orchestration": 0,
    }
    assert saved["incomplete_phases"] == ["warmup"]
    assert "benchmark" not in saved["timings_s"]
    assert saved["startup_observations"] == []


def test_rest_is_separate_from_runner_wall(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(
        "llm_optimizer.trial_telemetry.time.perf_counter", lambda: clock[0]
    )
    record = TrialTelemetry(tmp_path)
    clock[0] = 5
    record.finish()
    with record.rest():
        clock[0] += 10
    saved = json.loads(record.path.read_text())
    assert saved["timings_s"]["runner_wall"] == 5
    assert saved["timings_s"]["rest"] == 10
    assert saved["rest_status"] == "complete"


def test_explicit_log_observations_are_not_summed(tmp_path):
    path = tmp_path / "server.log"
    path.write_text(
        "rank 0 Loading model weights took 2 seconds\n"
        "rank 1 Loading model weights took 3 GiB memory and 2 seconds\n"
        "torch.compile takes 300 ms\n"
        "Graph capturing finished in 1.5 secs\n"
        "Autotuning took 1 s\n"
        "Loading model weights...\n"
        "Entire trial took 100 seconds\n"
    )
    observations = parse_startup_log(path)
    assert [o["seconds"] for o in observations] == [2, 2, 0.3, 1.5, 1]
    assert [o["phase"] for o in observations] == [
        "weight_loading",
        "weight_loading",
        "compilation",
        "graph_capture",
        "autotuning",
    ]
    assert observations[1]["source"].startswith("rank 1")


@pytest.mark.parametrize(
    "outcome",
    ["complete", "failed", "pruned", "interrupted", "startup_failed", "cleanup_failed"],
)
@pytest.mark.parametrize("mute", [True, False])
def test_runner_retains_logs_and_partial_results(tmp_path, monkeypatch, outcome, mute):
    # More output than a typical pipe buffer: capture must drain while launching.
    script = (
        "import time; print('x'*200000,flush=True); "
        "print('Loading weights took 2 seconds',flush=True); time.sleep(60)"
    )
    command = f"{shlex.quote(sys.executable)} -u -c {shlex.quote(script)}"
    processes = []
    record = TrialTelemetry(tmp_path / "attempts")

    def ready(url, process=None):
        processes.append(process)
        deadline = time.monotonic() + 5
        while record.log_path.stat().st_size < 200000:
            assert time.monotonic() < deadline, "Output pipe was not drained"
            time.sleep(0.01)
        if outcome == "startup_failed":
            raise ServerNotReadyError("not ready")
        return True

    monkeypatch.setattr(server_utils, "wait_for_server", ready)
    failure = {
        "failed": RuntimeError("benchmark failed"),
        "pruned": optuna.TrialPruned(),
        "interrupted": KeyboardInterrupt(),
    }.get(outcome)

    def benchmark(args):
        # The exact manifest is passed through, without mutation.
        assert args["_request_manifest"] == {"requests": ["unchanged"]}
        args["_checkpoint_callback"]({"completed": 1, "total_requests": 1, "failed": 0})
        with args["_telemetry"].phase("benchmark"):
            if failure is not None:
                raise failure
        return {
            "completed": 2,
            "total_requests": 2,
            "failed": 0,
            "output_throughput": 3,
        }

    monkeypatch.setitem(
        sys.modules,
        "llm_optimizer.bench_client",
        SimpleNamespace(run_benchmark=benchmark),
    )
    if outcome == "cleanup_failed":

        def cleanup(process):
            server_utils.terminate_process_top_down(process)
            raise ServerCleanupError("verification failed")

        monkeypatch.setattr(cli, "terminate_process_top_down", cleanup)

    output = tmp_path / "results.jsonl"

    def run():
        return cli.run_single_configuration(
            {"client_args": {}, "server_args": {}, "server_cmd_args": []},
            framework="vllm",
            model="model",
            server_cmd=command,
            managed_server=False,
            gpus=1,
            gpu_type="cpu-test",
            host="localhost",
            port=8000,
            ready_url="http://unused",
            mute_server=mute,
            constraints_for_viz=[],
            output_jsonl_path=output,
            telemetry=record,
            workload_manifest={"requests": ["unchanged"]},
            checkpoint_callback=lambda values: None,
        )

    if outcome == "complete":
        result, _ = run()
        assert result["results"]["output_throughput"] == 3
        assert (
            json.loads(output.read_text())["metadata"]["execution"]["timings_s"][
                "cleanup"
            ]
            >= 0
        )
    else:
        with pytest.raises(BaseException) as raised:
            run()
        assert raised.value.execution["attempt_id"] == record.directory.name
        assert not output.exists(), "Exceptional attempts must not become candidates"
    saved = json.loads(record.path.read_text())
    assert saved["status"] == ("failed" if outcome == "startup_failed" else outcome)
    assert "Loading weights took 2 seconds" in record.log_path.read_text()
    assert saved["timings_s"]["cleanup"] >= 0
    assert saved["timings_s"]["launch_to_ready"] >= 0
    assert all(p.poll() is not None for p in processes)
    if failure:
        assert saved["request_accounting"]["completed"] == 1
    if outcome == "startup_failed":
        assert saved["phase_status"]["launch_to_ready"] == "partial"
        assert saved["readiness_reached"] is False


@pytest.mark.parametrize(
    "cleanup_error", [ServerCleanupError("survivors"), KeyboardInterrupt()]
)
def test_original_failure_survives_cleanup_failure(
    tmp_path, monkeypatch, cleanup_error
):
    record = TrialTelemetry(tmp_path)
    monkeypatch.setattr(cli, "start_server", lambda *args: object())

    def fail_cleanup(process):
        raise cleanup_error

    monkeypatch.setattr(cli, "terminate_process_top_down", fail_cleanup)

    def fail_benchmark(args):
        raise optuna.TrialPruned("checkpoint")

    monkeypatch.setitem(
        sys.modules,
        "llm_optimizer.bench_client",
        SimpleNamespace(run_benchmark=fail_benchmark),
    )
    with pytest.raises(type(cleanup_error)):
        cli.run_single_configuration(
            {"client_args": {}, "server_args": {}, "server_cmd_args": []},
            framework="vllm",
            model="model",
            server_cmd="unused",
            managed_server=False,
            gpus=1,
            gpu_type="test",
            host="localhost",
            port=8000,
            ready_url="unused",
            mute_server=True,
            constraints_for_viz=[],
            telemetry=record,
        )
    saved = json.loads(record.path.read_text())
    assert saved["status"] == "cleanup_failed"
    assert "TrialPruned" in saved["original_error"]
    assert saved["cleanup_error"] == str(cleanup_error)


@pytest.mark.parametrize("prune", [False, True])
def test_client_preparation_warmup_and_measured_interval(tmp_path, monkeypatch, prune):
    from llm_optimizer import bench_client

    clock = [0.0]
    monkeypatch.setattr(time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(
        time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )
    record = TrialTelemetry(tmp_path)

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return text.split()

    def tokenizer(*args, **kwargs):
        clock[0] += 3
        return Tokenizer()

    calls = []

    async def request(**kwargs):
        clock[0] += 5 if not calls else 7
        calls.append(True)
        return bench_client.RequestFuncOutput(
            success=True,
            latency=0.2,
            ttft=0.05,
            output_len=3,
            generated_text="a b c",
            itl=[0.05],
        )

    def checkpoint(values):
        if prune:
            raise optuna.TrialPruned()

    monkeypatch.setattr(bench_client, "get_tokenizer", tokenizer)
    monkeypatch.setattr(bench_client, "check_chat_template", lambda model: True)
    monkeypatch.setattr(bench_client, "set_ulimit", lambda: None)
    monkeypatch.setitem(bench_client.ASYNC_REQUEST_FUNCS, "vllm", request)
    args = {
        "backend": "vllm",
        "model": "unused",
        "port": 8000,
        "disable_tqdm": True,
        "warmup_requests": 1,
        "_telemetry": record,
        "_request_manifest": {
            "requests": [{"prompt": "a", "prompt_len": 1, "output_len": 3}]
        },
        "_checkpoint_requests": [1],
        "_checkpoint_callback": checkpoint,
    }
    if prune:
        with pytest.raises(optuna.TrialPruned) as raised:
            bench_client.run_benchmark(args)
        record.finish(raised.value)
    else:
        result = bench_client.run_benchmark(args)
        record.finish()
        assert result["completed"] == 1
    timings = record.data["timings_s"]
    assert timings["client_preparation"] == 3
    assert timings["warmup"] == 5
    assert timings["benchmark"] == 7
    assert timings["orchestration"] == 1  # existing post-warmup pause
    assert record.data["phase_status"]["benchmark"] == (
        "partial" if prune else "complete"
    )
