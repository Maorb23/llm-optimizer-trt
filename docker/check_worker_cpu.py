"""Offline worker checks; no serving-runtime import, model load, or GPU use."""

import argparse
import asyncio
import copy
import hashlib
import importlib.metadata as metadata
import json
import tempfile
import zipfile
from pathlib import Path


def check(framework, expected_source):
    import optuna_integration.wandb  # noqa: F401

    import llm_optimizer
    import wandb  # noqa: F401
    from llm_optimizer import bench_client
    from llm_optimizer.hf_workload import fingerprint, validate_manifest
    from llm_optimizer.optimization import extract_objective, summarize_study
    from llm_optimizer.predefined.client import CLIENT_CONFIGS
    from llm_optimizer.remote.contracts import OptimizationRequest
    from llm_optimizer.remote.worker import _checkpoint
    from llm_optimizer.trial_telemetry import TrialTelemetry

    runtime = "tensorrt-llm" if framework == "trtllm" else framework
    required = (
        runtime,
        "llm-optimizer",
        "boto3",
        "optuna",
        "wandb",
        "optuna-integration",
        "aiohttp",
        "numpy",
        "scipy",
        "transformers",
        "huggingface-hub",
        "tqdm",
    )
    versions = {name: metadata.version(name) for name in required}
    if framework == "trtllm":
        assert versions[runtime] == "1.3.0rc26", versions[runtime]
    for name in ("torch", "triton", "flashinfer-python", "tensorrt"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    print(json.dumps({"framework": framework, "versions": versions}, indent=2))

    package = Path(llm_optimizer.__file__).resolve().parent
    files = [
        path
        for path in expected_source.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix in {".py", ".json"}
    ]
    assert files, "Expected source directory is empty"
    for path in files:
        actual = package / path.relative_to(expected_source)
        assert actual.is_file(), str(actual)
        assert (
            hashlib.sha256(actual.read_bytes()).digest()
            == hashlib.sha256(path.read_bytes()).digest()
        ), str(actual)
    print(f"Imported optimizer: {package}; {len(files)} source files match")

    manifest = {
        "tokenizer_revision": "a" * 40,
        "dataset": {"repo_id": "example/data", "revision": "b" * 40},
        "row_indices": [0],
        "requests": [{"prompt": "synthetic", "prompt_len": 1, "output_len": 1}],
    }
    manifest["sha256"] = fingerprint(manifest)
    assert validate_manifest(manifest) == manifest
    tampered = copy.deepcopy(manifest)
    tampered["requests"][0]["prompt"] = "changed"
    try:
        validate_manifest(tampered)
    except ValueError:
        pass
    else:
        raise AssertionError("Tampered workload was accepted")
    assert "workload_manifest" in OptimizationRequest.__dataclass_fields__
    assert "workload_path" in CLIENT_CONFIGS

    class Response:
        status = 200

        def __init__(self, events):
            self.content = self.lines(events)

        async def lines(self, events):
            for event in events:
                text = event if isinstance(event, str) else json.dumps(event)
                yield ("data: " + text + "\n").encode()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    class Session(Response):
        def post(self, **kwargs):
            return self

    original = bench_client._create_bench_client_session
    request = bench_client.RequestFuncInput(
        "synthetic",
        "http://unused/v1/completions",
        1,
        1,
        "fake",
        None,
        None,
        {},
    )
    try:
        for events, success in [
            (
                [
                    {"choices": [{"text": "hello"}]},
                    {"choices": []},
                    {"choices": [{"text": "", "finish_reason": "length"}]},
                    {"choices": [], "usage": {"completion_tokens": 1}},
                    "[DONE]",
                ],
                True,
            ),
            ([{"choices": [], "usage": {"completion_tokens": 0}}, "[DONE]"], False),
            ([{"choices": [{"text": "partial"}]}], False),
        ]:
            bench_client._create_bench_client_session = lambda events=events: Session(
                events
            )
            result = asyncio.run(bench_client.async_request_openai_completions(request))
            assert result.success is success, result.error_category
    finally:
        bench_client._create_bench_client_session = original

    error = {"request_index": 17, "category": "empty_response", "error": "empty"}
    accounting = {"completed": 99, "failed": 1, "total_requests": 100}
    try:
        extract_objective({**accounting, "output_throughput": 42}, "output_throughput")
    except ValueError:
        pass
    else:
        raise AssertionError("Incomplete benchmark received a score")

    import optuna

    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    trial.set_user_attr("failure_reason", "synthetic incomplete benchmark")
    trial.set_user_attr(
        "execution",
        {
            "request_accounting": accounting,
            "request_errors": [error],
        },
    )
    study.tell(trial, state=optuna.trial.TrialState.FAIL)
    failed = summarize_study(study, ["output_throughput"])["failed_trials"][0]
    assert failed["request_errors"] == [error]

    class Store:
        def upload(self, path, *args):
            return path.name

    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        telemetry = TrialTelemetry(folder)
        telemetry.data.update(request_accounting=accounting, request_errors=[error])
        telemetry.save()
        assert json.loads(telemetry.path.read_text())["request_errors"] == [error]
        attempt = folder / "results-attempts" / "synthetic"
        attempt.mkdir(parents=True)
        (attempt / "execution.json").write_text(
            json.dumps(
                {
                    "request_accounting": accounting,
                    "request_errors": [error],
                    "commands": "excluded",
                    "environment": {"excluded": True},
                }
            )
        )
        artifacts = _checkpoint(Store(), folder, "synthetic", framework)
        assert "attempts.zip" in artifacts
        with zipfile.ZipFile(folder / "attempts.zip") as archive:
            record = json.loads(
                archive.read("results-attempts/synthetic/execution.json")
            )
        assert record == {"request_accounting": accounting, "request_errors": [error]}
    print(
        "PASS: stream parser, frozen workload, W&B imports, invalid scoring, "
        "failure summaries, telemetry, and filtered attempts.zip"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("framework", choices=["vllm", "sglang", "trtllm"])
    parser.add_argument("--expected-source", type=Path, required=True)
    args = parser.parse_args()
    check(args.framework, args.expected_source)
