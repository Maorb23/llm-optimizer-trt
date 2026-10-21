"""Load definitions only: never execute saved GPU, download or study-edit cells."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import optuna
import pytest
from test_optuna_notebook import lab as reference_lab_fixture

from llm_optimizer.experiment.lab import (
    BenchmarkFailure,
    Measurement,
    RepositoryBenchmark,
)

lab = reference_lab_fixture
DOCS = Path(__file__).resolve().parents[1] / "docs"


def definitions(filename):
    notebook = json.loads((DOCS / filename).read_text())
    nodes = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            nodes.extend(ast.parse("".join(cell["source"])).body)
    return nodes


def adapter(namespace, filename):
    selected = [
        node
        for node in definitions(filename)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and node.name
        in {
            "RepositoryBenchmark",
            "serialize_engine_config",
            "workload_for",
            "regime_key",
            "render_engine_config",
            "validate_measurement",
            "measurement_failure",
        }
    ]
    namespace.update(
        BenchmarkFailure=BenchmarkFailure,
        Measurement=Measurement,
        SharedRepositoryBenchmark=RepositoryBenchmark,
    )
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), filename, "exec"), namespace
    )
    return namespace["RepositoryBenchmark"]


def test_v2_search_functions_and_controls_remain_identical():
    old = definitions("optuna_inference_optimization_lab_prune.ipynb")
    new = definitions("optuna_inference_optimization_lab_v2.ipynb")
    for kind in (ast.FunctionDef, ast.Assign):

        def filtered(nodes, kind=kind):
            return [
                ast.dump(node)
                for node in nodes
                if isinstance(node, kind)
                and not (
                    isinstance(node, ast.Assign)
                    and any(
                        isinstance(t, ast.Name)
                        and t.id in {"p", "code", "old", "new", "path", "lines"}
                        for t in node.targets
                    )
                )
                and (
                    kind is ast.Assign
                    or node.name not in {"record_measurement", "run_study"}
                )
            ]

        assert filtered(old) == filtered(new)
    notebook = json.loads(
        (DOCS / "optuna_inference_optimization_lab_v2.ipynb").read_text()
    )
    assert not any(
        "p.write_text(" in "".join(cell["source"])
        or "path.write_text(code.replace" in "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


@pytest.mark.parametrize("framework", ["vllm", "sglang", "trtllm"])
def test_v2_adapter_preserves_native_settings_and_manifest(
    lab, tmp_path, monkeypatch, framework
):
    from llm_optimizer import bench_client, cli

    prepare = Mock(return_value=({"requests": []}, {"workload_sha256": "frozen"}))
    monkeypatch.setattr(bench_client, "prepare_workload_manifest", prepare)
    calls = []

    def runner(settings, **kwargs):
        calls.append((settings, kwargs))
        return {
            "results": {
                "completed": 2,
                "total_requests": 2,
                "failed": 0,
                "output_throughput": 10,
                "p95_ttft_ms": 20,
                "p95_e2e_latency_ms": 30,
            }
        }, None

    monkeypatch.setattr(cli, "run_single_configuration", runner)
    results = []
    manifests = []
    for filename in (
        "optuna_inference_optimization_lab_prune.ipynb",
        "optuna_inference_optimization_lab_v2.ipynb",
    ):
        namespace = dict(lab)
        namespace["PHASE_REQUESTS"] = {"A": 2, "B": 2}
        namespace["RUN_DIR"] = tmp_path
        cls = adapter(namespace, filename)
        if filename.endswith("v2.ipynb"):
            assert issubclass(cls, RepositoryBenchmark)
        benchmark = cls(
            framework,
            tmp_path,
            "test",
            1,
            rest_seconds=0,
            environment={
                "framework_version": "pinned",
                "cuda_version": "pinned",
                "driver_version": "pinned",
                "model_revision": "immutable",
                "native_config_verified": True,
            },
        )
        sampled = {
            key: values[0] for key, values in namespace["SPACES"][1][framework].items()
        }
        config = namespace["render_engine_config"](framework, 1, sampled)
        result = benchmark.measure(framework, config, 1)
        results.append(result)
        manifests.append(benchmark.manifest)
    assert calls[0][0] == calls[1][0]
    # Telemetry is the only additional runner argument.
    assert calls[0][1] == {
        key: value for key, value in calls[1][1].items() if key != "telemetry"
    }
    assert prepare.call_args_list[0] == prepare.call_args_list[1]
    assert manifests[0] == manifests[1]
    assert results[0].output_throughput == results[1].output_throughput
    assert results[0].p95_ttft_ms == results[1].p95_ttft_ms
    assert results[0].p95_e2e_latency_ms == results[1].p95_e2e_latency_ms
    assert results[1].execution["status"] == "complete"


@pytest.mark.parametrize(
    "exception", [optuna.TrialPruned(), KeyboardInterrupt(), RuntimeError("failed")]
)
def test_v2_adapter_preserves_exception_accounting(
    lab, tmp_path, monkeypatch, exception
):
    from llm_optimizer import bench_client, cli

    monkeypatch.setattr(
        bench_client, "prepare_workload_manifest", lambda args: ({"requests": []}, {})
    )

    def runner(settings, **kwargs):
        kwargs["checkpoint_callback"](
            {
                "completed": 1,
                "total_requests": 1,
                "failed": 0,
                "output_throughput": 10,
                "p95_ttft_ms": 20,
                "p95_e2e_latency_ms": 30,
            }
        )
        raise exception

    monkeypatch.setattr(cli, "run_single_configuration", runner)
    namespace = dict(lab)
    namespace["RUN_DIR"] = tmp_path
    cls = adapter(namespace, "optuna_inference_optimization_lab_v2.ipynb")
    benchmark = cls(
        "vllm",
        tmp_path,
        "test",
        1,
        rest_seconds=0,
        environment={
            "framework_version": "pinned",
            "cuda_version": "pinned",
            "driver_version": "pinned",
            "model_revision": "immutable",
            "native_config_verified": True,
        },
    )
    expected = (
        type(exception) if not isinstance(exception, RuntimeError) else BenchmarkFailure
    )
    with pytest.raises(expected) as raised:
        benchmark.measure("vllm", {}, 1, checkpoint_callback=lambda m: None)
    assert raised.value.request_accounting == {
        "completed": 1,
        "total_requests": 1,
        "failed": 0,
    }
    assert raised.value.gpu_seconds >= 0
    saved = json.loads(Path(raised.value.execution["attempt_artifact"]).read_text())
    assert saved["status"] in {"pruned", "interrupted", "failed"}


def test_caller_interrupt_is_recorded_after_worker_cleanup(tmp_path, monkeypatch):
    from llm_optimizer import bench_client, cli

    cleaned = []

    def runner(*args, **kwargs):
        cleaned.append(True)
        return {
            "results": {
                "completed": 1,
                "total_requests": 1,
                "failed": 0,
                "output_throughput": 10,
                "p95_ttft_ms": 20,
                "p95_e2e_latency_ms": 30,
            }
        }, None

    class Pool:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, execute):
            def result():
                execute()
                # Models an interrupt in the caller, after the executor has
                # waited for its worker's normal cleanup.
                raise KeyboardInterrupt()

            return SimpleNamespace(result=result)

    monkeypatch.setattr("concurrent.futures.ThreadPoolExecutor", Pool)
    monkeypatch.setattr(cli, "run_single_configuration", runner)
    monkeypatch.setattr(
        bench_client, "prepare_workload_manifest", lambda args: ({"requests": []}, {})
    )
    benchmark = RepositoryBenchmark(
        "vllm",
        tmp_path,
        "test",
        1,
        rest_seconds=0,
        environment={
            "framework_version": "pinned",
            "cuda_version": "pinned",
            "driver_version": "pinned",
            "model_revision": "immutable",
            "native_config_verified": True,
        },
        _context=SimpleNamespace(
            RUN_DIR=tmp_path,
            PHASE_MODELS={"A": "target"},
            workload_for=lambda *args: {"num_prompts": 1, "seed": 1},
            serialize_engine_config=lambda *args: {},
            validate_measurement=lambda m: m,
        ),
    )
    with pytest.raises(KeyboardInterrupt) as raised:
        benchmark.measure("vllm", {}, 1)
    assert cleaned == [True]
    saved = json.loads(Path(raised.value.execution["attempt_artifact"]).read_text())
    assert saved["status"] == "interrupted"
    assert raised.value.request_accounting["completed"] == 1
