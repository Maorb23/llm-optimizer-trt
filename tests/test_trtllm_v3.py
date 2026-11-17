"""CPU wiring, conditional search, telemetry, and lifecycle checks; no GPU claims."""

import asyncio
import copy
import json
import shlex
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import optuna
import pytest
import yaml

from llm_optimizer.experiment import phase1
from llm_optimizer.experiment.telemetry_trtllm_v3 import (
    DEFAULT_TELEMETRY,
    Collector,
    WindowTelemetry,
    _active,
    aggregate_prometheus,
    counted_request,
    summarize,
    unavailable_telemetry,
)
from llm_optimizer.experiment.trtllm_policy_v3 import (
    DEFAULT_OPTIONAL_PARAMS,
    TRTLLMLabV3,
    V3Logger,
    merge_config,
    native_command,
    validate_options,
)

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "docs/phase1_experiments_trtllm_v3.ipynb"


@pytest.fixture
def ns(tmp_path, monkeypatch):
    monkeypatch.setattr(phase1, "verify_precision", phase1.verify_precision)
    namespace = {}
    prefixes = (
        "from pathlib",
        "CAMPAIGN",
        "# Illustrative SLOs",
        "PREFILL_BUDGETS",
        "# Enable only the additional",
        "PRECISION_CHOICES",
        "# Phase B uses",
        "# Notebook-only TRT-LLM",
        "def make_runner",
        "# Attach the adapter",
    )
    for cell in json.loads(NOTEBOOK.read_text())["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and source.startswith(prefixes):
            exec(compile(source, "<trt-v3-cell>", "exec"), namespace)
    namespace.update(
        RUN_REAL_GPU=False,
        WANDB_ENABLED=False,
        RUN_ROOT=tmp_path,
        TRIALS=2,
        SEEDS=[11],
        SAMPLER_NAMES=["tpe"],
    )
    namespace["PRECISION_CHOICES"]["trtllm"]["weight_precision"] = ["bf16"]
    return namespace


def params(lab, concurrency, **overrides):
    values = {
        key: domain[0]
        for key, domain in lab.sampling_menu("trtllm", concurrency, "expanded").items()
    }
    values.update(
        graph_enabled=True,
        graph_capture_profile="powers",
        graph_padding=1,
        prefill_cuda_graph_backend="piecewise",
        enable_in_graph_sampling=True,
        enable_early_first_token_response=True,
    )
    values.update(overrides)
    return values


@pytest.mark.parametrize("concurrency", [1, 8, 16, 32, 128])
@pytest.mark.parametrize("backend", ["disabled", "piecewise", "breakable"])
def test_optional_roundtrip_and_dependencies(ns, concurrency, backend):
    first = ns["make_runner"]()
    for option in ns["optional_params"].values():
        option["enabled"] = True
    runner = ns["make_runner"]()
    assert runner.run_dir != first.run_dir
    lab = runner.lab
    assert type(lab) is TRTLLMLabV3
    trial = optuna.trial.FixedTrial(
        params(lab, concurrency, prefill_cuda_graph_backend=backend)
    )
    config = lab.suggest_config(trial, "trtllm", concurrency, "expanded")
    assert config["enable_in_graph_sampling"] is True
    assert config["enable_early_first_token_response"] is True
    assert config["dtype"] == "bfloat16"
    assert config["kv_cache_config"]["dtype"] == "fp8"
    assert config["kv_cache_config"]["tokens_per_block"] == 32
    assert ("torch_compile_config" in config) == (backend == "piecewise")
    assert ("prefill_capture_num_tokens" in config) == (backend != "disabled")
    if concurrency == 1:
        assert "graph_capture_profile" not in trial.params
        assert "graph_padding" not in trial.params
    command = shlex.split(
        native_command(
            lab, config, "/tmp/model", concurrency, executable="/tmp/trtllm-serve"
        )
    )
    assert command[:3] == ["/tmp/trtllm-serve", "serve", "/tmp/model"]
    native = yaml.safe_load(Path(command[command.index("--config") + 1]).read_text())
    assert native["enable_in_graph_sampling"] is True
    assert native["enable_iter_perf_stats"] is True
    assert "--num_serve_frontends" in command
    assert not any(x in native for x in ["quant_config", "calib_config"])
    assert "--enable_in_graph_sampling" not in command
    assert not first.lab.optional_params["enable_in_graph_sampling"]["enabled"]


def test_inactive_controls_and_graph_coverage(ns):
    for option in ns["optional_params"].values():
        option["enabled"] = True
    lab = ns["make_runner"]().lab
    trial = optuna.trial.FixedTrial(
        params(lab, 128, disable_overlap_scheduler=True, graph_enabled=False)
    )
    config = lab.suggest_config(trial, "trtllm", 128, "expanded")
    assert "enable_early_first_token_response" not in trial.params
    assert "enable_in_graph_sampling" not in trial.params
    config = lab.render_engine_config(
        "trtllm",
        128,
        {
            "max_batch_size": 128,
            "max_num_tokens": 8192,
            "kv_cache_free_gpu_memory_fraction": 0.75,
        },
    )
    assert config["cuda_graph_config"]["batch_sizes"] == [1, 2, 4, 8, 16, 32, 64, 128]
    trial = optuna.trial.FixedTrial(params(lab, 128, max_batch_size=128))
    assert (
        64
        in lab.suggest_config(trial, "trtllm", 128, "expanded")["cuda_graph_config"][
            "batch_sizes"
        ]
    )


def test_forbidden_config_and_nested_merge(ns):
    lab = ns["make_runner"]().lab
    base = {"kv_cache_config": {"dtype": "auto"}, "cuda_graph_config": None}
    for key in ["quant_config", "calib_config", "quantization"]:
        with pytest.raises(ValueError):
            lab.serialize_engine_config("trtllm", 8, {**base, key: {}})
    for extra in [
        {"num_serve_frontends": 2},
        {"tp_size": 2},
        {"disable_overlap_scheduler": True, "enable_early_first_token_response": True},
        {"prefill_cuda_graph_backend": "breakable", "torch_compile_config": {}},
    ]:
        with pytest.raises(ValueError):
            lab.serialize_engine_config("trtllm", 8, {**base, **extra})
    native = lab.serialize_engine_config("trtllm", 8, base)
    assert (
        yaml.safe_load(Path(native["config"]).read_text())["kv_cache_config"]["dtype"]
        == "bfloat16"
    )
    assert merge_config(
        {"scheduler_config": {"a": 1, "b": 2}}, {"scheduler_config": {"a": 3}}
    ) == {"scheduler_config": {"a": 3, "b": 2}}


def test_notebook_compiles_without_outputs():
    notebook = json.loads(NOTEBOOK.read_text())
    ids = [cell["id"] for cell in notebook["cells"]]
    assert len(ids) == len(set(ids))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), str(NOTEBOOK), "exec")
            assert cell["outputs"] == [] and cell["execution_count"] is None
    text = NOTEBOOK.read_text()
    assert "quant_algo" not in text
    assert "TRT_FP8_CALIBRATION" not in text
    assert "runtime_help()" in text


def test_native_prometheus_and_conservative_labels():
    body = "\n".join(
        [
            'trtllm_num_requests_running{model_name="m",engine_type="pytorch"} 8',
            'trtllm_num_requests_waiting{model_name="m",engine_type="pytorch"} 0',
            'trtllm_kv_cache_utilization{model_name="m",'
            'engine_type="pytorch",dp_rank="0"} .6',
        ]
    )
    values = aggregate_prometheus(body)
    assert values["running_requests"] == (8, None)
    assert values["queued_requests"] == (0, None)
    assert values["kv_cache_utilization"] == (0.6, None)
    assert values["gpu_utilization"][0] is None
    assert (
        aggregate_prometheus(body + '\ntrtllm_num_requests_running{pid="1"} 9')[
            "running_requests"
        ][0]
        is None
    )
    assert (
        aggregate_prometheus(
            body
            + '\ntrtllm_num_requests_running{model_name="m",engine_type="pytorch"} 9'
        )["running_requests"][0]
        is None
    )
    assert (
        unavailable_telemetry("disabled")["metrics"]["queued_requests"]["peak"] is None
    )


def test_time_weighting():
    metric = summarize(
        [
            (1, {"running_requests": (2, None)}),
            (2, {"running_requests": (8, None)}),
            (4, {"running_requests": (None, "timeout")}),
            (5, {"running_requests": (4, None)}),
        ],
        0,
        6,
        1,
    )["running_requests"]
    assert metric["time_weighted_avg"] == 5.5
    assert metric["peak"] == 8
    assert metric["coverage"] == pytest.approx(4 / 6)


def test_domain_validation():
    for key, value in [
        ("stream_interval", 0),
        ("enable_in_graph_sampling", 1),
        ("kv_tokens_per_block", 7),
    ]:
        options = copy.deepcopy(DEFAULT_OPTIONAL_PARAMS)
        options[key]["values"] = [value]
        with pytest.raises(ValueError):
            validate_options(options, DEFAULT_TELEMETRY)


def test_cpu_study_resume_and_unavailable_metrics(ns):
    for option in ns["optional_params"].values():
        option["enabled"] = True
    runner = ns["make_runner"]()
    study = next(iter(runner.run_objective_group("B", "throughput", [8]).values()))
    assert len(study.trials) == 2
    for trial in study.trials:
        assert trial.state == optuna.trial.TrialState.COMPLETE
        assert trial.user_attrs["configured_client_concurrency"] == 8
        metric = trial.user_attrs["server_telemetry"]["metrics"]["running_requests"]
        assert metric["peak"] is None and metric["time_weighted_avg"] is None
    resumed = ns["make_runner"]().run_objective_group("B", "throughput", [8])
    assert len(next(iter(resumed.values())).trials) == 2


def test_wandb_telemetry_flattening():
    captured = []
    logger = V3Logger(
        lambda study, trial: captured.append(trial.user_attrs), {"concurrency": 8}
    )
    attrs = {
        "execution": {
            "server_telemetry": {"metrics": {"running_requests": {"peak": 6}}}
        }
    }
    logger(None, SimpleNamespace(user_attrs=attrs))
    assert captured[0]["measurement"]["server/running_requests/peak"] == 6
    assert "measurement" not in attrs


@pytest.mark.parametrize("failure", [False, True])
def test_versioned_runtime_persists_and_cleans_up(tmp_path, monkeypatch, failure):
    from llm_optimizer.experiment import runtime_trtllm_v3 as runtime
    from llm_optimizer.experiment import telemetry_trtllm_v3 as telemetry

    process = SimpleNamespace()
    cleanup = []
    monkeypatch.setattr(runtime, "start_server", lambda *a: process)
    monkeypatch.setattr(runtime, "get_server_env", lambda *a: {})
    monkeypatch.setattr(
        runtime, "terminate_process_top_down", lambda p: cleanup.append(p)
    )
    monkeypatch.setattr(runtime, "_close_server_log", lambda p: None)

    def benchmark(args):
        record = args["_telemetry"]
        record.data["server_telemetry"] = {"metrics": {"running_requests": {"peak": 2}}}
        record.data["request_throughput"] = 4.0
        if failure:
            raise optuna.TrialPruned()
        return {
            "completed": 4,
            "failed": 0,
            "total_requests": 4,
            "request_throughput": 4.0,
            "server_telemetry": record.data["server_telemetry"],
        }

    monkeypatch.setattr(telemetry, "run_benchmark", benchmark)
    options = {
        "framework": "trtllm",
        "model": "/tmp/model",
        "server_cmd": "unused",
        "managed_server": True,
        "gpus": 1,
        "gpu_type": "CPU mock",
        "host": "127.0.0.1",
        "port": 30000,
        "ready_url": "unused",
        "mute_server": True,
        "constraints_for_viz": [],
        "output_jsonl_path": tmp_path / "bench.jsonl",
    }
    settings = {"client_args": {}, "server_args": {}, "server_cmd_args": []}
    if failure:
        with pytest.raises(optuna.TrialPruned) as caught:
            runtime.run_single_configuration(settings, **options)
        assert (
            caught.value.execution["server_telemetry"]["metrics"]["running_requests"][
                "peak"
            ]
            == 2
        )
        assert not options["output_jsonl_path"].exists()
    else:
        runtime.run_single_configuration(settings, **options)
        row = json.loads(options["output_jsonl_path"].read_text())
        assert row["results"]["request_throughput"] == 4
        assert (
            row["metadata"]["execution"]["server_telemetry"]["metrics"][
                "running_requests"
            ]["peak"]
            == 2
        )
    assert cleanup == [process]


def test_http_collection_cleanup_and_missing_gpu():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(
                b'trtllm_num_requests_running{model_name="m",engine_type="pytorch"} 3'
            )

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    collector = Collector(
        f"http://127.0.0.1:{server.server_port}/metrics",
        {"interval_s": 0.02, "timeout_s": 0.02},
    )
    try:
        collector.start()
        deadline = time.perf_counter() + 2
        while not collector.samples and time.perf_counter() < deadline:
            time.sleep(0.005)
        result = collector.stop()
        assert not collector.thread.is_alive()
        assert result["metrics"]["running_requests"]["peak"] == 3
        assert result["metrics"]["gpu_utilization"]["peak"] is None
        assert "gpu_index" in result["metrics"]["gpu_utilization"]["reason"]
        count = len(collector.samples)
        collector.sample()
        assert len(collector.samples) == count
        assert collector.stop() == result  # idempotent
    finally:
        collector.stop()
        server.shutdown()
        server.server_close()
        thread.join()


def test_scrape_failures_are_observations_not_trial_failures():
    def unavailable():
        raise TimeoutError("offline")

    collector = Collector("unused", {"interval_s": 0.01}, fetch=unavailable)
    collector.start()
    result = collector.stop()
    assert not collector.thread.is_alive()
    for metric in result["metrics"].values():
        assert metric["peak"] is None
        assert metric["reason"]


@pytest.mark.parametrize("raises", [False, True])
def test_measurement_boundaries_and_request_cleanup(raises):
    collector = Collector("unused", {"interval_s": 0.01}, fetch=lambda: "")
    original = SimpleNamespace(
        data={}, start=lambda name: None, stop=lambda name, complete=True: None
    )
    telemetry = WindowTelemetry(original, collector)

    async def request():
        await asyncio.sleep(0.005)
        if raises:
            raise RuntimeError("pruning/error surrogate")
        return 1

    async def scenario():
        token = _active.set(collector)
        try:
            telemetry.start("warmup")
            try:
                await counted_request(request)()
            except RuntimeError:
                pass
            assert not collector.active and not collector.client_events
            telemetry.start("benchmark")
            try:
                await counted_request(request)()
            except RuntimeError:
                pass
            finally:
                telemetry.stop("benchmark", complete=not raises)
            assert collector.inflight == 0
            assert not collector.thread.is_alive()
            assert not collector.active
            result = original.data["server_telemetry"]
            assert result["client_requests_in_flight"]["peak"] == 1
            count = len(collector.client_events)
            telemetry.start("rest")
            assert len(collector.client_events) == count
        finally:
            collector.stop()
            _active.reset(token)

    asyncio.run(scenario())


def test_nvml_device_explicit_and_shutdown(monkeypatch):
    calls = []
    nvml = SimpleNamespace(
        nvmlInit=lambda: calls.append("init"),
        nvmlDeviceGetHandleByIndex=lambda index: calls.append(index) or object(),
        nvmlDeviceGetUtilizationRates=lambda handle: SimpleNamespace(gpu=73),
        nvmlShutdown=lambda: calls.append("shutdown"),
    )
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    collector = Collector(
        "unused", {"interval_s": 0.01, "gpu_index": 2}, fetch=lambda: ""
    )
    collector.start()
    collector.sample()
    result = collector.stop()
    assert result["metrics"]["gpu_utilization"]["peak"] == 73
    assert calls == ["init", 2, "shutdown"]
