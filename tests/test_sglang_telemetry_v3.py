"""CPU tests for v3 metrics semantics and measured-window collection."""

import asyncio
import json
import sys
import threading
import time
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from llm_optimizer.experiment.telemetry_sglang_v3 import (
    Collector,
    WindowTelemetry,
    _active,
    aggregate_prometheus,
    counted_request,
    run_benchmark,
    summarize,
)


def gauge(name, value, **labels):
    label_text = ",".join(
        f"{key}={json.dumps(str(value))}" for key, value in labels.items()
    )
    return f"sglang:{name}{{{label_text}}} {value}"


def test_replica_rank_priority_aggregation():
    body = "\n".join(
        [
            gauge("num_running_reqs", 3, dp_rank=0, tp_rank=0, priority=""),
            gauge("num_running_reqs", 30, dp_rank=0, tp_rank=1),
            gauge("num_running_reqs", 2, dp_rank=0, tp_rank=0, priority=2),
            gauge("num_running_reqs", 5, dp_rank=1, tp_rank=0),
            gauge("num_queue_reqs", 0, dp_rank=0),
            gauge("full_token_usage", 0.2, dp_rank=0),
            gauge("full_token_usage", 0.6, dp_rank=1),
            gauge("token_usage", 0.9, dp_rank=0),
        ]
    )
    values = aggregate_prometheus(body)
    assert values["running_requests"] == (8, None)
    assert values["queued_requests"] == (0, None)  # observed zero is legitimate
    assert values["kv_cache_utilization"] == (0.6, None)
    assert values["cache_pool_bottleneck"] == (0.9, None)
    assert values["recurrent_state_utilization"][0] is None


@pytest.mark.parametrize(
    "body,reason",
    [
        (gauge("num_running_reqs", 2, worker="a"), "label layout"),
        (
            gauge("num_running_reqs", 2) + "\n" + gauge("num_running_reqs", 3),
            "duplicate",
        ),
        (
            gauge("num_running_reqs", 2, model_name="a")
            + "\n"
            + gauge("num_running_reqs", 3, model_name="b"),
            "multiple models",
        ),
        (gauge("num_running_reqs", "NaN"), "nonfinite"),
        (gauge("num_running_reqs", -1), "negative"),
        (gauge("num_running_reqs", 2, tp_rank=1), "not exposed"),
    ],
)
def test_ambiguous_or_unsupported_metrics_are_null(body, reason):
    value, explanation = aggregate_prometheus(body)["running_requests"]
    assert value is None and reason in explanation


def test_time_weighting_gaps_and_units():
    samples = [
        (1, {"running_requests": (2, None)}),
        (2, {"running_requests": (8, None)}),
        (4, {"running_requests": (None, "scrape failed")}),
        (5, {"running_requests": (4, None)}),
    ]
    metric = summarize(samples, 0, 6, 1)["running_requests"]
    assert metric["time_weighted_avg"] == 5.5  # (2*1+8*2+4*1)/4
    assert metric["coverage"] == pytest.approx(4 / 6)
    assert metric["peak"] == 8
    assert metric["sample_count"] == 3
    assert metric["unit"] == "requests"
    assert metric["reason"] == "scrape failed"
    absent = summarize([], 0, 1, 0.5)["gpu_utilization"]
    assert absent["time_weighted_avg"] is None and absent["peak"] is None
    assert absent["reason"] and absent["coverage"] == 0


def test_old_sample_expires_instead_of_filling_missing_window():
    metric = summarize([(0, {"running_requests": (7, None)})], 0, 10, 1)[
        "running_requests"
    ]
    assert metric["coverage"] == 0.2
    assert metric["time_weighted_avg"] == 7


def test_http_collection_cleanup_and_missing_gpu():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(gauge("num_running_reqs", 3).encode())

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


def test_client_wrapper_keeps_legacy_globals_and_partial_telemetry(monkeypatch):
    # A minimal benchmark algorithm using the same phase hooks and request map.
    module = types.ModuleType("llm_optimizer.bench_client")
    exec(
        """
import asyncio
async def request(**kwargs):
    await asyncio.sleep(.001)
ASYNC_REQUEST_FUNCS = {'vllm': request}
async def benchmark(args):
    telemetry = args['_telemetry']
    telemetry.start('warmup')
    await ASYNC_REQUEST_FUNCS['vllm']()
    telemetry.stop('warmup')
    telemetry.start('benchmark')
    try:
        await ASYNC_REQUEST_FUNCS['vllm']()
        if args.get('fail'):
            raise RuntimeError('pruned')
    finally:
        telemetry.stop('benchmark', complete=not args.get('fail'))
    return {'request_throughput': 12.5}
def run_benchmark(args):
    return asyncio.run(benchmark(args))
""",
        module.__dict__,
    )
    monkeypatch.setitem(sys.modules, "llm_optimizer.bench_client", module)
    import llm_optimizer

    monkeypatch.setattr(llm_optimizer, "bench_client", module, raising=False)
    legacy_request = module.ASYNC_REQUEST_FUNCS["vllm"]
    original = SimpleNamespace(
        data={
            "environment": {
                "sglang_v3_telemetry": {
                    "enabled": True,
                    "interval_s": 0.01,
                    "timeout_s": 0.001,
                }
            }
        },
        start=lambda name: None,
        stop=lambda name, complete=True: None,
    )
    for fail in (False, True):
        args = {"_telemetry": original, "port": 1, "max_concurrency": 8, "fail": fail}
        if fail:
            with pytest.raises(RuntimeError, match="pruned"):
                run_benchmark(args)
        else:
            result = run_benchmark(args)
            assert result["request_throughput"] == 12.5
            assert result["configured_client_concurrency"] == 8
        assert (
            original.data["server_telemetry"]["client_requests_in_flight"]["peak"] == 1
        )
        assert module.ASYNC_REQUEST_FUNCS["vllm"] is legacy_request
        assert _active.get() is None
        assert not any(t.name == "sglang-v3-metrics" for t in threading.enumerate())


def test_partial_collector_start_failure_still_closes_nvml(monkeypatch):
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "pynvml",
        SimpleNamespace(
            nvmlInit=lambda: None,
            nvmlDeviceGetHandleByIndex=lambda index: object(),
            nvmlShutdown=lambda: calls.append("shutdown"),
        ),
    )

    def failed_start(self):
        raise RuntimeError("thread start failed")

    monkeypatch.setattr(threading.Thread, "start", failed_start)
    collector = Collector("unused", {"gpu_index": 0})
    telemetry = WindowTelemetry(
        SimpleNamespace(
            data={}, start=lambda name: None, stop=lambda name, complete=True: None
        ),
        collector,
    )
    telemetry.start("benchmark")
    telemetry.stop("benchmark")
    assert calls == ["shutdown"]
    assert "collector start" in collector.errors[0]
    assert (
        telemetry.data["server_telemetry"]["metrics"]["running_requests"]["peak"]
        is None
    )


def test_cancellation_releases_client_inflight_and_collector():
    async def scenario():
        collector = Collector("unused", {"interval_s": 0.01}, fetch=lambda: "")
        token = _active.set(collector)
        gate = asyncio.Event()

        async def pending():
            gate.set()
            await asyncio.Event().wait()

        collector.start()
        try:
            task = asyncio.create_task(counted_request(pending)())
            await gate.wait()
            assert collector.inflight == 1
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert collector.inflight == 0
        finally:
            collector.stop()
            _active.reset(token)
        assert not collector.thread.is_alive()

    asyncio.run(scenario())
