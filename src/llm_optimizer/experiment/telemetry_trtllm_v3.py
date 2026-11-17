"""Measured-window TensorRT-LLM 1.4.0rc3 telemetry, isolated from legacy clients.

Official source: NVIDIA/TensorRT-LLM v1.4.0rc3, metrics/collector.py.
Counts are rank-0-owned; KV ratios are maximum across dp_rank.
The v3 runner uses one HTTP frontend and one GPU to avoid process duplicates.
Unknown labels and conflicting duplicate series are unavailable, never zero.
"""

from __future__ import annotations

import contextvars
import functools
import json
import math
import re
import threading
import time
from types import FunctionType

import requests

METRICS = {
    "running_requests": ("trtllm_num_requests_running", "requests", "sum"),
    "queued_requests": ("trtllm_num_requests_waiting", "requests", "sum"),
    "kv_cache_utilization": ("trtllm_kv_cache_utilization", "fraction", "max"),
    "gpu_utilization": (None, "percent", "max"),
}
DEFAULT_TELEMETRY = {
    "enabled": True,
    "interval_s": 0.5,
    "timeout_s": 0.25,
    "gpu_index": None,
}
_SAMPLE = re.compile(r"^([a-zA-Z_:][\w:]*)(?:\{(.*)\})?\s+([^\s]+)(?:\s+\S+)?$")
_LABEL = re.compile(r'([a-zA-Z_]\w*)="((?:\\.|[^"\\])*)"(?:,|$)')
_active = contextvars.ContextVar("trtllm_v3_collector", default=None)


def unavailable_telemetry(reason):
    """Explicit missing-data schema for disabled, synthetic or failed collectors."""
    return {
        "enabled": False,
        "reason": reason,
        "sampling_interval_s": None,
        "scrape_timeout_s": None,
        "sample_count": 0,
        "measured_window_s": None,
        "metrics": {
            name: {
                "available": False,
                "time_weighted_avg": None,
                "peak": None,
                "unit": unit,
                "sample_count": 0,
                "coverage": 0.0,
                "aggregation": operation,
                "reason": reason,
            }
            for name, (_, unit, operation) in METRICS.items()
        },
        "client_requests_in_flight": {
            "time_weighted_avg": None,
            "peak": None,
            "unit": "requests",
            "reason": reason,
        },
    }


def aggregate_prometheus(body):
    """Return metric -> (value, reason), conservatively aggregating known labels."""
    rows = {name: [] for name in METRICS}
    invalid = {}
    for line in body.splitlines():
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE.match(line)
        if not match:
            continue
        native, label_text, value_text = match.groups()
        for name, (expected, unit, _operation) in METRICS.items():
            if native != expected or expected is None:
                continue
            try:
                labels = {}
                remainder = label_text or ""
                while remainder:
                    label = _LABEL.match(remainder)
                    if label is None:
                        raise ValueError("malformed labels")
                    labels[label[1]] = json.loads('"' + label[2] + '"')
                    remainder = remainder[label.end() :]
                allowed = {"model_name", "engine_type", "dp_rank"}
                if set(labels) - allowed:
                    raise ValueError("unknown worker/replica label layout")
                if name != "kv_cache_utilization" and "dp_rank" in labels:
                    raise ValueError("request counts must be rank-0-owned totals")
                value = float(value_text)
                if not math.isfinite(value) or value < 0:
                    raise ValueError("nonfinite or negative gauge")
                if unit == "fraction" and value > 1:
                    raise ValueError("gauge outside fraction range")
                rows[name].append((labels, value))
            except (ValueError, TypeError) as exc:
                invalid[name] = str(exc)
    result = {}
    for name, values in rows.items():
        if name in invalid:
            result[name] = (None, invalid[name])
            continue
        if not values:
            result[name] = (None, "metric not exposed")
            continue
        identities = {(v[0].get("model_name"), v[0].get("engine_type")) for v in values}
        if len(identities) != 1:
            result[name] = (
                None,
                "multiple models or engine roles; aggregation ambiguous",
            )
            continue
        replicas = {}
        for labels, value in values:
            replica = labels.get("dp_rank", "0")
            if replica in replicas and replicas[replica] != value:
                invalid[name] = "conflicting duplicate replica series"
            replicas[replica] = value
        if name in invalid:
            result[name] = (None, invalid[name])
        else:
            operation = METRICS[name][2]
            result[name] = (
                sum(replicas.values())
                if operation == "sum"
                else max(replicas.values()),
                None,
            )
    return result


def summarize(samples, start, end, interval):
    """Integrate left holds; failures break coverage, holds expire at 2 intervals."""
    duration = max(0.0, end - start)
    output = {}
    for name, (_, unit, operation) in METRICS.items():
        area = covered = 0.0
        valid = []
        reasons = set()
        for index, (stamp, values) in enumerate(samples):
            value, reason = values.get(name, (None, "metric not exposed"))
            if value is None:
                reasons.add(reason or "unavailable")
                continue
            valid.append(value)
            next_stamp = samples[index + 1][0] if index + 1 < len(samples) else end
            width = max(
                0.0, min(end, next_stamp, stamp + 2 * interval) - max(start, stamp)
            )
            area += width * value
            covered += width
        output[name] = {
            "available": bool(valid),
            "time_weighted_avg": area / covered if covered else None,
            "peak": max(valid) if valid else None,
            "unit": unit,
            "sample_count": len(valid),
            "coverage": min(1.0, covered / duration) if duration else 0.0,
            "aggregation": "rank-0-owned request total"
            if operation == "sum"
            else "maximum across replicas/devices",
            "reason": "; ".join(sorted(reasons))
            if reasons
            else (None if valid else "no measured-window samples"),
        }
    return output


class Collector:
    def __init__(self, url, options, *, fetch=None, clock=time.perf_counter):
        self.options = {**DEFAULT_TELEMETRY, **options}
        self.url = url
        self.clock = clock
        self.fetch = fetch or self._fetch
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = None
        self.started = self.ended = None
        self.active = False
        self.inflight = 0
        self.client_events = []
        self.gpu_reason = "gpu_index not configured; no device inferred"
        self.nvml = self.handle = None
        self.owns_nvml = False
        self.errors = []

    def _fetch(self):
        response = requests.get(self.url, timeout=self.options["timeout_s"])
        response.raise_for_status()
        return response.text

    def start(self):
        self.started = self.clock()
        self.active = True
        self.client_events = [(self.started, 0)]
        if self.options["gpu_index"] is not None:
            try:
                import pynvml

                pynvml.nvmlInit()
                self.owns_nvml = True
                self.nvml = pynvml
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(
                    self.options["gpu_index"]
                )
            except Exception as exc:
                self.gpu_reason = f"NVML unavailable: {type(exc).__name__}"
        self.thread = threading.Thread(
            target=self._loop, daemon=True, name="trtllm-v3-metrics"
        )
        self.thread.start()

    def _loop(self):
        while not self.stop_event.is_set():
            self.sample()
            self.stop_event.wait(self.options["interval_s"])

    def sample(self):
        try:
            values = aggregate_prometheus(self.fetch())
        except Exception as exc:
            reason = f"metrics scrape failed: {type(exc).__name__}"
            values = dict.fromkeys(METRICS, (None, reason))
        if self.handle is not None:
            try:
                values["gpu_utilization"] = (
                    float(self.nvml.nvmlDeviceGetUtilizationRates(self.handle).gpu),
                    None,
                )
            except Exception as exc:
                values["gpu_utilization"] = (
                    None,
                    f"NVML read failed: {type(exc).__name__}",
                )
        else:
            values["gpu_utilization"] = (None, self.gpu_reason)
        # Discard scrapes completing after the window.
        stamp = self.clock()
        if self.active:
            self.samples.append((stamp, values))

    def change_inflight(self, amount):
        if self.active:
            self.inflight += amount
            self.client_events.append((self.clock(), self.inflight))

    def stop(self):
        if self.ended is None:
            self.ended = self.clock()
            self.active = False
            self.stop_event.set()
            if self.thread is not None and self.thread.ident is not None:
                self.thread.join()  # HTTP timeout bounds normal collector shutdown.
            if self.owns_nvml:
                try:
                    self.nvml.nvmlShutdown()
                except Exception as exc:
                    self.errors.append(f"NVML cleanup: {type(exc).__name__}")
                self.owns_nvml = False
        return self.result()

    def result(self):
        start = self.started if self.started is not None else self.clock()
        end = self.ended if self.ended is not None else self.clock()
        interval = self.options["interval_s"]
        duration = max(0.0, end - start)
        area = sum(
            value
            * max(
                0,
                min(
                    end,
                    self.client_events[i + 1][0]
                    if i + 1 < len(self.client_events)
                    else end,
                )
                - stamp,
            )
            for i, (stamp, value) in enumerate(self.client_events)
        )
        return {
            "enabled": True,
            "sampling_interval_s": interval,
            "scrape_timeout_s": self.options["timeout_s"],
            "sample_count": len(self.samples),
            "measured_window_s": duration,
            "metrics": summarize(self.samples, start, end, interval),
            "client_requests_in_flight": {
                "time_weighted_avg": area / duration if duration else None,
                "peak": max((v for _, v in self.client_events), default=None),
                "unit": "requests",
                "reason": None if self.started is not None else "not started",
                "meaning": (
                    "request coroutines after client semaphore admission; "
                    "includes transport time"
                ),
            },
            "gpu_index": self.options["gpu_index"],
            "errors": self.errors,
            "semantics": "Sampled gauges; left hold up to 2 intervals, gaps excluded. "
            "Not GPU kernel occupancy. Pool ratios include retained cached state. "
            "Server publication cadence can be coarser than scraping.",
        }


class WindowTelemetry:
    """Delegate attempt telemetry and collect at measured phase boundaries."""

    def __init__(self, original, collector):
        self.original = original
        self.collector = collector

    def __getattr__(self, name):
        return getattr(self.original, name)

    def start(self, name):
        self.original.start(name)
        if name == "benchmark":
            try:
                self.collector.start()
            except Exception as exc:
                self.collector.errors.append(f"collector start: {type(exc).__name__}")

    def stop(self, name, complete=True):
        if name == "benchmark":
            self.finish_collection()
        self.original.stop(name, complete=complete)

    def finish_collection(self):
        try:
            self.original.data["server_telemetry"] = self.collector.stop()
        except Exception as exc:
            self.original.data["server_telemetry"] = unavailable_telemetry(
                f"collector failure: {type(exc).__name__}"
            )


def counted_request(function):
    @functools.wraps(function)
    async def counted(*args, **kwargs):
        collector = _active.get()
        admitted = collector is not None and collector.active
        if admitted:
            collector.change_inflight(1)
        try:
            return await function(*args, **kwargs)
        finally:
            if admitted:
                collector.change_inflight(-1)

    return counted


def _clone(function, namespace):
    """Give reused client algorithms private globals; never patch legacy modules."""
    result = FunctionType(
        function.__code__,
        namespace,
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def run_benchmark(args_dict):
    from llm_optimizer import bench_client

    original = args_dict.get("_telemetry")
    options = (
        original.data.get("environment", {}).get("trtllm_v3_telemetry", {})
        if original
        else {}
    )
    configured = args_dict.get("max_concurrency")
    if not options.get("enabled", False):
        result = bench_client.run_benchmark(args_dict)
        result["configured_client_concurrency"] = configured
        result["server_telemetry"] = unavailable_telemetry("collection disabled")
        if original:
            original.data["server_telemetry"] = result["server_telemetry"]
            original.data["request_throughput"] = result.get("request_throughput")
            original.data["configured_client_concurrency"] = configured
        return result
    collector = Collector(
        f"http://{args_dict.get('host', '127.0.0.1')}:{args_dict['port']}/metrics",
        options,
    )
    window = WindowTelemetry(original, collector)
    args_dict = {**args_dict, "_telemetry": window}
    namespace = dict(vars(bench_client))
    namespace["ASYNC_REQUEST_FUNCS"] = {
        name: counted_request(function)
        for name, function in bench_client.ASYNC_REQUEST_FUNCS.items()
    }
    namespace["benchmark"] = _clone(bench_client.benchmark, namespace)
    run = _clone(bench_client.run_benchmark, namespace)
    token = _active.set(collector)
    try:
        result = run(args_dict)
        result["configured_client_concurrency"] = configured
        result["server_telemetry"] = original.data.get("server_telemetry")
        original.data["request_throughput"] = result.get("request_throughput")
        return result
    finally:
        window.finish_collection()
        original.data["configured_client_concurrency"] = configured
        _active.reset(token)
