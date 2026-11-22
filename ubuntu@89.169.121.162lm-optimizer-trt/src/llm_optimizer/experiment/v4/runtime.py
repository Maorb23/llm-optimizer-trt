"""Chat-only v4 wrappers composed with existing launch and telemetry APIs."""

from __future__ import annotations

import importlib
import json
import math
import shlex
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Generation:
    output_tokens: int = 256
    reasoning_tokens: int | None = None
    thinking: str = "default"
    eos: str = "stop"
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 0
    stop: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "stop", tuple(self.stop))
        if self.output_tokens < 1 or self.thinking not in {"default", "on", "off"}:
            raise ValueError("Invalid output budget/thinking mode")
        if self.eos not in {"stop", "ignore"}:
            raise ValueError("EOS must be stop or ignore")
        if self.reasoning_tokens is not None and self.reasoning_tokens < 1:
            raise ValueError("Reasoning budget must be positive")
        if (
            not math.isfinite(self.temperature)
            or self.temperature < 0
            or not 0 < self.top_p <= 1
        ):
            raise ValueError("Invalid sampling settings")
        if any(not isinstance(s, str) or not s for s in self.stop):
            raise ValueError("Stop sequences must be nonempty strings")


@dataclass(frozen=True)
class Completion:
    example_id: str
    text: str = ""
    reasoning: str = ""
    finish_reason: str | None = None
    output_tokens: int | None = None
    latency_s: float = 0.0
    ttft_s: float | None = None
    generation_error: str | None = None

    @property
    def truncated(self):
        return self.finish_reason in {"length", "max_tokens"}


def http_generate(url, payload, timeout):
    """Stream final text separately from reasoning; usage is never estimated."""
    import requests

    started = time.perf_counter()
    text, reasoning, finish, tokens, first = [], [], None, None, None
    with requests.post(url, json=payload, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line or not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(str(event["error"]))
            usage = event.get("usage") or {}
            if usage.get("completion_tokens") is not None:
                tokens = int(usage["completion_tokens"])
            for choice in event.get("choices", []):
                delta = choice.get("delta") or {}
                content = delta.get("content") or ""
                thought = (
                    delta.get("reasoning_content", delta.get("reasoning", "")) or ""
                )
                if (content or thought) and first is None:
                    first = time.perf_counter() - started
                text.append(content)
                reasoning.append(thought)
                if choice.get("finish_reason") is not None:
                    finish = choice["finish_reason"]
    if finish not in {"stop", "length", "max_tokens"}:
        raise RuntimeError(f"Incomplete/unsupported stream finish reason: {finish!r}")
    if tokens is not None and tokens < 0:
        raise RuntimeError("Invalid completion token count")
    return {
        "text": "".join(text),
        "reasoning": "".join(reasoning),
        "finish_reason": finish,
        "output_tokens": tokens,
        "ttft_s": first,
    }


class RuntimeV4:
    framework = ""

    def __init__(
        self,
        model,
        *,
        runtime_version,
        base_url="http://127.0.0.1:8000",
        concurrency=1,
        enable_gpu=False,
        thinking_template_verified=False,
        timeout_s=600,
        telemetry_options=None,
        transport=None,
    ):
        if concurrency < 1 or timeout_s <= 0 or not runtime_version:
            raise ValueError(
                "Positive concurrency/timeout and explicit runtime version required"
            )
        self.model = model
        self.runtime_version = runtime_version
        self.base_url = base_url.rstrip("/")
        self.concurrency = concurrency
        self.enable_gpu = enable_gpu
        self.thinking_template_verified = thinking_template_verified
        self.timeout_s = timeout_s
        self.telemetry_options = {"enabled": True, **(telemetry_options or {})}
        self.transport = transport or http_generate
        self.synthetic = transport is not None

    @property
    def identity(self):
        return {
            "framework": self.framework,
            "runtime_version": self.runtime_version,
            "model": self.model,
            "concurrency": self.concurrency,
            "thinking_template_verified": self.thinking_template_verified,
            "timeout_s": self.timeout_s,
            "telemetry": self.telemetry_options,
            "synthetic": self.synthetic,
        }

    def validate_generation(self, generation):
        # None of these versioned wrappers promises a separately enforced reasoning
        # token cap. max_tokens is the TOTAL generation cap, including reasoning.
        if generation.reasoning_tokens is not None:
            raise ValueError(
                f"{self.framework}: separate reasoning budget is unsupported"
            )
        if generation.thinking != "default":
            if self.framework == "trtllm" or not self.thinking_template_verified:
                raise ValueError(
                    "Explicit thinking requires a verified vLLM/SGLang "
                    "enable_thinking template"
                )
        if generation.eos == "ignore" and self.framework == "trtllm":
            raise ValueError("TensorRT-LLM wrapper does not support ignore_eos")

    def payload(self, example, generation, *, served_model=None):
        self.validate_generation(generation)
        # Whitelist just messages, even if caller supplies extra manifest fields.
        messages = example["payload"]["messages"]
        if any(set(m) != {"role", "content"} for m in messages):
            raise ValueError("Only text chat messages are supported")
        payload = {
            "model": served_model or self.model,
            "messages": messages,
            "max_tokens": generation.output_tokens,
            "temperature": generation.temperature,
            "top_p": generation.top_p,
            "seed": generation.seed,
            "n": 1,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if generation.stop:
            payload["stop"] = list(generation.stop)
        if self.framework in {"vllm", "sglang"}:
            payload["ignore_eos"] = generation.eos == "ignore"
        if generation.thinking != "default":
            payload["chat_template_kwargs"] = {
                "enable_thinking": generation.thinking == "on"
            }
        return payload

    def preflight(self, manifest):
        manifest.verify()
        doc = manifest.document
        if doc["model"] != self.model:
            raise ValueError("Runtime model differs from frozen model")
        generation = Generation(**doc["generation"])
        return [self.payload(e, generation) for e in doc["examples"]]

    def command(self, model_path, tokenizer_path, native_config):
        """Reuse native argument registry/serialization, without mutating it."""
        from urllib.parse import urlparse

        from llm_optimizer.args import Arg, ArgScope
        from llm_optimizer.cli import construct_benchmark_settings
        from llm_optimizer.predefined import get_framework_spec

        address = urlparse(self.base_url)
        if (
            address.scheme != "http"
            or not address.hostname
            or not address.port
            or address.path
        ):
            raise ValueError("Managed server requires http://host:port")
        registry = importlib.import_module(
            f"llm_optimizer.experiment.{self.framework}_policy_v3"
        ).V3_SERVER_CONFIGS
        reserved = {
            "model",
            "model_path",
            "host",
            "port",
            "revision",
            "hf_revision",
            "tokenizer",
            "tokenizer_path",
            "tokenizer_revision",
            "served_model_name",
            "chat_template",
            "custom_tokenizer",
        }
        if reserved & set(native_config):
            raise ValueError(
                "Model/tokenizer/host identity cannot be overridden by native config"
            )
        config = dict(native_config)
        config["tokenizer_path" if self.framework == "sglang" else "tokenizer"] = (
            tokenizer_path
        )
        if self.framework == "sglang" and self.telemetry_options["enabled"]:
            config["enable_metrics"] = True
        unknown = set(config) - set(registry)
        if unknown:
            raise ValueError(f"Unsupported native settings: {sorted(unknown)}")
        args = [
            Arg[registry[k].arg_type](scope=ArgScope.SERVER, name=k, value=v)
            for k, v in config.items()
        ]
        settings = construct_benchmark_settings(args, self.framework)
        spec = get_framework_spec(self.framework)
        base = spec.server_command.format(
            model=shlex.quote(model_path),
            host=shlex.quote(address.hostname),
            port=address.port,
        )
        return base + " " + shlex.join(settings["server_cmd_args"]), settings

    @contextmanager
    def server(self, manifest, config, telemetry):
        if not self.enable_gpu:
            raise RuntimeError("GPU execution disabled; explicitly set enable_gpu=True")
        from huggingface_hub import snapshot_download

        from llm_optimizer.server_utils import (
            _close_server_log,
            get_server_env,
            start_server,
            terminate_process_top_down,
        )

        doc = manifest.document
        with telemetry.phase("client_preparation"):
            from importlib.metadata import version

            package = {"vllm": "vllm", "sglang": "sglang", "trtllm": "tensorrt_llm"}[
                self.framework
            ]
            installed = version(package)
            if installed != self.runtime_version:
                raise ValueError(
                    f"Expected {package}=={self.runtime_version}, found {installed}"
                )
            for key in ("model_revision", "tokenizer_revision"):
                if len(doc[key]) != 40:
                    raise ValueError(
                        "Managed HF serving requires model/tokenizer commit SHAs"
                    )
            model_path = snapshot_download(doc["model"], revision=doc["model_revision"])
            tokenizer_path = snapshot_download(
                doc["tokenizer_id"], revision=doc["tokenizer_revision"]
            )
            command, settings = self.command(model_path, tokenizer_path, config)
        telemetry.data.update(command=command, config=settings)
        process = None
        try:
            with telemetry.activate():
                process = start_server(
                    command,
                    get_server_env(self.framework, 1, settings["server_args"]),
                    self.base_url + "/health",
                    False,
                )
            yield model_path
        finally:
            if process is not None:
                with telemetry.activate(), telemetry.phase("cleanup"):
                    try:
                        terminate_process_top_down(process)
                    except BaseException as exc:
                        telemetry.data["cleanup_error"] = str(exc)
                        raise
                    finally:
                        _close_server_log(process)

    def generate(
        self, manifest, *, native_config=None, output_dir="v4-results", managed=True
    ):
        from llm_optimizer.trial_telemetry import TrialTelemetry

        payloads = self.preflight(manifest)
        if not self.synthetic and not self.enable_gpu:
            raise RuntimeError("GPU execution disabled")
        if self.synthetic and managed:
            raise ValueError("Injected transports must use managed=False")
        config = dict(native_config or {})
        if not managed and config:
            raise ValueError(
                "Native configuration cannot be applied to an unmanaged server"
            )
        telemetry = TrialTelemetry(
            Path(output_dir) / "attempts",
            **self.identity,
            manifest=manifest.signature,
            native_config=config,
        )
        error = None
        try:
            if managed:
                with self.server(manifest, config, telemetry) as served_model:
                    for payload in payloads:
                        payload["model"] = served_model
                    return self._measure(manifest, payloads, telemetry)
            return self._measure(manifest, payloads, telemetry)
        except BaseException as exc:
            error = exc
            raise
        finally:
            telemetry.finish(error)

    def _measure(self, manifest, payloads, telemetry):
        module = importlib.import_module(
            f"llm_optimizer.experiment.telemetry_{self.framework}_v3"
        )
        collect = self.telemetry_options["enabled"] and not self.synthetic
        collector = (
            module.Collector(self.base_url + "/metrics", self.telemetry_options)
            if collect
            else None
        )

        def execute(pair):
            example, payload = pair
            started = time.perf_counter()
            if collector:
                collector.change_inflight(1)
            try:
                response = self.transport(
                    self.base_url + "/v1/chat/completions", payload, self.timeout_s
                )
                if response.get("finish_reason") not in {
                    "stop",
                    "length",
                    "max_tokens",
                }:
                    raise RuntimeError(
                        "Generation did not return a supported finish reason"
                    )
                return Completion(
                    example["example_id"],
                    latency_s=time.perf_counter() - started,
                    **response,
                )
            except Exception as exc:
                return Completion(
                    example["example_id"],
                    latency_s=time.perf_counter() - started,
                    generation_error=f"{type(exc).__name__}: {exc}",
                )
            finally:
                if collector:
                    collector.change_inflight(-1)

        try:
            if collector:
                collector.start()
            with telemetry.phase("benchmark"):
                started = time.perf_counter()
                with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                    outputs = list(
                        pool.map(execute, zip(manifest.document["examples"], payloads))
                    )
                elapsed = time.perf_counter() - started
        finally:
            telemetry.data["server_telemetry"] = (
                collector.stop()
                if collector
                else module.unavailable_telemetry(
                    "synthetic transport" if self.synthetic else "disabled"
                )
            )
        successful = [c for c in outputs if c.generation_error is None]
        counts_known = all(c.output_tokens is not None for c in successful)
        latencies = sorted(c.latency_s for c in successful)
        ttfts = sorted(c.ttft_s for c in successful if c.ttft_s is not None)
        metrics = {
            "inference_seconds": elapsed,
            "requests_per_second": len(successful) / elapsed,
            "output_tokens_per_second": sum(c.output_tokens for c in successful)
            / elapsed
            if counts_known and successful
            else None,
            "p95_latency_s": latencies[math.ceil(0.95 * len(latencies)) - 1]
            if latencies
            else None,
            "p95_ttft_s": ttfts[math.ceil(0.95 * len(ttfts)) - 1] if ttfts else None,
            "completed": len(successful),
            "generation_failures": len(outputs) - len(successful),
            "truncated": sum(c.truncated for c in outputs),
        }
        telemetry.data["request_accounting"] = metrics
        return {"completions": outputs, "metrics": metrics, "execution": telemetry.data}


class VLLMRuntimeV4(RuntimeV4):
    framework = "vllm"


class TRTLLMRuntimeV4(RuntimeV4):
    framework = "trtllm"


class SGLangRuntimeV4(RuntimeV4):
    framework = "sglang"
