"""Lightweight tests for benchmark accounting and workload identity."""

import math

from llm_optimizer.bench_client import (
    DatasetRow,
    RequestFuncOutput,
    build_workload_manifest,
    calculate_metrics,
    classify_error,
)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


def test_failure_categories_and_complete_success_requirement():
    assert classify_error("HTTP 429", 429) == "request_rejected"
    assert classify_error("CUDA out of memory", 500) == "oom"
    outputs = [
        RequestFuncOutput(
            success=True,
            latency=0.20,
            ttft=0.05,
            itl=[0.05],
            prompt_len=2,
            output_len=3,
            generated_text="a b c",
        ),
        RequestFuncOutput(
            error="HTTP 429",
            error_category="request_rejected",
        ),
    ]
    metrics, _ = calculate_metrics(
        [DatasetRow("a b", 2, 3), DatasetRow("c d", 2, 3)],
        outputs,
        1.0,
        FakeTokenizer(),
        "vllm",
    )
    assert metrics.failed == 1
    assert metrics.failure_rate == 0.5
    assert metrics.failure_categories == {"request_rejected": 1}
    assert not metrics.benchmark_valid


def test_empty_success_cannot_produce_valid_latency_metrics():
    output = RequestFuncOutput(success=False, error="timeout", error_category="timeout")
    metrics, _ = calculate_metrics(
        [DatasetRow("a", 1, 1)], [output], 1.0, FakeTokenizer(), "vllm"
    )
    assert not metrics.benchmark_valid
    assert math.isnan(metrics.mean_e2e_latency_ms)


def test_workload_manifest_is_order_sensitive_and_fingerprinted():
    args = {
        "dataset_name": "random",
        "num_prompts": 2,
        "seed": 7,
        "max_concurrency": 4,
        "random_input_len": 8,
        "random_output_len": 4,
    }
    requests = [DatasetRow("first", 1, 2), DatasetRow("second", 1, 2)]
    first = build_workload_manifest(requests, args, "model", "tokenizer")
    second = build_workload_manifest(list(reversed(requests)), args, "model", "tokenizer")
    assert len(first["sha256"]) == 64
    assert first["sha256"] != second["sha256"]
