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


def test_live_checkpoints_use_completed_requests_and_cancel_pending(monkeypatch):
    import asyncio

    import pytest

    from llm_optimizer import bench_client

    class Pruned(Exception):
        pass

    async def exercise(prune):
        finished = []
        cancelled = []
        snapshots = []

        async def request(request_func_input, pbar=None, **kwargs):
            index = int(request_func_input.prompt)
            try:
                await asyncio.sleep(0.001 * (index + 1))
                finished.append(index)
                return RequestFuncOutput(
                    success=True,
                    latency=0.2,
                    ttft=0.05,
                    output_len=3,
                    generated_text="a b c",
                    itl=[0.05],
                )
            except asyncio.CancelledError:
                cancelled.append(index)
                raise

        def callback(metrics):
            snapshots.append(metrics)
            if prune:
                raise Pruned()

        monkeypatch.setitem(bench_client.ASYNC_REQUEST_FUNCS, "vllm", request)
        monkeypatch.setattr(bench_client.time, "sleep", lambda _: None)
        args = {
            "backend": "vllm",
            "api_url": "http://unused",
            "base_url": "http://unused",
            "model_id": "fake",
            "tokenizer": FakeTokenizer(),
            "input_requests": [DatasetRow(str(i), i + 1, 3) for i in range(8)],
            "request_rate": float("inf"),
            "max_concurrency": 2,
            "disable_tqdm": True,
            "lora_names": None,
            "extra_request_body": {},
            "profile": False,
            "warmup_requests": 0,
            "args_dict": {
                "_checkpoint_requests": [2, 4, 8],
                "_checkpoint_callback": callback,
            },
        }
        if prune:
            with pytest.raises(Pruned):
                await bench_client.benchmark(**args)
            assert len(finished) == 2
            assert cancelled
            assert [m["completed"] for m in snapshots] == [2]
        else:
            result = await bench_client.benchmark(**args)
            assert result["completed"] == 8
            assert [m["completed"] for m in snapshots] == [2, 4, 8]
        for metrics in snapshots:
            count = metrics["completed"]
            assert metrics["total_requests"] == count
            assert metrics["total_input"] == sum(i + 1 for i in finished[:count])
            assert metrics["p95_e2e_latency_ms"] == 200
        assert not [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

    asyncio.run(exercise(prune=True))
    asyncio.run(exercise(prune=False))


def test_pruning_interrupts_slow_request_submission(monkeypatch):
    import asyncio

    import pytest

    from llm_optimizer import bench_client

    class Pruned(Exception):
        pass

    async def exercise():
        submission_cancelled = []

        async def slow_submission(requests, rate):
            yield requests[0]
            try:
                await asyncio.Event().wait()
            finally:
                submission_cancelled.append(True)

        async def request(**kwargs):
            return RequestFuncOutput(
                success=True,
                latency=0.2,
                ttft=0.05,
                output_len=3,
                generated_text="a b c",
                itl=[0.05],
            )

        def callback(metrics):
            assert metrics["completed"] == 1
            raise Pruned()

        monkeypatch.setattr(bench_client, "get_request", slow_submission)
        monkeypatch.setattr(bench_client.time, "sleep", lambda _: None)
        monkeypatch.setitem(bench_client.ASYNC_REQUEST_FUNCS, "vllm", request)
        with pytest.raises(Pruned):
            await asyncio.wait_for(
                bench_client.benchmark(
                    backend="vllm",
                    api_url="http://unused",
                    base_url="http://unused",
                    model_id="fake",
                    tokenizer=FakeTokenizer(),
                    input_requests=[DatasetRow(str(i), 1, 3) for i in range(4)],
                    request_rate=1,
                    max_concurrency=2,
                    disable_tqdm=True,
                    lora_names=None,
                    extra_request_body={},
                    profile=False,
                    warmup_requests=0,
                    args_dict={
                        "_checkpoint_requests": [1, 2, 4],
                        "_checkpoint_callback": callback,
                    },
                ),
                timeout=1,
            )
        assert submission_cancelled == [True]
        assert not [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

    asyncio.run(exercise())


def test_missing_output_is_incomplete_even_if_returned_outputs_succeeded():
    metrics, _ = calculate_metrics(
        [DatasetRow("a", 1, 1), DatasetRow("b", 1, 1)],
        [
            RequestFuncOutput(
                success=True, generated_text="a", output_len=1, ttft=0.01, latency=0.02
            )
        ],
        1.0,
        FakeTokenizer(),
        "vllm",
    )
    assert metrics.total_requests == 2
    assert metrics.failed == 1
    assert metrics.failure_categories == {"missing_response": 1}
    assert not metrics.benchmark_valid
