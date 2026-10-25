"""Failed request evidence survives without accepting incomplete measurements."""

import asyncio

import optuna
import pytest

from llm_optimizer import bench_client
from llm_optimizer.bench_client import DatasetRow, RequestFuncOutput
from llm_optimizer.optimization import extract_objective, summarize_study
from llm_optimizer.trial_telemetry import TrialTelemetry


class Tokenizer:
    def encode(self, text, **kwargs):
        return text.split()


@pytest.mark.parametrize("failed_index", [0, 17, 99])
def test_99_of_100_preserves_request_error_without_prompt_text(
    monkeypatch, tmp_path, failed_index
):
    submitted = []

    async def request(request_func_input, **kwargs):
        submitted.append((request_func_input.prompt, request_func_input.output_len))
        failed = (
            request_func_input.output_len == 128
            and request_func_input.prompt == f"private-prompt-{failed_index}"
        )
        return RequestFuncOutput(
            success=not failed,
            latency=0.2,
            ttft=0 if failed else 0.05,
            output_len=128,
            generated_text="private generated response",
            error="Successful response contained no streamed token." if failed else "",
            error_category="client_error" if failed else "",
        )

    monkeypatch.setitem(bench_client.ASYNC_REQUEST_FUNCS, "vllm", request)
    monkeypatch.setattr(bench_client.time, "sleep", lambda _: None)
    telemetry = TrialTelemetry(tmp_path)
    result = asyncio.run(
        bench_client.benchmark(
            backend="vllm",
            api_url="http://unused",
            base_url="http://unused",
            model_id="fake",
            tokenizer=Tokenizer(),
            input_requests=[
                DatasetRow(f"private-prompt-{i}", 512, 128) for i in range(100)
            ],
            request_rate=float("inf"),
            max_concurrency=24,
            disable_tqdm=True,
            lora_names=None,
            extra_request_body={},
            profile=False,
            warmup_requests=1,
            args_dict={"_telemetry": telemetry},
        )
    )
    assert submitted[0] == ("private-prompt-0", 32)
    assert sorted(prompt for prompt, length in submitted if length == 128) == sorted(
        f"private-prompt-{i}" for i in range(100)
    )
    assert len(submitted) == 101
    assert result["completed"] == 99
    assert result["failed"] == 1
    assert result["benchmark_valid"] is False
    assert result["request_errors"] == [
        {
            "request_index": failed_index,
            "category": "client_error",
            "error": "Successful response contained no streamed token.",
        }
    ]
    assert "private-prompt" not in str(result["request_errors"])
    assert "private generated response" not in str(result["request_errors"])
    assert telemetry.data["request_errors"] == result["request_errors"]
    with pytest.raises(ValueError, match="failed or incomplete"):
        extract_objective(result, "output_throughput")


def test_failed_trial_summary_retains_reason_and_request_accounting():
    study = optuna.create_study(directions=["maximize", "minimize"])
    trial = study.ask()
    trial.set_user_attr(
        "failure_reason", "Benchmark is failed or incomplete; objective is invalid"
    )
    execution = {
        "request_accounting": {"completed": 99, "total_requests": 100, "failed": 1},
        "request_errors": [{"request_index": 17, "error": "empty stream"}],
    }
    trial.set_user_attr("execution", execution)
    study.tell(trial, state=optuna.trial.TrialState.FAIL)
    summary = summarize_study(study, ["output_throughput", "mean_e2e_latency_ms"])
    assert summary["completed_trials"] == 0
    assert summary["pareto_front"] == []
    assert summary["failed_trials"][0]["request_errors"] == execution["request_errors"]
    assert summary["failed_trials"][0]["request_accounting"]["completed"] == 99


@pytest.mark.parametrize("failed_index", [0, 2, 4])
def test_checkpoint_rejection_retains_original_request_index(
    monkeypatch, tmp_path, failed_index
):
    async def request(request_func_input, **kwargs):
        index = int(request_func_input.prompt)
        await asyncio.sleep(0.001 * (5 - index))
        return RequestFuncOutput(
            success=index != failed_index,
            generated_text="token",
            output_len=1,
            ttft=0.01,
            latency=0.02,
            error_category="interrupted_stream" if index == failed_index else "",
            error="Stream ended without a finish event or [DONE]"
            if index == failed_index
            else "",
        )

    monkeypatch.setitem(bench_client.ASYNC_REQUEST_FUNCS, "vllm", request)
    monkeypatch.setattr(bench_client.time, "sleep", lambda _: None)
    telemetry = TrialTelemetry(tmp_path)
    with pytest.raises(ValueError, match="failed or incomplete"):
        asyncio.run(
            bench_client.benchmark(
                backend="vllm",
                api_url="http://unused",
                base_url="http://unused",
                model_id="fake",
                tokenizer=Tokenizer(),
                input_requests=[DatasetRow(str(i), 1, 1) for i in range(5)],
                request_rate=float("inf"),
                max_concurrency=5,
                disable_tqdm=True,
                lora_names=None,
                extra_request_body={},
                profile=False,
                warmup_requests=0,
                args_dict={
                    "_telemetry": telemetry,
                    "_checkpoint_requests": [],
                    "_checkpoint_callback": lambda values: extract_objective(
                        values, "output_throughput"
                    ),
                },
            )
        )
    import json

    persisted = json.loads(telemetry.path.read_text())
    assert persisted["request_errors"][0]["request_index"] == failed_index
    assert persisted["request_errors"][0]["category"] == "interrupted_stream"
    assert persisted["request_accounting"]["failed"] == 1


def test_diagnostics_are_bounded_and_do_not_echo_other_backend_errors():
    outputs = [
        RequestFuncOutput(
            error="Bearer secret and private prompt", error_category="server_error"
        )
        for _ in range(100)
    ]
    errors = bench_client.request_errors(outputs)
    assert len(errors) == 20
    assert "secret" not in str(errors)
    assert "private prompt" not in str(errors)
