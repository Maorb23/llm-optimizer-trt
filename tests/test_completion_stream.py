"""CPU regressions for OpenAI completion stream handling."""

import asyncio
import json

import aiohttp
import pytest

from llm_optimizer import bench_client as client


class Response:
    status = 200
    reason = ""

    def __init__(self, events, status=200):
        self.events = events
        self.status = status
        self.content = self.lines()

    async def lines(self):
        for event in self.events:
            if isinstance(event, Exception):
                raise event
            yield (
                event
                if isinstance(event, bytes)
                else (
                    "data: "
                    + (event if isinstance(event, str) else json.dumps(event))
                    + "\n"
                ).encode()
            )

    async def text(self):
        return ""

    async def json(self):
        return self.events[0]

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class Session(Response):
    def post(self, **kwargs):
        return Response(self.events, self.status)


def run(monkeypatch, events, status=200, **kwargs):
    monkeypatch.setattr(
        client, "_create_bench_client_session", lambda: Session(events, status)
    )
    return asyncio.run(
        client.async_request_openai_completions(
            client.RequestFuncInput(
                "secret prompt",
                "http://unused/v1/completions",
                4,
                10,
                "fake",
                None,
                None,
                {},
            ),
            **kwargs,
        )
    )


TEXT = {"choices": [{"text": "hello"}]}
FINISH = {"choices": [{"text": "", "finish_reason": "length"}]}
USAGE = {"choices": [], "usage": {"completion_tokens": 2, "prompt_tokens": 4}}


@pytest.mark.parametrize(
    "events",
    [
        [TEXT, FINISH, "[DONE]"],
        [TEXT, {"choices": [{"text": " world"}]}, FINISH, USAGE, "[DONE]"],
        [TEXT, {"choices": []}, FINISH],
        [TEXT, "[DONE]", b"bad ignored after terminator"],
        [b": heartbeat\n", b"\n", TEXT, FINISH, "[DONE]"],
    ],
)
def test_success(monkeypatch, events):
    output = run(monkeypatch, events)
    assert output.success
    assert output.generated_text.startswith("hello")
    assert 0 < output.ttft <= output.latency
    if USAGE in events:
        assert output.output_len == 2
        assert output.usage == {"completion_tokens": 2, "prompt_tokens": 4}
        assert len(output.itl) == 1
    else:
        assert output.output_len == 10
        assert not output.itl


@pytest.mark.parametrize(
    "events, category",
    [
        [[], "interrupted_stream"],
        [[USAGE, "[DONE]"], "empty_response"],
        [[FINISH, "[DONE]"], "empty_response"],
        [["{invalid secret output"], "malformed_event"],
        [[TEXT, {"choices": "invalid"}], "malformed_event"],
        [[{"choices": [{}]}], "malformed_event"],
        [[TEXT], "interrupted_stream"],
        [[TEXT, aiohttp.ClientPayloadError("secret output")], "interrupted_stream"],
        [[TEXT, {"error": {"message": "secret prompt"}}], "server_error"],
    ],
)
def test_failure(monkeypatch, events, category):
    output = run(monkeypatch, events)
    assert not output.success
    assert output.error_category == category
    assert output.latency > 0
    if TEXT in events:
        assert output.generated_text == "hello"
        assert output.ttft > 0
    assert "secret" not in str(client.request_errors([output]))


def test_http_error_and_nonstream(monkeypatch):
    output = run(monkeypatch, [], status=503)
    assert output.error_category == "server_error"
    output = run(
        monkeypatch,
        [{"choices": [{"text": "hello"}], "usage": {"completion_tokens": 1}}],
        disable_stream=True,
    )
    assert output.success and output.output_len == 1


def test_metadata_does_not_change_token_timing(monkeypatch):
    clock = iter([1.0, 2.0, 3.0, 5.0])
    monkeypatch.setattr(client.time, "perf_counter", lambda: next(clock))
    output = run(
        monkeypatch, [TEXT, {"choices": [{"text": " world"}]}, FINISH, USAGE, "[DONE]"]
    )
    assert output.ttft == 1.0
    assert output.itl == [1.0]
    assert output.latency == 4.0
    assert output.output_len == 2


def test_partial_usage_survives_interruption(monkeypatch):
    output = run(monkeypatch, [TEXT, USAGE, aiohttp.ClientPayloadError("broken")])
    assert not output.success
    assert output.generated_text == "hello"
    assert output.usage["completion_tokens"] == output.output_len == 2


def test_missing_usage_counts_observed_output_in_metrics(monkeypatch):
    output = run(monkeypatch, [TEXT, FINISH, "[DONE]"])

    class Tokenizer:
        def encode(self, text, **kwargs):
            return text.split()

    metrics, lengths = client.calculate_metrics(
        [client.DatasetRow("prompt", 4, 10)], [output], 1, Tokenizer(), "vllm"
    )
    assert output.output_len_is_estimate
    assert lengths == [1]
    assert metrics.total_output == 1
