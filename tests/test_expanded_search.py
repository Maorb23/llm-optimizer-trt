import ast
import base64
import json
import shlex
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from llm_optimizer import args as lo_args
from llm_optimizer import predefined
from llm_optimizer.remote.adapters import NebiusAdapter
from llm_optimizer.remote.contracts import OptimizationRequest
from llm_optimizer.remote.worker import _payload_from_environment, optimizer_command
from llm_optimizer.search_profiles import (
    CAPACITIES_128,
    EXPANDED_CHOICES,
    PREFILL_BUDGETS_1,
    SPACES,
    TOKEN_BUDGETS_128,
    expanded_server_args,
)


def request(framework, args, files):
    return {
        "model": "org/model",
        "revision": "a" * 40,
        "frameworks": [framework],
        "gpu_type": "gpu-h100-sxm",
        "images": {framework: "test:image"},
        "bucket": "results",
        "dry_run": False,
        "search_strategy": "optuna",
        "server_args": {framework: args},
        "server_config_files": {framework: files},
        "client_args": [
            "random_input_len=512",
            "random_output_len=128",
            "max_concurrency=32",
        ],
        "optuna": {
            "trials": 30,
            "objectives": ["output_throughput", "mean_e2e_latency_ms"],
        },
    }


@pytest.mark.parametrize("framework", ["vllm", "sglang", "trtllm"])
@pytest.mark.parametrize("concurrency", [1, 32, 128])
def test_expanded_domains_and_native_worker_serialization(
    tmp_path, framework, concurrency
):
    arguments, files = expanded_server_args(framework, concurrency)
    source = {
        a.partition("=")[0].partition(":")[0]: ast.literal_eval(a.partition("=")[2])
        for a in arguments
    }
    regime = 1 if concurrency == 1 else 128
    token_key = {
        "vllm": "max_num_batched_tokens",
        "sglang": "chunked_prefill_size",
        "trtllm": "max_num_tokens",
    }[framework]
    assert source[token_key] == SPACES[regime][framework][token_key]
    assert all(
        "cuda_graph_on" not in a and not a.startswith("graph_mode=") for a in arguments
    )
    if framework == "vllm":
        modes = [
            json.loads(shlex.split(v)[0])["cudagraph_mode"]
            for v in source["compilation_config"]
        ]
        assert modes == EXPANDED_CHOICES[framework]["graph_mode"]
        assert source["async_scheduling*no_async_scheduling"] == [
            [False, True],
            [True, False],
        ]
    if framework == "sglang":
        assert source["cuda_graph_backend_decode"] == [
            "disabled",
            "full",
        ]
        assert source["cuda_graph_backend_prefill"] == ["disabled", "breakable"]
        assert source["attention_backend"] == ["triton", "fa3", "flashinfer"]
        assert source["disable_radix_cache*schedule_policy"] == [
            [False, "fcfs"], [False, "lpm"], [True, "fcfs"], [True, "lpm"]
        ]
    data = request(framework, arguments, files)
    assert OptimizationRequest.parse(data).frameworks == [framework]
    folder = tmp_path / "directory with spaces"
    folder.mkdir()
    command = optimizer_command(
        {"request": data, "framework": framework}, folder / "results.json"
    )
    server = [command[i + 1] for i, arg in enumerate(command) if arg == "--server-args"]
    sets = lo_args.parse_args_str(
        ";".join(server),
        lo_args.ArgScope.SERVER,
        predefined.SERVER_CONFIGS[framework],
        False,
    )
    selected = [s.get_all_possible_arg_values()[-1] for s in sets]
    argv = shlex.split(
        "server " + " ".join(predefined.get_server_cmd_args(framework, selected))
    )
    if framework == "vllm":
        assert (
            json.loads(
                next(
                    a.split("=", 1)[1]
                    for a in argv
                    if a.startswith("--compilation-config=")
                )
            )["cudagraph_mode"]
            == "FULL_AND_PIECEWISE"
        )
    if framework == "trtllm":
        config = next(a.split("=", 1)[1] for a in argv if a.startswith("--config="))
        native = yaml.safe_load(Path(config).read_text())
        capacity = int(
            next(a.split("=", 1)[1] for a in argv if a.startswith("--max_batch_size="))
        )
        assert max(native["cuda_graph_config"]["batch_sizes"]) <= capacity
        assert native["attn_backend"] == "TRTLLM"
        assert native["kv_cache_config"]["free_gpu_memory_fraction"] in [
            0.65,
            0.75,
            0.85,
            0.95,
        ]
    assert "max_concurrency=32" in command
    assert (
        command[command.index("--optuna-objective") + 1]
        == "output_throughput,mean_e2e_latency_ms"
    )


def test_exact_full_grids():
    assert PREFILL_BUDGETS_1 == list(range(1024, 8193, 1024))
    assert TOKEN_BUDGETS_128 == list(range(1024, 32769, 256))
    assert CAPACITIES_128 == list(range(8, 129, 8))


def test_large_profile_transport_round_trip(monkeypatch):
    monkeypatch.delenv("NEBIUS_DEBUG_SSH_PUBLIC_KEY", raising=False)
    args, files = expanded_server_args("trtllm", 32)
    data = request("trtllm", args, files)
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout='{"id":"aijob-test"}')

    NebiusAdapter("project", run_command=run).submit(
        {"run_id": "test", "framework": "trtllm", "request": data}
    )
    encoded = next(
        v.split("=", 1)[1]
        for v in commands[0]
        if v.startswith("LLM_OPTIMIZER_PAYLOAD_ZLIB_B64=")
    )
    assert len(encoded) < 100000
    decoded = json.loads(zlib.decompress(base64.b64decode(encoded)))
    assert decoded["request"]["server_args"] == data["server_args"]
    assert decoded["request"]["server_config_files"] == data["server_config_files"]
    monkeypatch.setenv("LLM_OPTIMIZER_PAYLOAD_ZLIB_B64", encoded)
    assert _payload_from_environment() == decoded


@pytest.mark.parametrize(
    "filename", ["../escape.yaml", "/tmp/escape.yaml", "..", "a\\b.yaml"]
)
def test_config_file_names_are_validated(filename):
    with pytest.raises(ValueError, match="server_config_files"):
        OptimizationRequest.parse(request("trtllm", [], {filename: "{}"}))
