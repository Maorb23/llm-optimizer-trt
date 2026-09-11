"""Unit coverage for first-class TensorRT-LLM support."""

import json
from pathlib import Path
from typing import Optional
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

import llm_optimizer.args as lo_args
import llm_optimizer.predefined as predefined
from llm_optimizer.cli import (
    cli,
    construct_benchmark_settings,
    construct_result_config,
)
from llm_optimizer.common import ModelConfig
from llm_optimizer.exceptions import ServerNotReadyError
from llm_optimizer.predefined import trtllm
from llm_optimizer.result_utils import normalize_config, normalize_result_record
from llm_optimizer.server_utils import get_server_env, start_server, wait_for_server
from llm_optimizer.tuning.generation import (
    generate_advanced_tuning_configs,
    generate_simple_tuning_configs,
)
from llm_optimizer.visualization.visualize import ParetoLLMOptimizer


def _parse_server_args(value: str):
    return lo_args.parse_args_str(
        value,
        scope=lo_args.ArgScope.SERVER,
        configs=trtllm.SERVER_CONFIGS,
        strict=True,
    )


def _result(command: str, server_args: dict, framework: Optional[str] = None):
    config = {
        "client_args": {"max_concurrency": 8},
        "server_args": server_args,
        "server_cmd_args": [],
    }
    if framework:
        config["framework"] = framework
    return {
        "config": config,
        "results": {
            "output_throughput": 10.0,
            "mean_ttft_ms": 5.0,
        },
        "cmd": command,
    }


def test_framework_and_native_arguments_are_registered():
    assert "trtllm" in predefined.SERVER_CONFIGS
    assert predefined.get_framework_spec("trtllm").default_port == 8000
    assert trtllm.SERVER_CONFIGS["max_batch_size"].arg_type is int
    assert trtllm.SERVER_CONFIGS["kv_cache_dtype"].arg_type is str


def test_parameter_grid_and_native_command_spelling():
    server_sets = _parse_server_args("tp_size=[1,2];max_batch_size=[32,64]")
    client_sets = lo_args.parse_args_str(
        "max_concurrency=[8,16]",
        scope=lo_args.ArgScope.CLIENT,
        configs=predefined.CLIENT_CONFIGS,
        strict=True,
    )
    combinations = lo_args.get_all_arg_combinations(client_sets, server_sets)

    assert len(combinations) == 8
    settings = construct_benchmark_settings(list(combinations[-1]), "trtllm")
    assert "--tp_size=2" in settings["server_cmd_args"]
    assert "--max_batch_size=64" in settings["server_cmd_args"]
    assert settings["framework"] == "trtllm"
    assert settings["normalized_config"]["tensor_parallel_size"] == 2
    assert settings["normalized_config"]["max_concurrent_sequences"] == 64
    assert settings["normalized_config"]["world_size"] == 2


def test_trtllm_negative_telemetry_flag_uses_native_click_spelling():
    server_sets = _parse_server_args("telemetry=False")
    combo = server_sets[0].get_all_possible_arg_values()
    settings = construct_benchmark_settings(combo, "trtllm")
    assert settings["server_cmd_args"] == ["--no-telemetry"]


def test_engine_initialization_and_runtime_arguments_are_classified():
    engine, runtime = trtllm.classify_server_args(
        {
            "tp_size": 2,
            "max_num_tokens": 8192,
            "kv_cache_dtype": "fp8",
            "log_level": "debug",
            "num_serve_frontends": 2,
        }
    )

    assert engine == {
        "tp_size": 2,
        "max_num_tokens": 8192,
        "kv_cache_dtype": "fp8",
    }
    assert runtime == {"log_level": "debug", "num_serve_frontends": 2}


def test_trtllm_result_configuration_serializes_lifecycle_metadata():
    settings = {
        "framework": "trtllm",
        "client_args": {"max_concurrency": 8},
        "server_args": {"max_batch_size": 32, "log_level": "info"},
        "server_cmd_args": ["--max_batch_size=32", "--log_level=info"],
        "normalized_config": {"max_concurrent_sequences": 32},
    }
    serialized = json.loads(
        json.dumps(construct_result_config(settings, "trtllm"), sort_keys=True)
    )

    assert serialized["engine_args"] == {"max_batch_size": 32}
    assert serialized["runtime_args"] == {"log_level": "info"}
    assert serialized["normalized_config"] == {"max_concurrent_sequences": 32}


def test_gpu_assignment_honors_existing_visibility(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,7,8,9")
    environment = get_server_env(
        "trtllm", 4, {"tp_size": 2, "pp_size": 2, "cp_size": 1}
    )
    assert environment == {"CUDA_VISIBLE_DEVICES": "4,7,8,9"}


def test_gpu_assignment_rejects_impossible_parallelism():
    with pytest.raises(ValueError, match="requires 8 GPUs"):
        get_server_env("trtllm", 4, {"tp_size": 4, "pp_size": 2})


def test_gpu_assignment_rejects_conflicting_native_aliases():
    with pytest.raises(ValueError, match="Conflicting TensorRT-LLM aliases"):
        get_server_env("trtllm", 4, {"tp_size": 2, "tensor_parallel_size": 4})


def test_normalized_parameters_match_across_frameworks():
    vllm = normalize_config(
        "vllm",
        {"tensor_parallel_size": 2, "max_num_batched_tokens": 8192},
    )
    sglang = normalize_config("sglang", {"tp_size": 2, "max_total_tokens": 8192})
    trt = normalize_config("trtllm", {"tp_size": 2, "max_num_tokens": 8192})

    assert (
        vllm
        == sglang
        == trt
        == {
            "tensor_parallel_size": 2,
            "max_batched_tokens": 8192,
            "world_size": 2,
        }
    )


def test_legacy_result_deserialization_is_backward_compatible():
    legacy = _result(
        "vllm serve model --tensor-parallel-size=2",
        {"tensor_parallel_size": 2, "max_num_batched_tokens": 4096},
    )
    normalized = normalize_result_record(json.loads(json.dumps(legacy)))

    assert normalized["config"]["framework"] == "vllm"
    assert normalized["config"]["normalized_config"] == {
        "tensor_parallel_size": 2,
        "max_batched_tokens": 4096,
        "max_concurrency": 8,
        "world_size": 2,
    }
    assert "normalized_config" not in legacy["config"]


def test_estimator_generates_native_trtllm_tuning_parameters():
    model = ModelConfig(
        num_params=1_000_000_000,
        num_layers=24,
        hidden_dim=2048,
        num_heads=16,
    )
    simple = generate_simple_tuning_configs(
        framework="trtllm",
        num_gpus=2,
        gpu_name="A100",
        model_config=model,
        optimal_concurrency=32,
    )
    advanced = generate_advanced_tuning_configs(
        framework="trtllm",
        num_gpus=2,
        gpu_name="A100",
        model_config=model,
        optimal_concurrency=32,
    )

    assert simple[0].framework == "trtllm"
    assert simple[0].server_arg_sets[0].name == "tp_size"
    assert advanced[0].framework == "trtllm"
    assert any(arg.name == "max_num_tokens" for arg in advanced[0].server_arg_sets)


def test_visualization_loads_and_normalizes_multiple_framework_files(tmp_path):
    vllm_file = tmp_path / "vllm.json"
    trt_file = tmp_path / "trtllm.json"
    vllm_file.write_text(
        json.dumps(
            {
                "metadata": {"model_tag": "model", "total_tests": 1},
                "best_configurations": {},
                "test_results": [
                    _result(
                        "vllm serve model",
                        {"tensor_parallel_size": 2},
                    )
                ],
            }
        )
    )
    trt_file.write_text(
        json.dumps(
            {
                "metadata": {"model_tag": "model", "total_tests": 1},
                "best_configurations": {},
                "test_results": [
                    _result("trtllm-serve model", {"tp_size": 2}, "trtllm")
                ],
            }
        )
    )
    config_path = (
        Path(__file__).parents[1]
        / "src"
        / "llm_optimizer"
        / "visualization"
        / "visualization_config.json"
    )
    optimizer = ParetoLLMOptimizer(str(config_path))

    loaded = optimizer.load_benchmark_data(f"{vllm_file},{trt_file}")

    assert len(loaded["data"]) == 2
    assert {item["config"]["framework"] for item in loaded["data"]} == {
        "vllm",
        "trtllm",
    }
    assert all(
        item["config"]["normalized_config"]["tensor_parallel_size"] == 2
        for item in loaded["data"]
    )
    assert loaded["metadata"]["total_tests"] == 2

    dashboard = tmp_path / "comparison.html"
    optimizer.generate_dashboard(f"{vllm_file},{trt_file}", output_file=str(dashboard))
    html = dashboard.read_text()
    assert '"framework": "vllm"' in html
    assert '"framework": "trtllm"' in html
    assert '"tensor_parallel_size": 2' in html


def test_wait_for_server_stops_retrying_when_process_exits():
    process = Mock()
    process.poll.return_value = 1
    process.returncode = 1
    with patch("llm_optimizer.server_utils.requests.get") as request:
        assert not wait_for_server(
            "http://localhost:8000/health",
            max_retries=10,
            delay=0,
            process=process,
        )
    request.assert_not_called()


def test_startup_failure_cleans_up_process():
    process = Mock()
    process.pid = 123
    with (
        patch("llm_optimizer.server_utils.psutil.Popen", return_value=process),
        patch("llm_optimizer.server_utils.wait_for_server", return_value=False),
        patch("llm_optimizer.server_utils.terminate_process_top_down") as terminate,
        pytest.raises(ServerNotReadyError),
    ):
        start_server(
            "trtllm-serve model",
            {"CUDA_VISIBLE_DEVICES": "0"},
            "http://localhost:8000/health",
            mute=True,
        )
    terminate.assert_called_once_with(process)


def test_cli_dry_run_uses_managed_trtllm_command(tmp_path, monkeypatch):
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        cli,
        [
            "--framework",
            "trtllm",
            "--model",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--server-args",
            "tp_size=2;max_batch_size=32",
            "--client-args",
            "max_concurrency=8;num_prompts=10",
            "--gpus",
            "2",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "'framework': 'trtllm'" in result.output
    assert "--tp_size=2" in result.output
