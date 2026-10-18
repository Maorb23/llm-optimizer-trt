"""Tests for Dynamo topology validation and manifest generation."""

import json
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from llm_optimizer.cli import cli
from llm_optimizer.dynamo import (
    AIConfiguratorError,
    AIConfiguratorRequest,
    build_dynamo_manifest,
    kubernetes_file_server_dry_run,
    kubernetes_server_dry_run,
    run_ai_configurator,
    validate_dynamo_config,
)


def test_vllm_multinode_topology_and_batch_limits_are_valid():
    result = validate_dynamo_config(
        "vllm",
        {
            "tensor_parallel_size": 4,
            "pipeline_parallel_size": 2,
            "max_num_seqs": 64,
            "max_num_batched_tokens": 8192,
        },
        node_count=2,
        gpus_per_node=4,
    )

    assert result.valid
    assert result.worker_gpu_count == 8
    assert result.batch_parameters == {
        "max_num_seqs": 64,
        "max_num_batched_tokens": 8192,
    }
    assert result.normalized_config["batch_scope"] == "distributed_worker"


def test_topology_mismatch_is_rejected():
    result = validate_dynamo_config(
        "sglang",
        {"tp_size": 2, "pp_size": 2, "max_running_requests": 32},
        node_count=2,
        gpus_per_node=4,
    )

    assert not result.valid
    assert "TP×PP×CP=4" in result.errors[0]


def test_non_positive_batch_limit_and_conflicting_aliases_are_rejected():
    result = validate_dynamo_config(
        "trtllm",
        {"tp_size": 4, "tensor_parallel_size": 8, "max_batch_size": 0},
        node_count=1,
        gpus_per_node=4,
    )

    assert not result.valid
    assert any("conflicting aliases" in error for error in result.errors)
    assert "max_batch_size must be at least 1" in result.errors


def test_backend_topology_argument_must_match_dynamo_topology():
    result = validate_dynamo_config(
        "sglang",
        {"tp_size": 4, "pp_size": 2, "nnodes": 3},
        node_count=2,
        gpus_per_node=4,
    )

    assert not result.valid
    assert any("nnodes=3 conflicts" in error for error in result.errors)


def test_manifest_uses_multinode_dgd_and_backend_native_args():
    manifest = build_dynamo_manifest(
        "sglang",
        "meta-llama/Llama-3.1-70B-Instruct",
        {
            "tp_size": 4,
            "pp_size": 2,
            "max_running_requests": 64,
            "max_total_tokens": 16384,
        },
        node_count=2,
        gpus_per_node=4,
        namespace="inference",
        name="llama-sglang",
    )

    worker = manifest["spec"]["components"][1]
    container = worker["podTemplate"]["spec"]["containers"][0]
    assert manifest["metadata"] == {
        "name": "llama-sglang",
        "namespace": "inference",
    }
    assert manifest["spec"]["backendFramework"] == "sglang"
    assert worker["multinode"] == {"nodeCount": 2}
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "4"
    assert container["command"] == ["python3", "-m", "dynamo.sglang"]
    assert "--max-running-requests" in container["args"]


def test_cli_expands_parameter_grid_and_reports_every_configuration():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "dynamo-check",
            "--model",
            "model",
            "--framework",
            "trtllm",
            "--node-count",
            "2",
            "--gpus-per-node",
            "4",
            "--server-args",
            "tp_size=4;pp_size=2;max_batch_size=[32,64];max_num_tokens=8192",
        ],
    )

    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["valid"]
    assert len(report["configurations"]) == 2
    assert [
        item["batch_parameters"]["max_batch_size"] for item in report["configurations"]
    ] == [32, 64]
    names = [item["manifest"]["metadata"]["name"] for item in report["configurations"]]
    assert len(set(names)) == 2


@patch("llm_optimizer.dynamo.subprocess.run")
def test_cluster_check_uses_server_side_dry_run(run: Mock):
    run.return_value = Mock(
        returncode=0, stdout="configured (server dry run)\n", stderr=""
    )

    valid, output = kubernetes_server_dry_run({"kind": "DynamoGraphDeployment"})

    assert valid
    assert "server dry run" in output
    assert run.call_args.args[0] == [
        "kubectl",
        "apply",
        "--dry-run=server",
        "-f",
        "-",
    ]


@patch("llm_optimizer.dynamo.subprocess.run")
def test_generated_file_cluster_check_uses_server_side_dry_run(run: Mock):
    run.return_value = Mock(returncode=0, stdout="valid\n", stderr="")

    valid, output = kubernetes_file_server_dry_run("results/k8s_deploy.yaml")

    assert valid
    assert output == "valid"
    assert run.call_args.args[0] == [
        "kubectl",
        "apply",
        "--dry-run=server",
        "-f",
        "results/k8s_deploy.yaml",
    ]


@patch("llm_optimizer.dynamo.subprocess.run")
def test_ai_configurator_runs_support_then_gpu_free_sizing(run: Mock, tmp_path):
    def command_result(command, **kwargs):
        if command[2] == "default":
            save_dir = command[command.index("--save-dir") + 1]
            agg_dir = tmp_path / "aic" / "agg"
            assert save_dir == str(tmp_path / "aic")
            agg_dir.mkdir(parents=True)
            (agg_dir / "best_config_topn.csv").write_text(
                "Rank,tokens/s/gpu,TTFT,parallel\n1,322.69,546.92,tp4pp1\n"
            )
            (agg_dir / "k8s_deploy.yaml").write_text("kind: DynamoGraphDeployment\n")
            return Mock(returncode=0, stdout="Best Experiment Chosen: agg\n", stderr="")
        return Mock(returncode=0, stdout="supported\n", stderr="")

    run.side_effect = command_result
    result = run_ai_configurator(
        AIConfiguratorRequest(
            model="Qwen/Qwen3-32B-FP8",
            framework="vllm",
            gpu_system="h200_sxm",
            total_gpus=8,
            input_length=4000,
            output_length=500,
            ttft_ms=600,
            tpot_ms=16.67,
            backend_version="0.12.0",
        ),
        str(tmp_path / "aic"),
    )

    assert result["gpu_required"] is False
    assert result["ranked_configurations"]["agg"][0]["parallel"] == "tp4pp1"
    assert "agg/k8s_deploy.yaml" in result["artifacts"]
    assert run.call_count == 2
    optimize_command = run.call_args_list[1].args[0]
    assert optimize_command[:3] == ["aiconfigurator", "cli", "default"]
    assert optimize_command[optimize_command.index("--total-gpus") + 1] == "8"


@patch("llm_optimizer.dynamo.subprocess.run")
def test_ai_configurator_rejects_unsupported_combination(run: Mock, tmp_path):
    run.return_value = Mock(returncode=1, stdout="", stderr="unsupported model")

    with pytest.raises(AIConfiguratorError, match="unsupported model"):
        run_ai_configurator(
            AIConfiguratorRequest(
                model="unknown/model",
                framework="sglang",
                gpu_system="h100_sxm",
                total_gpus=8,
            ),
            str(tmp_path / "aic"),
        )

    assert run.call_count == 1


@patch("llm_optimizer.dynamo.subprocess.run")
def test_ai_configurator_refuses_stale_output_directory(run: Mock, tmp_path):
    run.return_value = Mock(returncode=0, stdout="supported", stderr="")
    output_dir = tmp_path / "aic"
    output_dir.mkdir()
    (output_dir / "old-result.csv").write_text("stale")

    with pytest.raises(AIConfiguratorError, match="not empty"):
        run_ai_configurator(
            AIConfiguratorRequest(
                model="model",
                framework="vllm",
                gpu_system="h100_sxm",
                total_gpus=8,
            ),
            str(output_dir),
        )


def test_optimize_cli_requires_gpu_system():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "dynamo-optimize",
            "--model",
            "model",
            "--framework",
            "vllm",
            "--node-count",
            "1",
            "--gpus-per-node",
            "1",
        ],
    )

    assert result.exit_code == 2
    assert "Missing option '--gpu-system'" in result.output


@patch("llm_optimizer.dynamo.run_ai_configurator")
def test_cli_runs_ai_configurator_without_manual_server_args(run: Mock):
    run.return_value = {
        "status": "complete",
        "gpu_required": False,
        "ranked_configurations": {"agg": [{"parallel": "tp4pp1"}]},
        "artifacts": ["agg/top1/agg/k8s_deploy.yaml"],
    }
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "dynamo-optimize",
            "--model",
            "model",
            "--framework",
            "vllm",
            "--node-count",
            "2",
            "--gpus-per-node",
            "4",
            "--gpu-system",
            "h100_sxm",
        ],
    )

    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["mode"] == "ai-configurator"
    assert report["ai_configurator"]["gpu_required"] is False
    request = run.call_args.args[0]
    assert request.total_gpus == 8
    assert request.gpu_system == "h100_sxm"
