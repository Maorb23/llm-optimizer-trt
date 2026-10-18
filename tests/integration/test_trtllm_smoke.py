"""Opt-in smoke test for an NVIDIA TensorRT-LLM environment."""

import os
import shutil
import subprocess

import pytest

MODEL = os.environ.get("LLM_OPTIMIZER_TRTLLM_SMOKE_MODEL")
pytestmark = pytest.mark.skipif(
    not MODEL or not shutil.which("trtllm-serve"),
    reason=(
        "set LLM_OPTIMIZER_TRTLLM_SMOKE_MODEL in a TensorRT-LLM GPU "
        "environment to enable"
    ),
)


def test_trtllm_managed_server_smoke(tmp_path):
    output = tmp_path / "trtllm-smoke.json"
    subprocess.run(
        [
            "llm-optimizer",
            "--framework",
            "trtllm",
            "--model",
            MODEL,
            "--server-args",
            "tp_size=1;max_batch_size=2;max_num_tokens=256",
            "--client-args",
            (
                "dataset_name=random;random_input_len=32;"
                "random_output_len=8;max_concurrency=1;num_prompts=2"
            ),
            "--gpus",
            "1",
            "--rest",
            "0",
            "--output-json",
            str(output),
        ],
        check=True,
        timeout=1200,
    )
    assert output.exists()
