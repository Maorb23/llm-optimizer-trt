"""Model setup is opt-in, pinned, and shared by search and validation adapters."""

import ast
import copy
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import pytest

TARGET_SHA = "a" * 40
DRAFT_SHA = "9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf"


@pytest.fixture
def setup(tmp_path, monkeypatch):
    path = (
        Path(__file__).resolve().parents[1]
        / "docs/optuna_inference_optimization_lab_prune.ipynb"
    )
    notebook = json.loads(path.read_text())
    cell = next(c for c in notebook["cells"] if c.get("id") == "hf-sglang-model-setup")
    calls = {"downloads": [], "revisions": [], "cli": []}
    profile = {
        "gpu_type": "old",
        "model_revision": "old",
        "native_config_verified": True,
        "quality_verified": True,
        "fixed_server": {"tp_size": 1},
        "expanded_choices": {
            "enable_mixed_chunk": [False, True],
            "disable_overlap_schedule": [False, True],
            "attention_backend": ["fa3"],
        },
    }
    namespace = {
        "Path": Path,
        "json": json,
        "RUN_DIR": tmp_path,
        "RUN_REAL_GPU": False,
        "REAL_FRAMEWORKS": ("sglang",),
        "REAL_PREFLIGHT": {("B", "sglang"): profile},
        "MODEL_SNAPSHOTS": {"B": "original-target"},
        "PHASE_MODELS": {"B": "Qwen/Qwen3.5-4B"},
        "expanded_choices": lambda framework, choices: choices,
        "display": lambda result: None,
    }
    exec(compile("".join(cell["source"]), str(path), "exec"), namespace)

    def resolve(model_id, revision):
        calls["revisions"].append((model_id, revision))
        return SimpleNamespace(sha=TARGET_SHA)

    def download(**kwargs):
        calls["downloads"].append(kwargs)
        snapshot = tmp_path / kwargs["revision"]
        snapshot.mkdir(exist_ok=True)
        (snapshot / "config.json").write_text("{}")
        return str(snapshot)

    def cli(command, **kwargs):
        calls["cli"].append(command)
        return SimpleNamespace(
            returncode=0,
            stderr="",
            stdout="DFLASH --speculative-algorithm "
            "--speculative-draft-model-path --speculative-num-draft-tokens",
        )

    monkeypatch.setattr(
        huggingface_hub, "HfApi", lambda: SimpleNamespace(model_info=resolve)
    )
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True, get_device_name=lambda i: "H100"
            ),
            version=SimpleNamespace(cuda="12.8"),
        ),
    )
    monkeypatch.setattr(subprocess, "run", cli)
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: "570.0\n")
    monkeypatch.setattr(importlib.metadata, "version", lambda package: "test-sglang")
    return namespace, calls, notebook


def test_setup_disabled_and_simulator_paths_do_not_download(setup):
    namespace, calls, _ = setup
    assert namespace["HF_MODEL_SETUP_ENABLED"] is False
    assert namespace["MODEL_SNAPSHOTS"]["B"] == "original-target"
    with pytest.raises(ValueError, match="RUN_REAL_GPU"):
        namespace["configure_hf_sglang_models"]("Qwen/Qwen3.5-4B")
    assert calls["downloads"] == calls["revisions"] == []


def test_target_and_draft_are_pinned_and_passed_to_existing_adapter(setup):
    namespace, calls, notebook = setup
    namespace["RUN_REAL_GPU"] = True
    result = namespace["configure_hf_sglang_models"](
        "Qwen/Qwen3.5-4B",
        enable_dflash=True,
        draft_model_id="z-lab/Qwen3.5-4B-DFlash",
        draft_revision=DRAFT_SHA,
    )
    assert calls["revisions"] == [("Qwen/Qwen3.5-4B", "main")]
    assert [call["revision"] for call in calls["downloads"]] == [TARGET_SHA, DRAFT_SHA]
    assert calls["cli"][0][:3] == [sys.executable, "-m", "sglang.launch_server"]
    profile = namespace["REAL_PREFLIGHT"][("B", "sglang")]
    assert profile["model_revision"] == TARGET_SHA
    assert profile["draft_revision"] == DRAFT_SHA
    assert profile["gpu_type"] == "H100"
    assert profile["framework_version"] == "test-sglang"
    assert profile["native_config_verified"] is True
    assert profile["expanded_choices"]["enable_mixed_chunk"] == [False]
    assert profile["expanded_choices"]["disable_overlap_schedule"] == [True]
    assert profile["fixed_server"]["tp_size"] == 1
    assert (
        profile["fixed_server"]["speculative_draft_model_path"] == result["draft_path"]
    )
    assert namespace["MODEL_SNAPSHOTS"]["B"] == result["target_path"]
    assert (
        json.loads((namespace["RUN_DIR"] / "hf-sglang-models.json").read_text())
        == result
    )

    namespace["RepositoryBenchmark"] = lambda *args, **kwargs: SimpleNamespace(
        model=args[1],
        **kwargs,
    )
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            for node in ast.parse("".join(cell["source"])).body:
                if isinstance(node, ast.FunctionDef) and node.name == "benchmark_for":
                    exec(
                        compile(
                            ast.Module(body=[node], type_ignores=[]),
                            "benchmark-for",
                            "exec",
                        ),
                        namespace,
                    )
    benchmark = namespace["benchmark_for"]("B", "sglang", 1, "search")
    validation = namespace["benchmark_for"](
        "B",
        "sglang",
        1,
        "validation",
        num_prompts=1000,
        repeat=1,
    )
    assert benchmark.model == result["target_path"]
    assert benchmark.fixed_server == validation.fixed_server == profile["fixed_server"]
    assert benchmark.environment["speculative_decoding"]["draft_revision"] == DRAFT_SHA

    # Exercise the actual registry/CLI encoding used by RepositoryBenchmark.measure.
    from llm_optimizer.args import Arg, ArgScope
    from llm_optimizer.cli import construct_benchmark_settings
    from llm_optimizer.predefined import SERVER_CONFIGS

    args = [
        Arg[SERVER_CONFIGS["sglang"][key].arg_type](
            scope=ArgScope.SERVER, name=key, value=value
        )
        for key, value in benchmark.fixed_server.items()
    ]
    settings = construct_benchmark_settings(args, "sglang")
    assert "--speculative-algorithm=DFLASH" in settings["server_cmd_args"]
    assert "--speculative-num-draft-tokens=16" in settings["server_cmd_args"]
    assert (
        f"--speculative-draft-model-path={result['draft_path']}"
        in settings["server_cmd_args"]
    )


def test_turning_draft_off_restores_choices_and_keeps_other_fixed_settings(setup):
    namespace, calls, _ = setup
    namespace["RUN_REAL_GPU"] = True
    configure = namespace["configure_hf_sglang_models"]
    configure(
        "Qwen/Qwen3.5-4B",
        enable_dflash=True,
        draft_model_id="z-lab/Qwen3.5-4B-DFlash",
        draft_revision=DRAFT_SHA,
    )
    result = configure("Qwen/Qwen3.5-4B", target_revision=TARGET_SHA)
    profile = namespace["REAL_PREFLIGHT"][("B", "sglang")]
    assert profile["fixed_server"] == {"tp_size": 1}
    assert "draft_revision" not in profile
    assert "speculative_decoding" not in profile
    assert profile["expanded_choices"]["enable_mixed_chunk"] == [False, True]
    assert result["draft_path"] is None
    assert len(calls["downloads"]) == 3


def test_runtime_without_dflash_fails_before_download_or_profile_changes(
    setup, monkeypatch
):
    namespace, calls, _ = setup
    namespace["RUN_REAL_GPU"] = True
    before = copy.deepcopy(namespace["REAL_PREFLIGHT"])
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="old help", stderr=""),
    )
    with pytest.raises(RuntimeError, match="required DFlash flags"):
        namespace["configure_hf_sglang_models"](
            "Qwen/Qwen3.5-4B",
            enable_dflash=True,
            draft_model_id="z-lab/Qwen3.5-4B-DFlash",
        )
    assert calls["downloads"] == []
    assert namespace["REAL_PREFLIGHT"] == before


def test_failed_draft_download_does_not_change_active_models(setup, monkeypatch):
    namespace, _, _ = setup
    namespace["RUN_REAL_GPU"] = True
    before = copy.deepcopy(namespace["REAL_PREFLIGHT"])
    hub = sys.modules["huggingface_hub"]
    original = hub.snapshot_download

    def download(**kwargs):
        if kwargs["repo_id"].startswith("z-lab/"):
            raise RuntimeError("Download failed")
        return original(**kwargs)

    monkeypatch.setattr(hub, "snapshot_download", download)
    with pytest.raises(RuntimeError, match="Download failed"):
        namespace["configure_hf_sglang_models"](
            "Qwen/Qwen3.5-4B",
            enable_dflash=True,
            draft_model_id="z-lab/Qwen3.5-4B-DFlash",
            draft_revision=DRAFT_SHA,
        )
    assert namespace["REAL_PREFLIGHT"] == before
    assert namespace["MODEL_SNAPSHOTS"]["B"] == "original-target"
