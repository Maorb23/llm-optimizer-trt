"""CPU-only serialization/identity checks; no TRT runtime or GPU execution."""

import copy
import json

import pytest
import yaml

from llm_optimizer.experiment.hf_models import fingerprint_payload
from llm_optimizer.predefined.trtllm import serialize_config


def test_speculative_identity_contains_revision_and_settings():
    profile = {
        "model_revision": "a" * 40,
        "draft_revision": "b" * 40,
        "speculative_decoding": {"algorithm": "Eagle3", "max_draft_len": 3},
        "fixed_server": {"speculative_config": {"decoding_type": "Eagle3"}},
    }
    before = fingerprint_payload("target", "trtllm", 1, {}, profile)
    for key, value in (
        ("draft_revision", "c" * 40),
        ("model_revision", "d" * 40),
        ("speculative_decoding", {"algorithm": "DFlash"}),
    ):
        changed = {**profile, key: value}
        assert fingerprint_payload("target", "trtllm", 1, {}, changed) != before
    assert fingerprint_payload("target", "trtllm", 1, {}, {}) == json.dumps(
        ["target", "trtllm", 1, {}], sort_keys=True
    )


def test_native_yaml_is_content_addressed_and_retained_in_cli_metadata(tmp_path):
    from llm_optimizer.args import Arg, ArgScope
    from llm_optimizer.cli import construct_benchmark_settings, construct_result_config

    config = {
        "speculative_config": {
            "decoding_type": "Eagle3",
            "max_draft_len": 3,
            "speculative_model": "/snapshots/" + "b" * 40,
        },
        "cuda_graph_config": None,
        "scheduler_config": {"capacity_scheduler_policy": "MAX_UTILIZATION"},
        "kv_cache_config": {"free_gpu_memory_fraction": 0.75},
    }
    original = copy.deepcopy(config)
    native = serialize_config(config, tmp_path)
    assert config == original
    assert yaml.safe_load(open(native["config"])) == config
    changed = copy.deepcopy(config)
    changed["speculative_config"]["max_draft_len"] = 4
    assert serialize_config(changed, tmp_path) != native
    settings = construct_benchmark_settings(
        [Arg[str](scope=ArgScope.SERVER, name="config", value=native["config"])],
        "trtllm",
    )
    assert settings["server_cmd_args"] == [f"--config={native['config']}"]
    result = construct_result_config(settings, "trtllm")
    assert result["native_config"] == config
    assert result["speculative_decoding"] == config["speculative_config"]
    assert result["draft_revision"] == "b" * 40


@pytest.mark.parametrize("key", ["config", "extra_llm_api_options"])
def test_yaml_collision_is_explicit(tmp_path, key):
    with pytest.raises(ValueError, match="Merge a fixed TRT YAML"):
        serialize_config({key: "fixed.yaml", "speculative_config": {}}, tmp_path)


@pytest.mark.parametrize("field", ["draft_revision", "speculative_decoding"])
@pytest.mark.parametrize("implementation", ["python", "notebook"])
def test_previous_phase_rejects_changed_draft(field, implementation, tmp_path):
    import ast
    from pathlib import Path
    from types import SimpleNamespace

    from llm_optimizer.experiment.lab import Lab

    original = {
        "measurement": "MEASURED",
        "requests_sha256": "identical",
        "draft_revision": "b" * 40,
        "speculative_decoding": {"algorithm": "Eagle3", "max_draft_len": 3},
    }
    changed = {**original, field: "changed"}
    contract = {
        "phase": "A",
        "concurrency": 1,
        "framework": "trtllm",
        "target": "e2e",
        "environment": original,
        "workload": {},
        "sampling_menu": {},
    }
    settings = {
        "RUN_REAL_GPU": True,
        "STUDIES": {"previous": SimpleNamespace(user_attrs={"contract": contract})},
        "REAL_PREFLIGHT": {},
        "sampling_menu": lambda *args: {},
    }
    benchmark = SimpleNamespace(environment=changed, workload={})
    if implementation == "python":
        lab = Lab.__new__(Lab)
        lab.__dict__.update(settings)
        verify = lab.verify_pruning_workload
    else:
        notebook = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "docs/optuna_inference_optimization_lab_prune.ipynb"
            ).read_text()
        )
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            for node in ast.parse("".join(cell["source"])).body:
                if (
                    isinstance(node, ast.FunctionDef)
                    and node.name == "verify_pruning_workload"
                ):
                    exec(
                        compile(
                            ast.Module(body=[node], type_ignores=[]), "notebook", "exec"
                        ),
                        settings,
                    )
        verify = settings["verify_pruning_workload"]
    with pytest.raises(ValueError, match=field):
        verify(benchmark, "A", 1, "trtllm", "e2e")


@pytest.mark.parametrize("field", ["draft_revision", "speculative_decoding"])
def test_regular_study_resume_rejects_changed_draft(field, tmp_path):
    from llm_optimizer.experiment.config import DEFAULTS, ExperimentConfig
    from llm_optimizer.experiment.workflow import create_lab

    lab = create_lab(ExperimentConfig(copy.deepcopy(DEFAULTS)), tmp_path)
    benchmark = lab.Simulator()
    benchmark.environment.update(
        draft_revision="b" * 40,
        speculative_decoding={"algorithm": "Eagle3", "max_draft_len": 3},
    )
    lab.run_study(
        "draft-resume",
        "trtllm",
        "random",
        concurrency=1,
        target="e2e",
        trials=1,
        benchmark=benchmark,
    )
    benchmark.environment = {**benchmark.environment, field: "changed"}
    with pytest.raises(ValueError, match="different experimental settings"):
        lab.run_study(
            "draft-resume",
            "trtllm",
            "random",
            concurrency=1,
            target="e2e",
            trials=2,
            benchmark=benchmark,
        )
