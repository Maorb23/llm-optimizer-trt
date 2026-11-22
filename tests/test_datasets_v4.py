"""CPU-only v4 regression tests. No test launches a GPU server."""

import json
import subprocess
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_optimizer.experiment.v4.datasets import (
    FrozenManifest,
    Sampling,
    adapt,
    canonical,
    freeze,
    sample,
)
from llm_optimizer.experiment.v4.experiment import (
    DatasetExperimentV4,
    ExperimentSettings,
)
from llm_optimizer.experiment.v4.runtime import (
    Completion,
    Generation,
    SGLangRuntimeV4,
    TRTLLMRuntimeV4,
    VLLMRuntimeV4,
    http_generate,
)
from llm_optimizer.experiment.v4.scorers import (
    ScorerSettings,
    aggregate_scores,
    code_solution,
    executable_score,
    extraction_score,
    gpqa_score,
    score,
)

SHA = "a" * 40
IMAGE = "sha256:" + "b" * 64
RUNTIMES = [VLLMRuntimeV4, TRTLLMRuntimeV4, SGLangRuntimeV4]


def extraction_example(i=0):
    return adapt(
        {
            "id": str(i),
            "instruction": "Extract name",
            "text": f"Person {i}",
            "schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
                "additionalProperties": False,
            },
            "expected": {"name": f"secret-{i}"},
        },
        "extraction",
    )


def manifest(examples=None, **kwargs):
    examples = examples or [extraction_example()]
    options = {
        "dataset_id": "local/test",
        "revision": "sha256:" + "c" * 64,
        "source_format": "extraction",
        "split": "test",
        "model": "model",
        "model_revision": SHA,
        "tokenizer_revision": SHA,
        "generation": Generation(),
        "sampling": Sampling(count=len(examples)),
        "evaluation": ScorerSettings().identity,
    }
    options.update(kwargs)
    return freeze(examples, **options)


@pytest.mark.parametrize("fmt", ["messages", "sharegpt"])
def test_entire_history_and_only_final_assistant_is_held_out(fmt):
    turns = [
        ("system", "Be helpful"),
        ("user", "first question"),
        ("assistant", "first answer"),
        ("user", "follow-up"),
        ("assistant", "secret-gold"),
    ]
    row = (
        {"messages": [{"role": r, "content": c} for r, c in turns]}
        if fmt == "messages"
        else {
            "conversations": [
                {"from": {"user": "human", "assistant": "gpt"}.get(r, r), "value": c}
                for r, c in turns
            ]
        }
    )
    example = adapt(row, fmt)
    assert example.payload["messages"] == [
        {"role": r, "content": c} for r, c in turns[:-1]
    ]
    assert example.reference == {"target": "secret-gold"}
    assert "secret-gold" not in example.payload_json
    # Property access returns detached objects.
    example.payload["messages"].clear()
    assert len(example.payload["messages"]) == 4
    with pytest.raises(FrozenInstanceError):
        example.task = "other"


def test_no_target_and_alpaca_conventions():
    assert (
        adapt(
            {"messages": [{"role": "user", "content": "question"}]}, "messages"
        ).reference
        == {}
    )
    example = adapt(
        {"instruction": "Task", "input": "extra", "output": "gold"}, "alpaca"
    )
    assert example.payload["messages"][0]["content"] == "Task\n\nextra"
    assert example.reference == {"target": "gold"}
    assert "gold" not in example.payload_json


@pytest.mark.parametrize(
    "row",
    [
        [],
        [{"role": "tool", "content": "x"}],
        [{"role": "user", "content": ""}],
        [{"role": "system", "content": "x"}],
    ],
)
def test_malformed_conversations_rejected(row):
    with pytest.raises(ValueError):
        adapt({"messages": row}, "messages")


def test_gpqa_diamond_labels_and_gold_projection():
    row = {
        "id": "q",
        "Question": "Which one?",
        "Correct Answer": "correct",
        **{f"Incorrect Answer {i}": f"wrong-{i}" for i in range(1, 4)},
        "Explanation": "secret rationale",
    }
    first = adapt(row, "gpqa_diamond", seed=7)
    assert first == adapt(row, "gpqa_diamond", seed=7)
    letter = first.reference["answer"]
    assert f"{letter}. correct" in first.payload["messages"][0]["content"]
    assert "secret rationale" not in first.payload_json
    assert "Correct Answer" not in first.payload_json
    with pytest.raises(ValueError, match="dataset_config"):
        manifest([first], source_format="gpqa_diamond")


@pytest.mark.parametrize("task", ["humaneval", "humaneval_plus"])
def test_code_references_never_in_payload(task):
    row = {
        "task_id": "HumanEval/0",
        "prompt": "def f(x):\n",
        "entry_point": "f",
        "canonical_solution": "    return x\n",
        "test": "def check(candidate): assert candidate(1) == 1",
        "base_input": [[1]],
        "plus_input": [[2]],
        "atol": 0,
    }
    example = adapt(row, task)
    assert example.payload["messages"][0]["content"].endswith(row["prompt"])
    for text in (
        "canonical_solution",
        "return x",
        "assert candidate",
        "base_input",
        "plus_input",
    ):
        assert text not in example.payload_json
    assert example.reference["test"] == row["test"]


def test_manifest_deterministic_frozen_and_separate(tmp_path):
    first = manifest()
    second = manifest()
    assert first == second and first.signature == second.signature
    assert "secret-0" not in first.document_json
    directory = first.save(tmp_path)
    assert FrozenManifest.load(directory) == first
    assert first.save(tmp_path) == directory
    assert (
        json.loads((directory / "references.json").read_text())["0"]["expected"]["name"]
        == "secret-0"
    )
    tampered = first.document
    tampered["examples"][0]["payload"]["messages"][0]["content"] += "tampered"
    with pytest.raises(ValueError, match="Payload hash"):
        FrozenManifest(canonical(tampered), first.references_json).verify()
    with pytest.raises(ValueError, match="reference hash"):
        FrozenManifest(first.document_json, "{}").verify()


@pytest.mark.parametrize("revision", ["main", "latest", "v1", "", "abc"])
def test_mutable_revisions_rejected(revision):
    with pytest.raises(ValueError, match="immutable"):
        manifest(revision=revision)


def test_natural_and_bucket_sampling_preserve_text():
    examples = [
        adapt({"id": str(i), "instruction": "text " * (i + 1)}, "alpaca")
        for i in range(8)
    ]
    natural = sample(examples, Sampling(count=4, seed=9))
    assert natural == sample(reversed(examples), Sampling(count=4, seed=9))
    chosen = sample(
        examples,
        Sampling("token_bucket", 4, 1, ((1, 4, 2), (4, 9, 2))),
        lambda messages: len(messages[0]["content"].split()),
    )
    assert len({e.example_id for e in chosen}) == 4
    assert all(e in examples for e in chosen)
    with pytest.raises(ValueError, match="Insufficient"):
        sample(examples, Sampling(count=9))
    with pytest.raises(ValueError, match="Insufficient"):
        sample(examples, Sampling("token_bucket", 2, 0, ((99, 100, 2),)), lambda _: 3)
    with pytest.raises(ValueError, match="disjoint"):
        Sampling("token_bucket", 2, 0, ((0, 10, 1), (9, 20, 1)))


@pytest.mark.parametrize("cls", RUNTIMES)
def test_runtime_payloads_have_only_inference_fields(cls):
    runtime = cls("model", runtime_version="test")
    frozen = manifest()
    payload = runtime.preflight(frozen)[0]
    assert payload["messages"] == frozen.document["examples"][0]["payload"]["messages"]
    assert payload["max_tokens"] == 256 and payload["stream"]
    assert "expected" not in canonical(payload) and "secret-0" not in canonical(payload)
    assert payload["seed"] == 0 and payload["n"] == 1
    if cls != TRTLLMRuntimeV4:
        assert payload["ignore_eos"] is False
    with pytest.raises(ValueError, match="reasoning budget"):
        runtime.preflight(manifest(generation=Generation(reasoning_tokens=12)))
    with pytest.raises(ValueError, match="thinking"):
        runtime.preflight(manifest(generation=Generation(thinking="on")))
    with pytest.raises(RuntimeError, match="disabled"):
        runtime.generate(frozen)


@pytest.mark.parametrize("cls", [VLLMRuntimeV4, SGLangRuntimeV4])
def test_thinking_and_eos_explicit_payload(cls):
    runtime = cls("model", runtime_version="test", thinking_template_verified=True)
    payload = runtime.preflight(
        manifest(generation=Generation(thinking="off", eos="ignore"))
    )[0]
    assert payload["ignore_eos"] is True
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    with pytest.raises(ValueError, match="ignore_eos"):
        TRTLLMRuntimeV4("model", runtime_version="test").preflight(
            manifest(generation=Generation(eos="ignore"))
        )


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Final answer: B", 1),
        ("B", 1),
        ("I considered B, but Final answer: A", 0),
        ("Final answer: A\nFinal answer: B", 0),
        ("Reasoning B", 0),
        ("Final answer: B\nmore text", 0),
    ],
)
def test_gpqa_final_answer(text, expected):
    assert gpqa_score(text, {"answer": "B"})["accuracy"] == expected


def test_extraction_validity_schema_and_field_accuracy():
    reference = extraction_example().reference
    assert extraction_score('{"name":"secret-0"}', reference) == {
        "json_validity": 1.0,
        "schema_compliance": 1.0,
        "field_accuracy": 1.0,
    }
    assert extraction_score('{"name":4}', reference)["schema_compliance"] == 0
    assert extraction_score('{"name":"wrong"}', reference)["field_accuracy"] == 0
    assert (
        extraction_score('{"name":"secret-0","extra":1}', reference)["field_accuracy"]
        == 0.5
    )
    for text in (
        "{",
        '{"name": NaN}',
        '{"name":"a","name":"secret-0"}',
        "```json\n{}\n```",
    ):
        assert extraction_score(text, reference)["json_validity"] == 0
    reference = {"schema": {}, "expected": {"nested": [1, True]}}
    assert extraction_score('{"nested": [1, 1]}', reference)["field_accuracy"] == 0.5


def test_failure_types_and_all_request_denominator():
    ref = extraction_example().reference
    good = score(
        "extraction", Completion("good", '{"name":"secret-0"}'), ref, ScorerSettings()
    )
    bad = score(
        "extraction",
        Completion("bad", generation_error="network"),
        ref,
        ScorerSettings(),
    )
    invalid = score(
        "extraction",
        Completion("invalid", "{", finish_reason="length"),
        ref,
        ScorerSettings(),
    )
    scorer_failure = score(
        "extraction",
        Completion("scorer", "{}"),
        {"schema": {"type": "nonsense"}, "expected": {}},
        ScorerSettings(),
    )
    assert scorer_failure["scorer_error"]
    summary = aggregate_scores([good, bad, invalid, scorer_failure])
    assert summary["quality"]["field_accuracy"] == 0.25
    assert summary["generation_failures"] == 1
    assert summary["scorer_failures"] == 1
    assert summary["truncated"] == 1


def test_container_isolation_and_pass_fail(monkeypatch):
    commands = []
    returncode = [0]

    def run(command, **kwargs):
        commands.append(command)
        if command[1] == "run":
            mount = command[command.index("--mount") + 1]
            root = Path(mount.split("src=", 1)[1].split(",", 1)[0])
            data = json.loads((root / "input.json").read_text())
            assert data["code"] == "def f(x):\n    return x\n"
        return SimpleNamespace(returncode=returncode[0])

    monkeypatch.setattr(subprocess, "run", run)
    ref = {
        "prompt": "def f(x):\n",
        "entry_point": "f",
        "test": "def check(c): assert c(1) == 1",
    }
    settings = ScorerSettings(image=IMAGE)
    assert executable_score("humaneval", "    return x\n", ref, settings) == {
        "pass@1": 1.0
    }
    command = commands[0]
    for flag in (
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--pids-limit=64",
        "--memory=1g",
    ):
        assert flag in command
    assert "--gpus" not in command
    assert commands[1][:3] == ["docker", "rm", "-f"]
    returncode[0] = 10
    assert executable_score("humaneval", "    return x\n", ref, settings) == {
        "pass@1": 0.0
    }
    returncode[0] = 125
    with pytest.raises(RuntimeError, match="infrastructure"):
        executable_score("humaneval", "    return x\n", ref, settings)
    with pytest.raises(ValueError, match="digest-pinned"):
        ScorerSettings(image="python:latest")


def test_container_timeout_cleans_up(monkeypatch):
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if command[1] == "run":
            raise subprocess.TimeoutExpired(command, 1)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(RuntimeError, match="timeout"):
        executable_score(
            "humaneval",
            "pass",
            {"prompt": "", "entry_point": "f", "test": "def check(c): pass"},
            ScorerSettings(image=IMAGE),
        )
    assert commands[-1][1:3] == ["rm", "-f"]


def test_code_completion_and_solution_modes():
    ref = {"prompt": "def f():\n", "entry_point": "f"}
    assert code_solution("    return 1", ref, "auto") == "def f():\n    return 1"
    assert (
        code_solution("```python\ndef f():\n    return 1\n```", ref, "auto")
        == "def f():\n    return 1"
    )


def test_streaming_payload_usage_and_reasoning(monkeypatch):
    import requests

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        def iter_lines(self):
            for event in [
                {"choices": [{"delta": {"reasoning_content": "private"}}]},
                {
                    "choices": [
                        {
                            "delta": {"content": "Final answer: A"},
                            "finish_reason": "stop",
                        }
                    ]
                },
                {"choices": [], "usage": {"completion_tokens": 7}},
            ]:
                yield b"data: " + json.dumps(event).encode()
            yield b"data: [DONE]"

    monkeypatch.setattr(requests, "post", lambda *a, **kw: Response())
    output = http_generate("http://unused", {}, 1)
    assert output["text"] == "Final answer: A"
    assert output["reasoning"] == "private" and output["output_tokens"] == 7
    assert output["ttft_s"] is not None


def test_heldout_identity_thresholds_and_optuna(tmp_path):
    frozen = manifest()
    valid = manifest([extraction_example(1)], role="validation")
    calls = []

    def transport(url, payload, timeout):
        calls.append(payload)
        i = 1 if "Person 1" in payload["messages"][0]["content"] else 0
        return {
            "text": json.dumps({"name": f"secret-{i}"}),
            "finish_reason": "stop",
            "output_tokens": 8,
        }

    runtime = VLLMRuntimeV4("model", runtime_version="synthetic", transport=transport)
    settings = ExperimentSettings(
        trials=2, quality_thresholds=(("field_accuracy", 1.0),)
    )
    experiment = DatasetExperimentV4(
        runtime,
        frozen,
        validation_manifest=valid,
        settings=settings,
        output_dir=tmp_path,
        managed=False,
    )
    study, validation = experiment.run(storage="sqlite:///:memory:")
    assert len(study.trials) == 2 and len(calls) == 3
    assert validation["role"] == "validation" and validation["quality_feasible"]
    assert (
        study.best_trial.user_attrs["execution"]["phase_status"]["benchmark"]
        == "complete"
    )
    assert validation["execution"]["server_telemetry"]["enabled"] is False
    assert study.user_attrs["held_out"]["feasible"]
    assert validation["scoring_seconds"] > 0
    other = DatasetExperimentV4(
        runtime,
        manifest(generation=Generation(output_tokens=30)),
        settings=settings,
        output_dir=tmp_path,
        managed=False,
    )
    assert experiment.study_name != other.study_name
    with pytest.raises(ValueError, match="overlap"):
        DatasetExperimentV4(
            runtime,
            frozen,
            validation_manifest=manifest(role="validation"),
            managed=False,
        )
    with pytest.raises(ValueError, match="Scorer settings"):
        DatasetExperimentV4(
            runtime, frozen, scorer=ScorerSettings(timeout_s=9), managed=False
        )


def test_quality_pruning_and_missing_usage(tmp_path):
    def transport(*args):
        return {"text": "{}", "finish_reason": "length", "output_tokens": 10}

    runtime = VLLMRuntimeV4("model", runtime_version="test", transport=transport)
    experiment = DatasetExperimentV4(
        runtime,
        manifest(),
        output_dir=tmp_path,
        managed=False,
        settings=ExperimentSettings(
            trials=1, quality_thresholds=(("field_accuracy", 1),)
        ),
    )
    study, validation = experiment.run(storage="sqlite:///:memory:")
    assert study.trials[0].state.name == "PRUNED" and validation is None
    assert study.trials[0].user_attrs["metrics"]["truncated"] == 1


@pytest.mark.parametrize("cls", RUNTIMES)
def test_native_launch_serialization_uses_existing_registry(cls):
    runtime = cls("model", runtime_version="test")
    config = {
        "vllm": {"gpu_memory_utilization": 0.8},
        "sglang": {"mem_fraction_static": 0.8},
        "trtllm": {"max_batch_size": 4},
    }[runtime.framework]
    command, settings = runtime.command("/pinned/model", "/pinned/tokenizer", config)
    assert "/pinned/model" in command and "/pinned/tokenizer" in command
    assert all(settings["server_args"][k] == v for k, v in config.items())
    with pytest.raises(ValueError, match="Unsupported native"):
        runtime.command("model", "tokenizer", {"imaginary_flag": True})
    with pytest.raises(ValueError, match="identity"):
        runtime.command("model", "tokenizer", {"revision": "main"})


@pytest.mark.parametrize(
    "name",
    [
        "phase1_vllm__datasets_v4.ipynb",
        "phase1_trtllm_datasets_v4.ipynb",
        "phase1_sglang_datasets_v4.ipynb",
    ],
)
def test_notebooks_run_cpu_only(name, tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    notebook = json.loads((root / "docs" / name).read_text())
    namespace = {"display": lambda *args: None}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            exec(compile(source, name, "exec"), namespace)
            if "CAMPAIGN =" in source:
                namespace["RUN_ROOT"] = tmp_path / name
    assert namespace["RUN_REAL_GPU"] is False and namespace["WANDB_ENABLED"] is False
    assert namespace["DATASETS"].training
    assert namespace["DATASETS"].validation
    assert "B_loaded_throughput_studies" not in namespace
    assert notebook["metadata"]["kernelspec"]["name"] == "python3"
    plt.close("all")


@pytest.mark.parametrize(
    "body,passed",
    [
        ("    return x + 1\n", True),
        ("    return x\n", False),
        ("    raise ValueError('bad')\n", False),
        ("    import os; os._exit(0)\n", False),
        ("    while True: pass\n", False),
    ],
)
def test_executable_harness_with_trusted_cpu_fixtures(tmp_path, body, passed):
    """Exercise the driver using only trusted, fixed CPU fixtures."""
    import sys

    from llm_optimizer.experiment.v4.scorers import _CONTAINER_DRIVER

    reference = {
        "prompt": "def f(x):\n",
        "entry_point": "f",
        "test": "def check(candidate):\n    assert candidate(1) == 2\n",
    }
    source = tmp_path / "input.json"
    source.write_text(
        canonical(
            {
                "task": "humaneval",
                "reference": reference,
                "code": reference["prompt"] + body,
                "timeout_s": 0.3,
            }
        )
    )
    driver = tmp_path / "driver.py"
    driver.write_text(
        _CONTAINER_DRIVER.replace("'/work/input.json'", repr(str(source)))
    )
    result = subprocess.run([sys.executable, str(driver)], timeout=5, check=False)
    assert result.returncode == (0 if passed else 10)


def test_runtime_failures_preserve_accounting(tmp_path):
    def transport(*args):
        raise RuntimeError("broken transport")

    runtime = SGLangRuntimeV4("model", runtime_version="test", transport=transport)
    result = runtime.generate(manifest(), output_dir=tmp_path, managed=False)
    assert result["metrics"]["generation_failures"] == 1
    assert result["metrics"]["completed"] == 0
    assert result["metrics"]["output_tokens_per_second"] is None
    assert "broken transport" in result["completions"][0].generation_error


def test_reference_projection_is_enforced_even_for_manual_examples():
    example = extraction_example()
    payload = example.payload
    payload["expected"] = example.reference["expected"]
    with pytest.raises(ValueError, match="only messages"):
        manifest([replace(example, payload_json=canonical(payload))])


def test_settings_change_study_identity(tmp_path):
    runtime = VLLMRuntimeV4("model", runtime_version="test", transport=lambda *a: {})
    first = DatasetExperimentV4(runtime, manifest(), output_dir=tmp_path, managed=False)
    variants = [
        DatasetExperimentV4(
            runtime,
            manifest(),
            output_dir=tmp_path,
            managed=False,
            settings=ExperimentSettings(quality_thresholds=(("field_accuracy", 0.5),)),
        ),
        DatasetExperimentV4(
            runtime,
            manifest(generation=Generation(thinking="default", eos="ignore")),
            output_dir=tmp_path,
            managed=False,
        ),
        DatasetExperimentV4(
            VLLMRuntimeV4(
                "model", runtime_version="test", concurrency=2, transport=lambda *a: {}
            ),
            manifest(),
            output_dir=tmp_path,
            managed=False,
        ),
    ]
    assert len({first.study_name, *(e.study_name for e in variants)}) == 4


def test_scorer_and_generation_failures_both_zero_pass_at_one():
    good = {
        "example_id": "good",
        "metrics": {"pass@1": 1.0},
        "generation_error": None,
        "scorer_error": None,
        "truncated": False,
    }
    bad = {**good, "example_id": "bad", "metrics": {}, "generation_error": "failed"}
    assert aggregate_scores([good, bad])["quality"]["pass@1"] == 0.5


@pytest.mark.parametrize("cls", RUNTIMES)
def test_managed_lifecycle_pins_snapshots_and_cleans_up(cls, tmp_path, monkeypatch):
    import importlib.metadata

    import huggingface_hub

    from llm_optimizer import server_utils
    from llm_optimizer.experiment.v4 import runtime as runtime_module

    events = []
    monkeypatch.setattr(importlib.metadata, "version", lambda package: "test")

    def snapshot(repo, revision):
        events.append(("snapshot", repo, revision))
        return "/pinned/snapshot"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    monkeypatch.setattr(server_utils, "get_server_env", lambda *a: {})

    def launch(command, environment, ready, mute):
        events.append(("launch", command, ready))
        return object()

    monkeypatch.setattr(server_utils, "start_server", launch)
    monkeypatch.setattr(
        server_utils,
        "terminate_process_top_down",
        lambda process: events.append(("cleanup",)),
    )
    monkeypatch.setattr(
        server_utils,
        "_close_server_log",
        lambda process: events.append(("log_closed",)),
    )
    monkeypatch.setattr(
        runtime_module,
        "http_generate",
        lambda *a: {"text": "{}", "finish_reason": "stop", "output_tokens": 2},
    )
    runtime = cls(
        "model",
        runtime_version="test",
        enable_gpu=True,
        telemetry_options={"enabled": False},
    )
    result = runtime.generate(manifest(), output_dir=tmp_path)
    assert events[:2] == [("snapshot", "model", SHA), ("snapshot", "model", SHA)]
    assert events[-2:] == [("cleanup",), ("log_closed",)]
    assert result["execution"]["phase_status"]["cleanup"] == "complete"
    assert result["execution"]["status"] == "complete"


def test_scoring_is_outside_inference_clock(tmp_path, monkeypatch):
    import time

    from llm_optimizer.experiment.v4 import experiment as experiment_module

    clock = [100.0]
    monkeypatch.setattr(time, "perf_counter", lambda: clock[0])

    def transport(*args):
        clock[0] += 5
        return {"text": "{}", "finish_reason": "stop", "output_tokens": 2}

    def slow_score(task, completion, reference, settings):
        clock[0] += 100
        return {
            "example_id": completion.example_id,
            "metrics": {"field_accuracy": 0.0},
            "generation_error": None,
            "scorer_error": None,
            "truncated": False,
        }

    monkeypatch.setattr(experiment_module, "score", slow_score)
    runtime = VLLMRuntimeV4("model", runtime_version="test", transport=transport)
    experiment = DatasetExperimentV4(
        runtime, manifest(), output_dir=tmp_path, managed=False
    )
    result = experiment.evaluate(manifest(), {}, label="timing-test")
    assert result["metrics"]["inference_seconds"] == 5
    assert result["metrics"]["p95_latency_s"] == 5
    assert result["scoring_seconds"] == 100
    assert result["execution"]["timings_s"]["benchmark"] == 5


def test_wandb_receives_identity_quality_and_telemetry(tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    from llm_optimizer.wandb_tracking import WandbTracker

    run = MagicMock()
    tracker = WandbTracker(run)
    monkeypatch.setattr(WandbTracker, "start", lambda **kwargs: tracker)
    # Artifact SDK behavior is covered by the existing tracker tests.
    monkeypatch.setattr(tracker, "artifacts", lambda *args, **kwargs: None)
    runtime = VLLMRuntimeV4(
        "model",
        runtime_version="test",
        transport=lambda *a: {
            "text": '{"name":"secret-0"}',
            "finish_reason": "stop",
            "output_tokens": 8,
        },
    )
    experiment = DatasetExperimentV4(
        runtime,
        manifest(),
        output_dir=tmp_path,
        managed=False,
        settings=ExperimentSettings(trials=1, wandb_enabled=True),
    )
    study, _ = experiment.run(storage="sqlite:///:memory:")
    identity = run.config.update.call_args.args[0]
    assert identity["dataset"]["task"] == "extraction"
    assert identity["dataset"]["generation"]["thinking"] == "default"
    assert run.log.call_args.args[0]["benchmark/quality/field_accuracy"] == 1
    assert "server_telemetry" in study.best_trial.user_attrs["execution"]
    run.finish.assert_called_once()


def test_sampling_and_partitions_deduplicate_task_payloads():
    from llm_optimizer.experiment.v4.datasets import held_out

    first = extraction_example()
    duplicate = replace(first, example_id="duplicate")
    another = extraction_example(1)
    assert len(sample([first, duplicate, another], Sampling(count=2))) == 2
    with pytest.raises(ValueError, match="Insufficient"):
        sample([first, duplicate], Sampling(count=2))
    with pytest.raises(ValueError, match="Repeated inference"):
        manifest([first, duplicate])
    train, valid = held_out([first, duplicate, another], validation_count=1)
    assert {e.payload_json for e in train}.isdisjoint(e.payload_json for e in valid)
