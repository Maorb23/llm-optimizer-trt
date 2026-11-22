"""V4 notebook parity and managed v3 campaign integration."""

import asyncio
import json
import textwrap
from pathlib import Path
from types import SimpleNamespace

import optuna
import pytest

from llm_optimizer.experiment.v4 import campaign
from llm_optimizer.experiment.v4.runtime import Completion

ROOT = Path(__file__).resolve().parents[1]
ENGINES = ("vllm", "trtllm", "sglang")


def cells(engine, version):
    if version == 3:
        name = f"phase1_experiments_{engine}_v3.ipynb"
    else:
        suffix = "_" if engine == "vllm" else ""
        name = f"phase1_{engine}{suffix}_datasets_v4.ipynb"
    book = json.loads((ROOT / "docs" / name).read_text())
    if version == 4:
        import nbformat

        nbformat.validate(nbformat.from_dict(book))
    return [
        "".join(cell["source"]) for cell in book["cells"] if cell["cell_type"] == "code"
    ]


@pytest.mark.parametrize("engine", ENGINES)
def test_notebook_search_and_study_cells_preserved(engine):
    old, new = cells(engine, 3), cells(engine, 4)
    for source in old:
        if source.startswith(
            (
                "PREFILL_BUDGETS",
                "# Enable only",
                "PRECISION_CHOICES",
                "# Independent Phase B axes",
                "# Phase B uses",
            )
        ):
            assert source in new
        if " = run_objective_group(" in source or source.startswith(
            ("diagnostics =", "pruning_comparison =")
        ):
            assert any(
                cell.startswith("if RUN_REAL_GPU and RUN_STUDIES:\n")
                and textwrap.dedent(cell.split("\n", 1)[1]).strip() == source.strip()
                for cell in new
            )
    assert any(
        "validate_candidates(runner.lab.shortlist(studies))" in cell for cell in new
    )


def prepare_notebook(engine, tmp_path):
    namespace = {"display": lambda *args: None}
    for source in cells(engine, 4):
        exec(compile(source, engine, "exec"), namespace)
        if source.startswith("CAMPAIGN ="):
            namespace.update(
                RUN_ROOT=tmp_path,
                REQUESTS_BY_CONCURRENCY={1: 2, 8: 8, 16: 8, 32: 8, 128: 8},
                PRUNING_MIN_COMPLETED_REQUESTS=dict.fromkeys((1, 8, 16, 32, 128), 1),
                DATASET_REPEAT_REQUESTS=8,
                HELD_OUT_COUNT=3,
                TRIALS=2,
                SEEDS=[11],
                SAMPLER_NAMES=["tpe"],
                QUALITY_THRESHOLDS=(("field_accuracy", 1),),
            )
    return namespace


@pytest.mark.parametrize("engine", ENGINES)
def test_managed_v3_study_and_held_out_with_mock_inference(
    engine, tmp_path, monkeypatch
):
    import importlib

    from llm_optimizer import bench_client

    env = prepare_notebook(engine, tmp_path)
    if engine == "trtllm":
        env["PRECISION_CHOICES"][engine]["weight_precision"] = ["bf16"]
    runner = env["get_runner"]()
    lab = runner.lab
    policy = importlib.import_module(f"llm_optimizer.experiment.{engine}_policy_v3")
    runtime = importlib.import_module(f"llm_optimizer.experiment.runtime_{engine}_v3")
    telemetry_module = importlib.import_module(
        f"llm_optimizer.experiment.telemetry_{engine}_v3"
    )
    collectors = []
    start_collection = telemetry_module.Collector.start

    def track_telemetry(collector):
        collectors.append(collector)
        start_collection(collector)

    monkeypatch.setattr(telemetry_module.Collector, "_fetch", lambda _: "")
    monkeypatch.setattr(telemetry_module.Collector, "start", track_telemetry)
    monkeypatch.setattr(policy, "verify_runtime", lambda: None)
    if engine == "trtllm":
        monkeypatch.setattr(policy, "resolve_cli", lambda: "trtllm-serve")
        # Match the notebook's CPU mapping in the absence of TRT schema packages.
        original = lab.serialize_engine_config

        def serialize(*args):
            lab.RUN_REAL_GPU = False
            try:
                return original(*args)
            finally:
                lab.RUN_REAL_GPU = True

        lab.serialize_engine_config = serialize
    tokenizer = SimpleNamespace(encode=lambda text, **kw: list(text))

    def reject_legacy_loader(*args, **kwargs):
        pytest.fail("Frozen task campaign entered the legacy loader")

    monkeypatch.setattr(bench_client, "get_dataset", reject_legacy_loader)
    monkeypatch.setattr(bench_client, "get_tokenizer", lambda _: tokenizer)
    monkeypatch.setattr(bench_client, "check_chat_template", lambda _: True)
    monkeypatch.setattr(bench_client, "set_ulimit", lambda: None)
    monkeypatch.setattr(
        bench_client,
        "time",
        SimpleNamespace(
            perf_counter=bench_client.time.perf_counter,
            sleep=lambda _: None,
        ),
    )
    running = False
    active = max_active = calls = 0
    generated_ids = set()
    reject = True

    def start(*args):
        nonlocal running
        assert not running
        running = True
        return object()

    def stop(*args):
        nonlocal running
        assert running and active == 0
        running = False

    async def infer(item, example_id, payload, pbar):
        nonlocal active, max_active, calls
        assert running and not any(
            key in payload for key in ("schema", "expected", "test")
        )
        assert payload["max_tokens"] in {32, 256}
        active += 1
        max_active = max(active, max_active)
        calls += 1
        generated_ids.add(example_id)
        await asyncio.sleep(0.001)
        active -= 1
        value = int(
            payload["messages"][-1]["content"].rsplit("id=", 1)[1].splitlines()[0]
        )
        text = json.dumps({"id": value + int(reject)})
        out = bench_client.RequestFuncOutput(
            success=True,
            generated_text=text,
            latency=0.003,
            ttft=0.001,
            output_len=8,
            prompt_len=item.prompt_len,
        )
        return out, Completion(
            example_id, text=text, finish_reason="stop", output_tokens=8
        )

    native_score = campaign.score
    score_calls = 0
    extra_score_time = 0
    clock = campaign.time.perf_counter
    monkeypatch.setattr(
        campaign,
        "time",
        SimpleNamespace(
            perf_counter=lambda: clock() + extra_score_time,
        ),
    )

    def score(*args):
        nonlocal score_calls, reject, extra_score_time
        assert not running and active == 0
        assert collectors and all(not collector.active for collector in collectors)
        score_calls += 1
        extra_score_time += 10
        result = native_score(*args)
        if score_calls == 8:
            reject = False
        return result

    monkeypatch.setattr(runtime, "start_server", start)
    monkeypatch.setattr(runtime, "terminate_process_top_down", stop)
    monkeypatch.setattr(runtime, "_close_server_log", lambda _: None)
    monkeypatch.setattr(runtime, "get_server_env", lambda *args: {})
    monkeypatch.setattr(campaign, "chat_request", infer)
    monkeypatch.setattr(campaign, "score", score)
    lab.telemetry_options.update(enabled=True, gpu_index=None)
    lab.RUN_REAL_GPU = True
    lab.REST_SECONDS = 0
    lab.MODEL_SNAPSHOTS["A"] = str(tmp_path)
    lab.REAL_PREFLIGHT["A", engine]["native_config_verified"] = True

    (study,) = runner.run_objective_group("A", "throughput", [8]).values()
    assert [trial.state for trial in study.trials] == [
        optuna.trial.TrialState.PRUNED,
        optuna.trial.TrialState.COMPLETE,
    ]
    assert [
        t.user_attrs["dataset_v4"]["quality"]["field_accuracy"] for t in study.trials
    ] == [0, 1]
    assert max_active == 8 and calls == 18 and score_calls == 16
    for trial in study.trials:
        assert trial.user_attrs["dataset_v4"]["scoring_seconds"] >= 80
        assert trial.user_attrs["measurement"]["wall_seconds"] < 10
        assert 0 < trial.user_attrs["server_telemetry"]["measured_window_s"] < 10
    (resumed,) = runner.run_objective_group("A", "throughput", [8]).values()
    assert len(resumed.trials) == 2 and calls == 18
    tuning_ids = generated_ids.copy()
    generated_ids.clear()
    result = campaign.evaluate_held_out(runner, study)
    assert result["trial"] == study.best_trial.number
    assert result["contract"]["concurrency"] == 8
    assert result["measurement"]["total_requests"] == 3
    assert not tuning_ids & generated_ids
    assert len(study.trials) == 2
    assert calls == 22 and score_calls == 19
    candidates = lab.shortlist([study])
    repeats = lab.validate_candidates(candidates, repeats=2)
    assert repeats[0]["successful_repeats"] == 2
    assert repeats[0]["num_prompts_per_repeat"] == 8
    assert not (
        {
            e["example_id"]
            for e in runner.dataset_workloads.validation["A"].document["examples"]
        }
        & {
            r["example_id"]
            for manifest in runner.dataset_workloads.repeats.values()
            for r in manifest.document["examples"]
        }
    )

    lab.MODEL_SNAPSHOTS["B"] = str(tmp_path)
    lab.REAL_PREFLIGHT["B", engine].update(
        native_config_verified=True,
        quality_verified=True,
    )
    (phase_b,) = runner.run_objective_group("B", "throughput", [8], "median").values()
    contract = phase_b.user_attrs["contract"]
    assert contract["space_kind"] == "expanded"
    assert contract["pruner_class"] == "MedianPruner"
    assert contract["checkpoint_requests"][-1] == 8
    assert contract["sampling_menu"] == lab.sampling_menu(engine, 8, "expanded")
    assert all(trial.params and trial.intermediate_values for trial in phase_b.trials)


def test_legacy_source_hash_and_separate_campaign_identity(tmp_path):
    env = prepare_notebook("vllm", tmp_path)
    task_runner = env["get_runner"]()
    env["DATA_SOURCE"] = "legacy_random"
    env["QUALITY_THRESHOLDS"] = ()
    legacy_runner = env["get_runner"]()
    assert legacy_runner.run_dir != task_runner.run_dir
    workload = legacy_runner.lab.workload_for(8)
    assert workload["dataset_name"] == "random"
    assert (workload["random_input_len"], workload["random_output_len"]) == (512, 128)
    assert "disable_ignore_eos" not in workload
    assert not hasattr(legacy_runner, "held_out_benchmark")
    source = tmp_path / "legacy.json"
    source.write_text("[]")
    _, first = campaign.prepare_legacy(source, None)
    source.write_text("[{}]")
    _, second = campaign.prepare_legacy(source, None)
    assert first["source_sha256"] != second["source_sha256"]


def test_managed_study_identity_follows_frozen_controls(tmp_path):
    from dataclasses import replace

    env = prepare_notebook("vllm", tmp_path)
    prep = next(cell for cell in cells("vllm", 4) if "def dataset_controls" in cell)
    first = env["get_runner"]().run_dir
    env["GENERATION"] = replace(env["GENERATION"], output_tokens=512, seed=37)
    with pytest.raises(ValueError, match="Dataset settings changed"):
        env["get_runner"]()
    exec(compile(prep, "prep", "exec"), env)
    second = env["get_runner"]().run_dir
    assert first != second
    env["QUALITY_THRESHOLDS"] = ()
    exec(compile(prep, "prep", "exec"), env)
    assert env["get_runner"]().run_dir not in {first, second}
