"""Model grouping, W&B adapters and descriptive statistics without network access."""

import importlib.util
import json
import re
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
SPEC = importlib.util.spec_from_file_location(
    "optuna_metadata_analysis",
    Path(__file__).resolve().parents[1] / "docs/optuna_metadata_analysis.py",
)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


@pytest.mark.parametrize(
    "suffix,expected",
    [
        ("q3_4", "qwen3-4b"),
        ("q3_8", "qwen3-8b"),
        ("dsparkq3_4", "qwen3-4b dspark"),
        ("dsparkq3_8", "qwen3-8b dspark"),
        ("cmp_2", "cpm5_2"),
        ("cpm_2", "cpm5_2"),
        ("gemma4_12", "gemma4-12b"),
        ("dsparkg4_12", "gemma4-12b dspark"),
        ("q3.5_4", "qwen3.5-4b"),
        ("dsparkq3_4-q3.5_4", "qwen3-4b dspark"),
    ],
)
def test_suffix_fallback_without_model_contract(suffix, expected):
    assert (
        analysis.model_group(
            "B-vllm-c128-e2e-s11-r3-" + suffix,
            {},
        )
        == expected
    )


def test_recorded_model_takes_precedence_and_flags_mismatch():
    name = "A-sglang-c128-e2e-gp-s11-r1-q3.5_4"
    contract = {"target_model": "Qwen/Qwen3.5-0.8B"}
    assert analysis.model_group(name, contract) == "qwen3.5-0.8b"
    assert "metadata=qwen3.5-0.8b" in analysis.model_suffix_mismatch(name, contract)
    assert (
        analysis.model_suffix_mismatch(name, {"target_model": "Qwen/Qwen3.5-4B"})
        is None
    )


def test_unsuffixed_model_and_draft_fallback():
    assert (
        analysis.model_group(
            "old",
            {
                "environment": {
                    "hf_target_model_id": "google/gemma-4-12B-it",
                    "draft_model_id": "deepseek-ai/dspark_gemma4_12b_block7",
                }
            },
        )
        == "gemma4-12b dspark"
    )
    assert analysis.model_group("unknown", {}) == "unclassified"


def test_export_adapters_and_trial_deduplication(tmp_path):
    p = tmp_path / "export.json"
    p.write_text(
        json.dumps(
            {
                "project": "team/project",
                "runs": [
                    {
                        "id": "a",
                        "name": "run-q3_4",
                        "run_state": "finished",
                        "params": {"metrics": ["output_throughput"]},
                        "summary_metrics": {"_runtime": 500},
                        "trial_history": [
                            {"trial/number": 0, "trial/state": "RUNNING"},
                            {
                                "trial/number": 0,
                                "trial/state": "COMPLETE",
                                "parameters/tokens": 10,
                                "objective/output_throughput": 5,
                                "benchmark/execution.timings_s.launch_to_ready": 12,
                            },
                        ],
                    }
                ],
            }
        )
    )
    run = analysis.load_export(p, "team/project")[0]
    assert run["state"] == "finished"
    assert run["summary"]["_runtime"] == 500
    assert len(run["history"]) == 1
    assert run["history"][0]["values"] == [5]
    assert run["history"][0]["params"] == {"tokens": 10}
    assert run["history"][0]["measurement"]["execution.timings_s.launch_to_ready"] == 12


def test_objective_statistics_separate_models_objectives_and_coverage():
    base = {
        "model_group": "qwen3-4b",
        "framework": "vllm",
        "phase": "B",
        "target": "pareto_throughput",
        "concurrency": 128,
        "coverage": "all W&B COMPLETE trials",
        "run_identity": "a",
        "metrics": ["output_throughput", "p95_e2e_latency_ms"],
    }
    frame = pd.DataFrame(
        [
            dict(base, values=[10, 4]),
            dict(base, values=[14, 8], coverage="all local COMPLETE trials"),
            dict(base, values=[20, 10], framework="sglang", run_identity="b"),
            dict(base, values=[1000, 1000], coverage="W&B candidate summary only"),
            dict(base, values=[999, 999], model_group="qwen3-4b dspark"),
        ]
    )
    result = analysis.objective_statistics(frame)
    throughput = result[
        (result.model_group == "qwen3-4b")
        & (result.metric == "output_throughput")
        & (result.coverage == "full trial history")
        & (result.scope == "model")
    ]
    vllm = throughput[throughput.framework == "vllm"].iloc[0]
    assert (vllm["mean"], vllm["variance"], vllm["n"]) == (12, 8, 2)
    overall = throughput[throughput.framework == "overall"].iloc[0]
    assert overall["mean"] == pytest.approx(44 / 3)
    assert overall["runs"] == 2
    singleton = throughput[throughput.framework == "sglang"].iloc[0]
    assert pd.isna(singleton["variance"])


def test_startup_does_not_sum_overlapping_loading_observations_or_partial_times():
    run = {
        "model_group": "gemma4-12b",
        "framework": "vllm",
        "phase": "B",
        "study_name": "s",
        "wandb_id": "id",
    }
    trial = {
        "number": 1,
        "state": "COMPLETE",
        "params": {"tokens": 32},
        "execution": {
            "timings_s": {"launch_to_ready": 100, "warmup": 2},
            "phase_status": {"launch_to_ready": "complete", "warmup": "partial"},
            "startup_observations": [
                {"phase": "weight_loading", "seconds": 30, "source": "rank0"},
                {"phase": "weight_loading", "seconds": 40, "source": "rank1"},
            ],
        },
    }
    rows = analysis.startup_rows(run, [trial], "test")
    detail = analysis.startup_trial_table(pd.DataFrame(rows))
    assert len(detail) == 1
    assert detail.iloc[0]["params"] == {"tokens": 32}
    assert detail.iloc[0]["weight_loading (s)"] == [30, 40]
    result = analysis.startup_statistics(pd.DataFrame(rows))
    assert "warmup" not in set(result.timing)
    launch = result[
        (result.timing == "launch_to_ready")
        & (result.framework == "overall")
        & (result.scope == "model")
    ]
    assert launch.iloc[0]["mean_seconds"] == 100
    loading = result[
        (result.timing == "weight_loading")
        & (result.framework == "overall")
        & (result.scope == "model")
    ]
    assert loading.iloc[0]["mean_seconds"] == 35
    assert loading.iloc[0]["variance_seconds2"] == 50


def test_parameter_table_escapes_data_and_keeps_line_breaks():
    rendered = analysis.table_html(
        pd.DataFrame(
            [
                {
                    "run": "<script>bad</script>",
                    "params": {"<key>": "<value>", "other": 5},
                }
            ]
        )
    )
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "&lt;key&gt;" in rendered
    assert "&lt;value&gt;" in rendered
    assert "<br>" in rendered


def test_html_report_has_all_requested_models_and_handles_missing_data(tmp_path):
    path = tmp_path / "report.html"
    empty = pd.DataFrame()
    analysis.write_report_html(
        path,
        analysis.MODEL_ORDER,
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
        exported_at="2026-11-03",
    )
    rendered = path.read_text()
    for model in analysis.MODEL_ORDER:
        assert f"<h2>{model}</h2>" in rendered
    assert "No data available" in rendered
    assert "ddof=1" in rendered


def test_optuna_importance_uses_both_pareto_targets_and_excludes_failed_trials():
    optuna = pytest.importorskip("optuna")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    run = {
        "model_group": "qwen3-4b",
        "framework": "vllm",
        "study_name": "test",
        "contract": {
            "metrics": ["output_throughput", "p95_e2e_latency_ms"],
            "directions": ["MAXIMIZE", "MINIMIZE"],
            "sampling_menu": {"tokens": [1, 2, 3]},
        },
    }
    trials = [
        {
            "state": "COMPLETE",
            "params": {"tokens": i % 3 + 1, "cuda_graph_on": i % 2},
            "values": [10 + i, 100 - i],
        }
        for i in range(20)
    ]
    trials.append({"state": "FAIL", "params": {}, "values": [None, None]})
    scores, reason = analysis.history_importance(run, trials)
    assert reason is None
    assert {r["metric"] for r in scores} == set(run["contract"]["metrics"])
    assert {r["parameter"] for r in scores} == {"tokens"}
    assert all(0 <= r["importance"] <= 1 for r in scores)


def test_v2_importance_filters_phase_and_excluded_parameter(tmp_path):
    optuna = pytest.importorskip("optuna")
    import math

    path = Path(__file__).resolve().parents[1] / "docs/optuna_run_metadata_v2.ipynb"
    notebook = json.loads(path.read_text())
    cells = ["".join(cell["source"]) for cell in notebook["cells"]]
    namespace = {
        "pd": pd,
        "math": math,
        "re": re,
        "ROOT": tmp_path,
        "history_importance": analysis.history_importance,
        "importance_coverage": [],
        "result_issues": [],
    }
    importance_cell = next(
        c for c in cells if c.startswith("def parameter_importance(")
    )
    exec(importance_cell, namespace)
    exec(next(cell for cell in cells if cell.startswith("def run_tokens(")), namespace)
    assert namespace["run_tokens"]("B-vllm-s11-r1-pruner-median") == ("s11", "r1")
    assert namespace["run_tokens"]("B-vllm-s11-r10") == ("s11", "r10")
    study = optuna.create_study(
        study_name="test", storage=f"sqlite:///{tmp_path / 'test.db'}"
    )
    for i in range(20):
        study.add_trial(
            optuna.trial.create_trial(
                params={"tokens": i % 3 + 1, "cuda_graph_on": i % 2},
                distributions={
                    "tokens": optuna.distributions.IntDistribution(1, 3),
                    "cuda_graph_on": optuna.distributions.IntDistribution(0, 1),
                },
                value=float(i),
            )
        )
    base = {
        "database": "test.db",
        "study_name": "test",
        "model_group": "qwen3-4b",
        "framework": "vllm",
        "wandb_id": "id",
        "contract": {"metrics": ["metric"], "directions": ["MINIMIZE"]},
    }
    result = namespace["parameter_importance"](
        [dict(base, phase="A"), dict(base, phase="B")]
    )
    assert list(result["parameter"]) == ["tokens"]
    assert result.iloc[0]["importance"] == pytest.approx(1)
