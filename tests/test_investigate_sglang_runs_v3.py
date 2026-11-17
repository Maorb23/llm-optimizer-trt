"""Offline tests for the read-only W&B comparison report."""

import importlib.util
import json
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / "tools/investigate_sglang_runs_v3.py"
spec = importlib.util.spec_from_file_location("investigate_sglang_v3", MODULE)
investigate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(investigate)


def test_history_does_not_fill_parameters_between_trials():
    trials = investigate.history_trials(
        [
            {"trial/number": 0, "parameters/attention_backend": "triton"},
            {
                "trial/number": 0,
                "trial/state": "COMPLETE",
                "objective/p95_e2e_latency_ms": 10,
            },
            {"trial/number": 1, "trial/state": "PRUNED"},
            {"unrelated": 42},
        ]
    )
    assert len(trials) == 2
    assert investigate.metric(trials[0], investigate.E2E) == 10
    assert investigate.params(trials[1]) == {}


def test_compare_separates_configuration_changes_from_repeat_variation(
    tmp_path, capsys
):
    config = {
        "target": "e2e",
        "metrics": [investigate.E2E],
        "concurrency": 16,
        "sampler": "gp",
        "sampler_settings": {"seed": 11},
        "environment": {"requests_sha256": "same", "framework_version": "0.5.23"},
        "workload": {"seed": 2026, "random_input_len": 512},
    }

    def row(number, e2e, capacity, state="COMPLETE"):
        return {
            "trial/number": number,
            "trial/state": state,
            "benchmark/p95_e2e_latency_ms": e2e,
            "trial/config": {"max_running_requests": capacity},
            "parameters/max_running_requests": capacity,
        }

    runs = [
        {
            "id": "a",
            "name": "seed11",
            "path": "e/p/a",
            "state": "finished",
            "config": config,
            "summary": {},
            "trials": [row(0, 5000, 16), row(1, 100, 8, "PRUNED")],
        },
        {
            "id": "b",
            "name": "seed29",
            "path": "e/p/b",
            "state": "finished",
            "config": {**config, "sampler_settings": {"seed": 29}},
            "summary": {},
            "trials": [row(0, 5200, 16), row(1, 2600, 32)],
        },
        {
            "id": "c",
            "name": "seed47",
            "path": "e/p/c",
            "state": "finished",
            "config": {**config, "workload": {**config["workload"], "seed": 999}},
            "summary": {},
            "trials": [row(0, 5500, 16)],
        },
    ]
    investigate.compare(runs, tmp_path)
    report = json.loads((tmp_path / "comparison.json").read_text())
    assert report["summaries"][0]["best_e2e"] == 5000
    assert report["summaries"][1]["best_e2e"] == 2600
    assert len(report["matched_configurations"]) == 1
    assert report["matched_configurations"][0]["observations"] == [
        ["a", 0, 5000.0],
        ["b", 0, 5200.0],
        ["c", 0, 5500.0],
    ]
    assert [d["field"] for d in report["contract_differences"]] == ["workload/seed"]
    text = capsys.readouterr().out
    assert "Slowest/fastest ratio: 1.100" in text
    assert (tmp_path / "trials.csv").is_file()


def test_constrained_and_missing_data_are_not_fabricated():
    run = {
        "config": {"constrained": True},
        "trials": [
            {
                "trial/state": "COMPLETE",
                "benchmark/p95_e2e_latency_ms": 10,
                "trial/slo_feasible": False,
            },
            {
                "trial/state": "COMPLETE",
                "benchmark/p95_e2e_latency_ms": 20,
                "trial/slo_feasible": True,
            },
        ],
    }
    assert len(investigate.eligible(run)) == 1
    assert investigate.metric({}, investigate.E2E) is None
    assert investigate.native_config({}) is None
    assert investigate.clean({"api_key": "do-not-export", "score": float("inf")}) == {
        "score": None
    }
