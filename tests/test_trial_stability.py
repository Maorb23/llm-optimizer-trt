"""Historical stability comparisons keep workload and native settings explicit."""

import json
import sqlite3
from pathlib import Path

import pytest

from llm_optimizer.experiment.stability import (
    eligible_records,
    exact_repeat_summary,
    load_measurements,
    parameter_differences,
    similar_parameter_pairs,
)


def make_run(root, rows, phase="B", framework="vllm", native=None):
    root.mkdir(parents=True, exist_ok=True)
    study = f"{phase}-{framework}-c128-throughput-r1"
    with sqlite3.connect(root / "studies.sqlite3") as db:
        db.executescript("""
            CREATE TABLE studies(study_id INTEGER, study_name TEXT);
            CREATE TABLE study_user_attributes(study_id INTEGER,
                                               key TEXT, value_json TEXT);
            CREATE TABLE trials(trial_id INTEGER, study_id INTEGER, number INTEGER,
                                state TEXT, datetime_start TEXT,
                                datetime_complete TEXT);
            CREATE TABLE trial_user_attributes(trial_id INTEGER,
                                               key TEXT, value_json TEXT);
        """)
        db.execute("INSERT INTO studies VALUES(1, ?)", (study,))
        contract = {
            "phase": phase,
            "framework": framework,
            "concurrency": 128,
            "sampler": "tpe",
            "target": "throughput",
        }
        db.execute(
            'INSERT INTO study_user_attributes VALUES(1, "contract", ?)',
            (json.dumps(contract),),
        )
        logs = []
        for number, item in enumerate(rows):
            value, count = item.get("value", 100), item.get("count", 10)
            metrics = {
                "output_throughput": value,
                "p95_ttft_ms": value / 2,
                "p95_e2e_latency_ms": value * 2,
                "completed": count,
                "total_requests": count,
                "failed": 0,
                "benchmark_valid": True,
            }
            metrics.update(item.get("metrics", {}))
            state = item.get("state", "COMPLETE")
            db.execute(
                "INSERT INTO trials VALUES(?, 1, ?, ?, ?, ?)",
                (
                    number,
                    number,
                    state,
                    f"2026-01-01T00:{number:02d}:00",
                    f"2026-01-01T00:{number:02d}:01",
                ),
            )
            db.execute(
                'INSERT INTO trial_user_attributes VALUES(?, "measurement", ?)',
                (number, json.dumps(metrics)),
            )
            metadata = {
                "phase": phase,
                "target_model": "model",
                "model": "/models/test",
                "model_revision": "pinned",
                "framework_version": "1",
                "cuda_version": "12",
                "driver_version": "1",
                "gpu_type": "test-gpu",
                "gpu_count": 1,
                "gpu_uuid": "GPU-1",
                "requests_sha256": "requests",
                "workload_sha256": "workload",
            }
            metadata.update(item.get("metadata", {}))
            args = dict(native or {"max_num_seqs": 32, "gpu_memory_utilization": 0.8})
            args.update(item.get("server", {}))
            logs.append(
                {
                    "config": {
                        "framework": framework,
                        "server_args": args,
                        "client_args": {
                            "max_concurrency": 128,
                            "num_prompts": count,
                            "seed": item.get("seed", 2026),
                            "request_rate": None,
                        },
                    },
                    "metadata": metadata,
                    "results": metrics,
                    "cmd": "server",
                }
            )
    path = root / f"{study}-{phase}-{framework}-c128-measured.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in logs))
    return path


def test_exact_ranges_join_trials_without_double_counting(tmp_path):
    source = make_run(tmp_path / "run", [{"value": 100}, {"value": 110}, {"value": 90}])
    original_db = (source.parent / "studies.sqlite3").read_bytes()
    original_log = source.read_bytes()
    audit = load_measurements([source.parent, source.parent])
    selected, excluded = eligible_records(audit["records"])
    assert len(selected) == len(audit["trials"]) == 3
    assert not excluded and not audit["issues"]
    assert not selected[0]["identity_warnings"]
    report = exact_repeat_summary(selected)
    throughput = next(
        r for r in report["metrics"] if r["metric"] == "output_throughput"
    )
    assert throughput["min"] == 90 and throughput["max"] == 110
    assert throughput["median"] == 100 and throughput["n"] == 3
    assert throughput["spread_pct"] == pytest.approx(20)
    assert throughput["cv_pct"] == pytest.approx(10)
    assert throughput["max_abs_deviation_pct"] == pytest.approx(10)
    assert [r["trial_number"] for r in selected] == [0, 1, 2]
    assert source.read_bytes() == original_log
    assert (source.parent / "studies.sqlite3").read_bytes() == original_db


def test_workload_environment_and_phase_cannot_be_pooled(tmp_path):
    make_run(
        tmp_path / "b",
        [
            {"value": 100},
            {"value": 110},
            {"value": 200, "seed": 29},
            {"value": 300, "count": 20},
            {"value": 400, "metadata": {"driver_version": "2"}},
        ],
    )
    make_run(tmp_path / "a", [{"value": 100}], phase="A")
    make_run(tmp_path / "sg", [{"value": 100}], framework="sglang")
    selected, _ = eligible_records(
        load_measurements([tmp_path / "b", tmp_path / "a", tmp_path / "sg"])["records"]
    )
    report = exact_repeat_summary(selected)
    assert len(report["groups"]) == 6
    assert len(report["metrics"]) == 3
    assert all(
        row["n"] == 2 and row["phase"] == "B" and row["framework"] == "vllm"
        for row in report["metrics"]
    )


def test_incomplete_failed_and_duplicate_records_are_excluded(tmp_path):
    source = make_run(
        tmp_path / "run",
        [
            {"value": 100},
            {
                "value": 10000,
                "metrics": {"failed": 1, "completed": 9, "benchmark_valid": False},
            },
            {"value": 120, "state": "FAIL"},
        ],
    )
    with source.open("a") as stream:
        stream.write(source.read_text().splitlines()[0] + "\n")
    audit = load_measurements([source.parent])
    selected, excluded = eligible_records(audit["records"])
    assert len(selected) == 1 and len(excluded) == 3
    assert any("already counted" in row["exclusion_reason"] for row in excluded)
    assert not exact_repeat_summary(selected)["metrics"]


def test_trt_yaml_contents_resolve_after_moving_logs(tmp_path):
    root = tmp_path / "trt"
    source = make_run(
        root,
        [{"value": 100}, {"value": 105}],
        framework="trtllm",
        native={"max_batch_size": 32, "config": "/old/container/config-one.yaml"},
    )
    (root / "config-one.yaml").write_text(
        "kv_cache_config:\n  free_gpu_memory_fraction: 0.8\n"
        "cuda_graph_config:\n  batch_sizes: [1, 2, 4]\n"
    )
    lines = [json.loads(line) for line in source.read_text().splitlines()]
    lines[1]["config"]["server_args"]["config"] = "/different/path/config-two.yaml"
    source.write_text("".join(json.dumps(row) + "\n" for row in lines))
    (root / "config-two.yaml").write_text((root / "config-one.yaml").read_text())
    selected, excluded = eligible_records(load_measurements([root])["records"])
    assert not excluded
    assert selected[0]["config_id"] == selected[1]["config_id"]
    assert selected[0]["server_config"]["config"]["cuda_graph_config"][
        "batch_sizes"
    ] == [1, 2, 4]
    assert len(exact_repeat_summary(selected)["groups"]) == 1
    (root / "config-two.yaml").unlink()
    selected, excluded = eligible_records(load_measurements([root])["records"])
    assert len(selected) == 1
    assert excluded[0]["exclusion_reason"].startswith("Native config")


def test_numeric_neighbors_reject_graph_boolean_and_missing_flags():
    a = {
        "max_batch_size": 24,
        "config": {
            "kv_cache_config": {"free_gpu_memory_fraction": 0.8},
            "graph": {"sizes": [1, 2, 4]},
        },
        "enabled": True,
    }
    b = {**a, "max_batch_size": 32}
    assert parameter_differences(a, b)[0]["parameter"] == "max_batch_size"
    assert parameter_differences(a, {**b, "enabled": False}) is None
    assert parameter_differences(a, {**b, "max_batch_size": 64}) is None
    assert (
        parameter_differences(a, {k: v for k, v in a.items() if k != "enabled"}) is None
    )
    assert (
        parameter_differences(
            a,
            {
                **a,
                "config": {
                    "kv_cache_config": {"free_gpu_memory_fraction": 0.8},
                    "graph": {"sizes": [1, 2]},
                },
            },
        )
        is None
    )
    assert parameter_differences(a, a) is None
    nested = {
        **a,
        "config": {
            **a["config"],
            "kv_cache_config": {"free_gpu_memory_fraction": 0.85},
        },
    }
    assert (
        parameter_differences(a, nested)[0]["parameter"]
        == "config.kv_cache_config.free_gpu_memory_fraction"
    )
    with pytest.raises(ValueError, match="tolerances"):
        parameter_differences(a, b, tolerances={"max_batch_size": -1})


def test_neighbor_pairs_report_effects_separately_from_exact_repeats(tmp_path):
    make_run(
        tmp_path / "run",
        [
            {"value": 100},
            {"value": 110},
            {"value": 200, "server": {"max_num_seqs": 40}},
        ],
    )
    rows, _ = eligible_records(load_measurements([tmp_path / "run"])["records"])
    pairs = similar_parameter_pairs(rows)
    assert len(pairs) == 3
    throughput = next(r for r in pairs if r["metric"] == "output_throughput")
    assert throughput["left_n"] == 2 and throughput["right_n"] == 1
    assert throughput["left_median"] == 105 and throughput["right_median"] == 200
    assert throughput["median_gap_pct"] == pytest.approx(100 * 95 / 152.5)
    assert throughput["parameter_changes"][0]["left"] == 32


def test_source_errors_and_missing_sqlite_are_visible(tmp_path):
    source = make_run(tmp_path / "run", [{"value": 100}])
    source.write_text(source.read_text() + "{broken\n")
    (source.parent / "studies.sqlite3").unlink()
    audit = load_measurements([source.parent])
    assert len(audit["issues"]) == 2
    assert len(eligible_records(audit["records"])[0]) == 0
    assert len(eligible_records(audit["records"], include_unmatched=True)[0]) == 1
    assert not (source.parent / "studies.sqlite3").exists()


def test_notebook_has_distinct_phase_sections_and_no_gpu_launches():
    notebook = json.loads(
        (
            Path(__file__).resolve().parents[1] / "docs/optuna_trial_stability.ipynb"
        ).read_text()
    )
    sources = [
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"
    ]
    assert 'show_phase("A")\n' in sources and 'show_phase("B")\n' in sources
    assert not any("experiment run" in s or "register_magic" in s for s in sources)
    for source in sources:
        compile(source, "stability-notebook", "exec")
