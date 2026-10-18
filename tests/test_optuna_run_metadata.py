"""Duration selection and notebook integrity, without W&B or GPU access."""

import ast
import json
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
NOTEBOOK = Path(__file__).resolve().parents[1] / "docs/optuna_run_metadata.ipynb"


def helpers():
    notebook = json.loads(NOTEBOOK.read_text())
    namespace = {
        "pd": pd,
        "math": __import__("math"),
        "json": json,
        "re": __import__("re"),
    }
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            tree = ast.parse("".join(cell["source"]))
            functions = [
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef)
                and node.name
                in {
                    "seconds",
                    "duration_for",
                    "exclusion",
                    "frontier_indices",
                    "decode_param",
                    "report_runs",
                    "plot_framework_coverage",
                    "early_vllm_latency_runs",
                    "normalize_wandb_summary",
                    "read_wandb_snapshot",
                    "run_tokens",
                }
            ]
            exec(
                compile(
                    ast.Module(body=functions, type_ignores=[]), str(NOTEBOOK), "exec"
                ),
                namespace,
            )
    return namespace


def test_duration_precedence_and_nested_wandb_summary():
    duration = helpers()["duration_for"]
    assert duration(
        {"optimization_wall_seconds": 300},
        [],
        {"study": {"optimization_wall_seconds": 400}},
    ) == (300, "local optimization_wall_seconds")
    assert duration({}, [], {"study": {"optimization_wall_seconds": 400}}) == (
        400,
        "W&B optimization_wall_seconds",
    )


def test_trial_span_requires_terminal_trials():
    duration = helpers()["duration_for"]
    trial = {
        "state": "COMPLETE",
        "datetime_start": "2026-01-01 00:00:00",
        "datetime_complete": "2026-01-01 00:05:00",
    }
    assert duration({}, [trial]) == (300, "local terminal trial span")
    assert duration({}, [dict(trial, state="RUNNING")]) == (None, "unknown")
    assert duration({}, [], {"_runtime": 600}, "running") == (None, "unknown")
    assert duration({}, [], {"_runtime": 600}, "finished") == (
        600,
        "W&B terminal _runtime",
    )


@pytest.mark.parametrize(
    "duration,reason",
    [(299.99, "under five minutes"), (300, None), (None, "unknown duration")],
)
def test_five_minute_boundary(duration, reason):
    namespace = helpers()
    namespace.update(MIN_RUN_SECONDS=300, INCLUDE_SYNTHETIC=False)
    record = {"framework": "vllm", "contract": {}, "duration_seconds": duration}
    assert namespace["exclusion"](record) == reason
    record["contract"] = {"environment": {"measurement": "SYNTHETIC"}}
    assert namespace["exclusion"](record) == "synthetic"


def test_mixed_direction_frontier_preserves_ties():
    frontier = helpers()["frontier_indices"]
    assert frontier([[10, 5], [9, 6], [12, 8], [10, 5]], ["MAXIMIZE", "MINIMIZE"]) == [
        0,
        2,
        3,
    ]
    assert frontier([[2], [1], [1], [3]], ["MINIMIZE"]) == [1, 2]
    assert frontier([], ["MAXIMIZE", "MINIMIZE"]) == []


def test_trial_parameters_decode_categorical_indices():
    decode = helpers()["decode_param"]
    distribution = json.dumps(
        {"name": "CategoricalDistribution", "attributes": {"choices": ["auto", "fp8"]}}
    )
    assert decode(1.0, distribution) == "fp8"
    assert decode(72.0, json.dumps({"name": "IntDistribution"})) == 72


def test_report_runs_isolates_phase_framework_concurrency_objective():
    namespace = helpers()

    def run(phase, framework, concurrency, target):
        return {
            "phase": phase,
            "framework": framework,
            "contract": {"concurrency": concurrency, "target": target},
        }

    expected = run("B", "vllm", 128, "pareto_throughput")
    namespace["included"] = [
        expected,
        run("A", "vllm", 128, "pareto_throughput"),
        run("B", "sglang", 128, "pareto_throughput"),
        run("B", "vllm", 1, "e2e"),
        run("B", "vllm", 128, "throughput"),
    ]
    assert namespace["report_runs"]("B", "vllm", 128, "pareto_throughput") == [expected]
    assert len(namespace["report_runs"]("B", "vllm")) == 3


def test_notebook_has_separate_phase_framework_sections():
    notebook = json.loads(NOTEBOOK.read_text())
    calls = [
        "".join(c["source"]).strip()
        for c in notebook["cells"]
        if c["cell_type"] == "code"
        and "".join(c["source"]).startswith("show_framework(")
    ]
    assert calls == [
        f'show_framework("{phase}", "{framework}")'
        for phase in ["B", "A"]
        for framework in ["vllm", "trtllm", "sglang"]
    ]


def test_run_tokens_distinguishes_seed_and_repeat():
    parse = helpers()["run_tokens"]
    assert parse("B-vllm-c128-throughput-gp-s22-r15") == ("s22", "r15")
    assert parse("B-vllm-c128-throughput-gp-s11-r22") == ("s11", "r22")
    assert parse("unnamed") == ("seed unknown", "repeat unknown")


def test_early_latency_inventory_keeps_original_run_identity():
    namespace = helpers()
    rows = [
        {
            "phase": "A",
            "framework": "vllm",
            "study_name": "A-vllm-c128-e2e-gp-s11-r1",
            "wandb_id": identity,
            "contract": {"concurrency": 128, "target": "e2e"},
        }
        for identity in ["early", "later"]
    ]
    rows.append(
        {
            "phase": "A",
            "framework": "vllm",
            "study_name": "A-vllm-c128-e2e-gp-s11-r2",
            "wandb_id": "r2",
            "contract": {"concurrency": 128, "target": "e2e"},
        }
    )
    namespace["included"] = rows
    assert [r["wandb_id"] for r in namespace["early_vllm_latency_runs"]()] == [
        "early",
        "later",
    ]


def test_cached_wandb_results_include_phase_b_e2e_without_network():
    from types import SimpleNamespace

    namespace = helpers()
    namespace["SimpleNamespace"] = SimpleNamespace
    path = NOTEBOOK.parents[1] / "experiment_results/wandb_run_snapshot.json"
    runs, exported_at = namespace["read_wandb_snapshot"](
        path, "maorblumberg-tel-aviv-university/inference-optimization"
    )
    run = next(r for r in runs if r.id == "e31bigam")
    assert run.config["study_name"] == "B-vllm-c128-e2e-tpe_multivariate-s11-r3"
    assert run.config["phase"] == "B"
    assert exported_at
    duration, source = namespace["duration_for"]({}, [], run.summary, run.state)
    assert duration >= 300
    assert run.summary["best_or_pareto_trials"]
    assert (
        namespace["normalize_wandb_summary"]({"study.states.COMPLETE": 19})["study"][
            "states"
        ]["COMPLETE"]
        == 19
    )


def test_notebook_uses_cached_wandb_runs_by_default():
    notebook = json.loads(NOTEBOOK.read_text())
    setup = next(
        "".join(c["source"])
        for c in notebook["cells"]
        if "LOAD_WANDB =" in "".join(c["source"])
    )
    assert "LOAD_WANDB = False" in setup
    assert "wandb_run_snapshot.json" in setup


def test_framework_duration_legend_has_one_entry_per_sampler():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    namespace = helpers()
    frame = pd.DataFrame(
        [
            {
                "study_name": "one",
                "target": "throughput",
                "concurrency": 128,
                "sampler": "tpe",
                "duration_minutes": 10,
            },
            {
                "study_name": "two",
                "target": "e2e",
                "concurrency": 1,
                "sampler": "tpe",
                "duration_minutes": 20,
            },
            {
                "study_name": "three",
                "target": "e2e",
                "concurrency": 1,
                "sampler": "gp",
                "duration_minutes": 30,
            },
            {
                "study_name": "four",
                "target": "e2e",
                "concurrency": None,
                "sampler": None,
                "duration_minutes": 40,
            },
        ]
    )
    namespace.update(
        plt=plt,
        np=np,
        metadata=frame,
        MIN_RUN_SECONDS=300,
        OBJECTIVE_ORDER=["e2e", "throughput"],
        FRAMEWORK_NAMES={"vllm": "vLLM"},
    )
    fig, axes = namespace["plot_framework_coverage"](frame, "B", "vllm")
    try:
        labels = axes[1].get_legend_handles_labels()[1]
        assert labels == ["gp", "tpe", "unknown", "Cutoff: 5 min"]
        assert sum(
            len(collection.get_offsets()) for collection in axes[1].collections
        ) == len(frame)
        assert [t.get_text() for t in axes[1].get_xticklabels()] == [
            "e2e",
            "throughput",
        ]
        assert axes[0].get_legend_handles_labels()[1] == ["c1", "c128", "unknown"]
    finally:
        plt.close(fig)
