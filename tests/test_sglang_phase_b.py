"""CPU coverage for the resolved Phase B SGLang search and GP failure policy."""

import ast
import copy
import json
from pathlib import Path

import optuna
import pytest
from test_optuna_notebook import lab as reference_notebook_lab_fixture

from llm_optimizer.args import Arg, ArgScope
from llm_optimizer.cli import construct_benchmark_settings
from llm_optimizer.experiment.config import DEFAULTS, ExperimentConfig
from llm_optimizer.experiment.gp_policy import BlockedGPProposal, GPFailurePolicy
from llm_optimizer.experiment.lab import BenchmarkFailure
from llm_optimizer.experiment.sglang_space import (
    CACHE_POLICIES,
    CACHE_POLICY_PARAMETER,
    resolve_cache_policy,
    sampling_choices,
)
from llm_optimizer.experiment.workflow import create_lab

notebook_lab = reference_notebook_lab_fixture


@pytest.fixture
def lab(tmp_path):
    return create_lab(ExperimentConfig(copy.deepcopy(DEFAULTS)), tmp_path)


@pytest.mark.parametrize("concurrency", [1, 128])
@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize("policy", ["fcfs", "lpm"])
def test_separate_parameters_preserve_all_four_combinations(
    lab, concurrency, disabled, policy
):
    menu = lab.sampling_menu("sglang", concurrency, "expanded")
    assert CACHE_POLICY_PARAMETER not in menu
    assert menu["disable_radix_cache"] == [False, True]
    assert menu["schedule_policy"] == ["fcfs", "lpm"]
    params = {key: values[0] for key, values in menu.items()}
    params.update(disable_radix_cache=disabled, schedule_policy=policy)
    trial = optuna.trial.FixedTrial(params)
    config = lab.suggest_config(trial, "sglang", concurrency, "expanded")
    assert config["disable_radix_cache"] == disabled
    assert trial.params["disable_radix_cache"] == disabled
    assert config["schedule_policy"] == trial.params["schedule_policy"] == policy
    assert CACHE_POLICY_PARAMETER not in config
    assert CACHE_POLICY_PARAMETER not in trial.params


def test_retained_helper_can_still_reject_invalid_pairs_when_explicitly_used(lab):
    choices = copy.deepcopy(lab.EXPANDED_CHOICES["sglang"])
    choices["disable_radix_cache"] = [True]
    assert sampling_choices(choices)[CACHE_POLICY_PARAMETER] == [
        "cache_off__fcfs",
        "cache_off__lpm",
    ]
    choices["schedule_policy"] = ["shortest-prefill-first"]
    with pytest.raises(ValueError, match="no valid cache/policy pair"):
        sampling_choices(choices)
    with pytest.raises(ValueError):
        resolve_cache_policy("cache_off__shortest-prefill-first")


@pytest.mark.parametrize("sampler_name", ["tpe_multivariate", "gp"])
def test_real_sampler_uses_original_separate_parameters(lab, sampler_name):
    if sampler_name == "gp":
        pytest.importorskip("torch")
    # Reach model-based sampling within a short CPU test, not just startup proposals.
    lab.SAMPLER_SETTINGS = {**lab.SAMPLER_SETTINGS, "startup_multi": 2}
    sampler = lab.make_sampler(
        sampler_name, "sglang", 128, seed=11, space_kind="expanded"
    )
    if sampler_name == "tpe_multivariate":
        assert sampler._multivariate and sampler._group
    study = optuna.create_study(sampler=sampler)

    def objective(trial):
        config = lab.suggest_config(trial, "sglang", 128, "expanded")
        trial.set_user_attr("config", config)
        return config["max_running_requests"] + config["schedule_conservativeness"]

    study.optimize(objective, n_trials=5)
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    assert all(
        tuple(t.distributions["disable_radix_cache"].choices) == (False, True)
        and tuple(t.distributions["schedule_policy"].choices)
        == ("fcfs", "lpm")
        for t in study.trials
    )
    assert all(
        CACHE_POLICY_PARAMETER not in t.params
        for t in study.trials
    )


def test_breakable_reaches_native_sglang_command(lab):
    assert lab.EXPANDED_CHOICES["sglang"]["cuda_graph_backend_decode"] == [
        "disabled",
        "full",
    ]
    assert lab.EXPANDED_CHOICES["sglang"]["cuda_graph_backend_prefill"] == [
        "disabled",
        "breakable",
    ]
    params = {
        key: values[0]
        for key, values in lab.sampling_menu("sglang", 128, "expanded").items()
    }
    params["cuda_graph_backend_prefill"] = "breakable"
    config = lab.suggest_config(
        optuna.trial.FixedTrial(params), "sglang", 128, "expanded"
    )
    native = lab.serialize_engine_config("sglang", 128, config)
    assert native["cuda_graph_backend_prefill"] == "breakable"
    argument = Arg[str](
        scope=ArgScope.SERVER,
        name="cuda_graph_backend_prefill",
        value=native["cuda_graph_backend_prefill"],
    )
    settings = construct_benchmark_settings([argument], "sglang")
    assert "--cuda-graph-backend-prefill=breakable" in settings["server_cmd_args"]


def test_contract_rejects_old_search_space_and_preserves_same_space_resume(lab):
    study = lab.run_study(
        "B-sglang-c128-e2e-random-s11",
        "sglang",
        "random",
        phase="B",
        concurrency=128,
        target="e2e",
        space_kind="expanded",
        trials=1,
    )
    contract = study.user_attrs["contract"]
    assert CACHE_POLICY_PARAMETER not in contract["sampling_menu"]
    assert contract["expanded_choices"]["cuda_graph_backend_prefill"] == [
        "disabled",
        "breakable",
    ]
    resumed = lab.run_study(
        study.study_name,
        "sglang",
        "random",
        phase="B",
        concurrency=128,
        target="e2e",
        space_kind="expanded",
        trials=1,
    )
    assert len(resumed.trials) == len(study.trials)
    old = copy.deepcopy(contract)
    old["sampling_menu"] = sampling_choices(old["sampling_menu"])
    study.set_user_attr("contract", old)
    with pytest.raises(ValueError, match="different experimental settings"):
        lab.run_study(
            study.study_name,
            "sglang",
            "random",
            phase="B",
            concurrency=128,
            target="e2e",
            space_kind="expanded",
            trials=2,
        )


def test_resumed_gp_restores_effective_config_without_audit_file(tmp_path):
    storage = f"sqlite:///{tmp_path / 'gp.sqlite3'}"
    study = optuna.create_study(study_name="B-sglang-c128-e2e-gp-s11", storage=storage)
    effective = {
        **resolve_cache_policy("cache_off__lpm"),
        "cuda_graph_backend_prefill": "breakable",
    }
    study.add_trial(
        optuna.trial.create_trial(
            state=optuna.trial.TrialState.FAIL,
            user_attrs={"effective_config": effective, "failure_reason": "OOM"},
        )
    )
    resumed = optuna.load_study(study_name=study.study_name, storage=storage)
    policy = GPFailurePolicy(resumed, tmp_path, {"model": "phase-b", "gpu": "H100"})
    trial = resumed.ask()
    with pytest.raises(BlockedGPProposal):
        policy.check(trial, dict(reversed(list(effective.items()))))
    resumed.tell(trial, state=optuna.trial.TrialState.FAIL)
    assert resumed.trials[-1].user_attrs["gp_rejected"]
    assert all(t.values is None for t in resumed.trials)
    assert len(policy.data["failures"]) == 1


def test_gp_rejections_do_not_launch_or_consume_phase_b_budget(lab, monkeypatch):
    # Force a deterministic GP proposal sequence while exercising the real study loop.
    base = lab.render_engine_config(
        "sglang",
        128,
        {
            "max_running_requests": 8,
            "chunked_prefill_size": 4096,
            "mem_fraction_static": 0.75,
        },
    )
    good = {**base, **resolve_cache_policy("cache_on__fcfs")}
    bad = {**base, **resolve_cache_policy("cache_off__lpm"), "max_running_requests": 16}
    proposals = iter([bad, bad, good])
    monkeypatch.setattr(
        lab, "make_sampler", lambda *a, **k: optuna.samplers.GPSampler()
    )
    monkeypatch.setattr(lab, "suggest_config", lambda *a, **k: next(proposals).copy())
    launches = []
    benchmark = lab.Simulator(phase="B")
    original = benchmark.measure

    def measure(framework, config, concurrency):
        launches.append(config)
        if config == bad:
            raise BenchmarkFailure("OOM")
        return original(framework, config, concurrency)

    monkeypatch.setattr(benchmark, "measure", measure)
    study = lab.run_study(
        "B-sglang-c128-e2e-gp-budget",
        "sglang",
        "gp",
        phase="B",
        concurrency=128,
        target="e2e",
        space_kind="expanded",
        benchmark=benchmark,
        trials=2,
    )
    assert launches == [bad, good]
    assert [t.state.name for t in study.trials] == ["FAIL", "FAIL", "COMPLETE"]
    assert study.trials[1].user_attrs["gp_rejected"]
    assert study.trials[0].values is study.trials[1].values is None
    log = json.loads(Path(study.user_attrs["gp_failure_log"]).read_text())
    assert log["rejections"][0]["server_launched"] is False


@pytest.mark.parametrize(
    "notebook",
    [
        "docs/optuna_inference_optimization_lab.ipynb",
        "docs/optuna_inference_optimization_lab_prune.ipynb",
        "optuna_inference_optimization_lab_updated.ipynb",
    ],
)
def test_notebook_search_representation_and_breakable(notebook_lab, notebook):
    path = Path(__file__).resolve().parents[1] / notebook
    cells = json.loads(path.read_text())["cells"]
    namespace = notebook_lab.copy()
    functions = {
        "expanded_choices",
        "expanded_config",
        "sampling_menu",
        "suggest_config",
    }
    for index in (6, 10):
        nodes = ast.parse("".join(cells[index]["source"])).body
        definitions = [
            node
            for node in nodes
            if (isinstance(node, ast.FunctionDef) and node.name in functions)
            or (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "EXPANDED_CHOICES"
                    for target in node.targets
                )
            )
        ]
        exec(
            compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"),
            namespace,
        )
    menu = namespace["sampling_menu"]("sglang", 128, "expanded")
    assert menu["cuda_graph_backend_prefill"] == ["disabled", "breakable"]
    assert menu["cuda_graph_backend_decode"] == (
        ["disabled", "full"]
        if notebook == "docs/optuna_inference_optimization_lab.ipynb"
        else ["disabled", "full", "tc_piecewise"]
    )
    separate = notebook.endswith("_prune.ipynb")
    if separate:
        assert CACHE_POLICY_PARAMETER not in menu
        assert menu["disable_radix_cache"] == [False, True]
        assert menu["schedule_policy"] == ["fcfs", "lpm", "shortest-prefill-first"]
        assert "sglang_space" not in "".join(
            "".join(cell["source"]) for cell in cells
        )
    else:
        expected = list(CACHE_POLICIES)
        if notebook == "docs/optuna_inference_optimization_lab.ipynb":
            expected = [p for p in expected if not p.endswith("shortest-prefill-first")]
        assert menu[CACHE_POLICY_PARAMETER] == expected
    policies = CACHE_POLICIES if separate else menu[CACHE_POLICY_PARAMETER]
    for policy in policies:
        params = {key: values[0] for key, values in menu.items()}
        if separate:
            params.update(resolve_cache_policy(policy))
        else:
            params[CACHE_POLICY_PARAMETER] = policy
        params["cuda_graph_backend_prefill"] = "breakable"
        config = namespace["suggest_config"](
            optuna.trial.FixedTrial(params), "sglang", 128, "expanded"
        )
        assert config["cuda_graph_backend_prefill"] == "breakable"
        assert (
            config["schedule_policy"] == resolve_cache_policy(policy)["schedule_policy"]
        )
        assert (
            config["disable_radix_cache"]
            == resolve_cache_policy(policy)["disable_radix_cache"]
        )
