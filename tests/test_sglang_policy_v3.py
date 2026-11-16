"""CPU policy, command, notebook isolation and lifecycle tests for SGLang v3."""

import json
from pathlib import Path
from types import SimpleNamespace

import optuna
import pytest

from llm_optimizer.experiment.sglang_policy_v3 import (
    DEFAULT_OPTIONAL_PARAMS,
    V3_SERVER_CONFIGS,
    SGLangLabV3,
    logging_measurement,
    native_command,
)
from llm_optimizer.predefined.sglang import SERVER_CONFIGS

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "docs/phase1_experiments_sglang_v3.ipynb"


def notebook_namespace(tmp_path):
    namespace = {}
    notebook = json.loads(NOTEBOOK.read_text())
    prefixes = (
        "from pathlib",
        "CAMPAIGN",
        "TTFT_LIMITS",
        "# Illustrative SLOs",
        "PREFILL_BUDGETS",
        "optional_params",
        "# Enable only the remotely verified",
        "PRECISION_CHOICES",
        "# v3 uses",
        "def make_runner",
        "# Preserve W&B naming",
        "# CPU wiring",
    )
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and source.startswith(prefixes):
            exec(compile(source, str(NOTEBOOK), "exec"), namespace)
    namespace.update(
        RUN_ROOT=tmp_path,
        WANDB_ENABLED=False,
        TRIALS=2,
        FRAMEWORKS=["sglang"],
        SAMPLER_NAMES=["tpe"],
        SEEDS=[11],
    )
    return namespace


@pytest.fixture
def lab(tmp_path):
    ns = notebook_namespace(tmp_path)
    for option in ns["optional_params"].values():
        option["enabled"] = True
    return ns["make_runner"]().lab


def params(lab, concurrency, **overrides):
    menu = lab.sampling_menu("sglang", concurrency, "expanded")
    result = {name: values[0] for name, values in menu.items()}
    result.update(
        cuda_graph_backend_decode="full",
        attention_backend="triton",
        prefill_decode_interval=4,
        **overrides,
    )
    return result


@pytest.mark.parametrize("concurrency", [1, 8, 16, 32, 128])
@pytest.mark.parametrize("policy", ["on__fcfs", "on__lpm", "off__fcfs"])
def test_policy_optional_rendering_and_real_cli(lab, concurrency, policy):
    trial = optuna.trial.FixedTrial(params(lab, concurrency, cache_policy=policy))
    config = lab.suggest_config(trial, "sglang", concurrency, "expanded")
    assert config["disable_radix_cache"] == policy.startswith("off")
    assert config["schedule_policy"] == policy.split("__")[1]
    assert "disable_radix_cache" not in trial.params
    assert "schedule_policy" not in trial.params
    command = native_command(lab, config, "/tmp/model", concurrency)
    assert "--linear-attn-decode-backend=triton" in command
    assert "--cuda-graph-config=" in command
    assert "--cuda-graph-backend-decode=" not in command
    assert "cache_policy" not in command and "decode_capture_profile" not in command
    assert "prefill_request_cap_choice" not in command
    native = lab.serialize_engine_config("sglang", concurrency, config)
    graph = json.loads(native["cuda_graph_config"])
    assert graph["prefill"]["backend"] == config["cuda_graph_backend_prefill"]
    assert max(graph["decode"]["bs"]) <= min(
        concurrency, config["max_running_requests"]
    )
    if concurrency == 1:
        assert graph["decode"]["bs"] == [1]
        assert config["prefill_decode_interval"] == 0
        assert "prefill_max_requests" not in trial.params
    else:
        assert config["prefill_max_requests"] <= min(
            concurrency, config["max_running_requests"]
        )
        assert config["prefill_decode_interval"] == 4
        assert config["prefill_decode_interval_min_bs"] == min(8, concurrency)


def test_legacy_registry_and_class_unchanged(lab):
    from llm_optimizer.experiment.phase1 import ExplicitLab

    assert type(lab) is SGLangLabV3
    assert "linear_attn_decode_backend" not in SERVER_CONFIGS
    assert "linear_attn_decode_backend" in V3_SERVER_CONFIGS
    assert ExplicitLab.expanded_config is not SGLangLabV3.expanded_config
    assert "off__lpm" not in lab.sampling_menu("sglang", 8, "expanded")["cache_policy"]


def test_inactive_parameters_not_sampled(lab):
    values = params(lab, 8)
    values.update(attention_backend="fa3", cuda_graph_backend_decode="disabled")
    trial = optuna.trial.FixedTrial(values)
    config = lab.suggest_config(trial, "sglang", 8, "expanded")
    assert "triton_attention_num_kv_splits" not in trial.params
    assert "decode_capture_profile" not in trial.params
    assert "cuda_graph_config" not in config


def test_disabled_optional_domains_and_fingerprints(tmp_path):
    ns = notebook_namespace(tmp_path)
    runner = ns["make_runner"]()
    lab = runner.lab
    menu = lab.sampling_menu("sglang", 8, "expanded")
    assert not set(DEFAULT_OPTIONAL_PARAMS).intersection(menu)
    ns["optional_params"]["linear_attn_decode_backend"]["enabled"] = True
    second = ns["make_runner"]()
    assert second.run_dir != runner.run_dir
    assert runner.lab.optional_params["linear_attn_decode_backend"]["enabled"] is False


def test_optuna_stable_domains_with_capacity_changes(lab):
    study = optuna.create_study()

    def objective(trial):
        config = lab.suggest_config(trial, "sglang", 128, "expanded")
        assert not (
            config["disable_radix_cache"] and config["schedule_policy"] == "lpm"
        )
        return config["max_running_requests"]

    study.optimize(objective, n_trials=8)
    assert len(study.trials) == 8
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)


def test_wandb_payload_rps_concurrency_null_and_partial():
    execution = {
        "request_throughput": 13.5,
        "configured_client_concurrency": 32,
        "server_telemetry": {
            "metrics": {
                "gpu_utilization": {
                    "peak": None,
                    "time_weighted_avg": None,
                    "reason": "no NVML",
                }
            }
        },
    }
    trial = SimpleNamespace(user_attrs={"execution": execution})
    values = logging_measurement(trial, {"concurrency": 32})
    assert values["request_throughput"] == 13.5
    assert values["configured_client_concurrency"] == 32
    assert values["server/gpu_utilization/peak"] is None
    assert values["server/gpu_utilization/reason"] == "no NVML"


def test_notebook_cells_compile_and_previous_outputs_preserved():
    notebook = json.loads(NOTEBOOK.read_text())
    old = json.loads((ROOT / "docs/phase1_experiments_sglang_v2.ipynb").read_text())
    assert len(notebook["cells"]) == len(old["cells"]) + 2
    for index, cell in enumerate(old["cells"]):
        copied = notebook["cells"][index if index < 10 else index + 2]
        assert copied.get("outputs") == cell.get("outputs")
        assert copied["metadata"] == cell["metadata"]
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code":
            compile(source, str(NOTEBOOK), "exec")
            assert "VLLM" not in source and '["vllm"]' not in source


@pytest.mark.parametrize("failure", [False, True])
def test_versioned_runtime_persists_and_cleans_up(tmp_path, monkeypatch, failure):
    from llm_optimizer.experiment import runtime_sglang_v3 as runtime
    from llm_optimizer.experiment import telemetry_sglang_v3 as telemetry

    process = SimpleNamespace()
    cleanup = []
    monkeypatch.setattr(runtime, "start_server", lambda *a: process)
    monkeypatch.setattr(runtime, "get_server_env", lambda *a: {})
    monkeypatch.setattr(
        runtime, "terminate_process_top_down", lambda p: cleanup.append(p)
    )
    monkeypatch.setattr(runtime, "_close_server_log", lambda p: None)

    def benchmark(args):
        record = args["_telemetry"]
        record.data["server_telemetry"] = {"metrics": {"running_requests": {"peak": 2}}}
        record.data["request_throughput"] = 4.0
        if failure:
            raise optuna.TrialPruned()
        return {
            "completed": 4,
            "failed": 0,
            "total_requests": 4,
            "request_throughput": 4.0,
            "server_telemetry": record.data["server_telemetry"],
        }

    monkeypatch.setattr(telemetry, "run_benchmark", benchmark)
    options = {
        "framework": "sglang",
        "model": "/tmp/model",
        "server_cmd": "unused",
        "managed_server": True,
        "gpus": 1,
        "gpu_type": "CPU mock",
        "host": "127.0.0.1",
        "port": 30000,
        "ready_url": "unused",
        "mute_server": True,
        "constraints_for_viz": [],
        "output_jsonl_path": tmp_path / "bench.jsonl",
    }
    settings = {"client_args": {}, "server_args": {}, "server_cmd_args": []}
    if failure:
        with pytest.raises(optuna.TrialPruned) as caught:
            runtime.run_single_configuration(settings, **options)
        assert (
            caught.value.execution["server_telemetry"]["metrics"]["running_requests"][
                "peak"
            ]
            == 2
        )
        assert not options["output_jsonl_path"].exists()
    else:
        runtime.run_single_configuration(settings, **options)
        row = json.loads(options["output_jsonl_path"].read_text())
        assert row["results"]["request_throughput"] == 4
        assert (
            row["metadata"]["execution"]["server_telemetry"]["metrics"][
                "running_requests"
            ]["peak"]
            == 2
        )
    assert cleanup == [process]


def test_cpu_study_resume_and_explicit_missing_server_metrics(tmp_path):
    ns = notebook_namespace(tmp_path)
    for option in ns["optional_params"].values():
        option["enabled"] = True
    runner = ns["make_runner"]()
    first = runner.run_objective_group("B", "throughput", [8])
    study = next(iter(first.values()))
    assert len(study.trials) == 2
    for trial in study.trials:
        assert trial.state == optuna.trial.TrialState.COMPLETE
        assert trial.user_attrs["configured_client_concurrency"] == 8
        metric = trial.user_attrs["server_telemetry"]["metrics"]["running_requests"]
        assert metric["peak"] is None and metric["time_weighted_avg"] is None
        assert "simulation" in metric["reason"]
    resumed = ns["make_runner"]().run_objective_group("B", "throughput", [8])
    assert len(next(iter(resumed.values())).trials) == 2


def test_measurement_attrs_preserve_rps_and_server_metrics(lab):
    from llm_optimizer.experiment.lab import Measurement

    study = optuna.create_study()
    trial = study.ask()
    telemetry = {"metrics": {"running_requests": {"peak": 7}}}
    execution = {"server_telemetry": telemetry, "request_throughput": 6.5}
    measurement = Measurement(
        50.0, 10.0, 10, 10, 0, 2.0, 2.0, "CPU mock", 100.0, execution
    )
    lab.record_measurement(trial, measurement, 8)
    attrs = study.trials[0].user_attrs
    assert attrs["request_throughput"] == 6.5
    assert attrs["server_telemetry"] == telemetry
    assert attrs["measurement"]["execution"]["server_telemetry"] == telemetry


def test_wandb_callback_receives_flat_telemetry_without_mutating_trial():
    from llm_optimizer.experiment.sglang_policy_v3 import V3Logger

    captured = []
    logger = V3Logger(
        lambda study, trial: captured.append(trial.user_attrs), {"concurrency": 8}
    )
    attrs = {
        "execution": {
            "request_throughput": 5.0,
            "server_telemetry": {
                "metrics": {"running_requests": {"peak": 6, "unit": "requests"}}
            },
        }
    }
    trial = SimpleNamespace(user_attrs=attrs)
    logger(None, trial)
    assert captured[0]["measurement"]["request_throughput"] == 5
    assert captured[0]["measurement"]["server/running_requests/peak"] == 6
    assert "measurement" not in attrs
