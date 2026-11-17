"""CPU integration checks; run in the repository's dev/notebook environment."""
import copy
import json
import shlex
from pathlib import Path
from types import SimpleNamespace

import optuna
import pytest

from llm_optimizer.experiment.vllm_policy_v3 import (
    DEFAULT_OPTIONAL_PARAMS, VLLMLabV3, native_command, validate_options,
)
from llm_optimizer.experiment.telemetry_vllm_v3 import (
    DEFAULT_TELEMETRY, aggregate_prometheus, summarize, unavailable_telemetry,
)
from llm_optimizer.predefined.vllm import SERVER_CONFIGS

ROOT = Path(__file__).resolve().parents[1]


def namespace(tmp_path):
    ns = {}
    notebook = json.loads((ROOT / 'docs/phase1_experiments_vllm_v3.ipynb').read_text())
    prefixes = ('from pathlib', 'CAMPAIGN', '# Illustrative SLOs', 'PREFILL_BUDGETS',
                '# Enable only the additional', '# Independent Phase B',
                '# Compatibility is implemented', 'def make_runner')
    for cell in notebook['cells']:
        source = ''.join(cell['source'])
        if cell['cell_type'] == 'code' and source.startswith(prefixes):
            exec(compile(source, 'vllm-v3-notebook', 'exec'), ns)
    ns.update(RUN_REAL_GPU=False, WANDB_ENABLED=False, RUN_ROOT=tmp_path,
              TRIALS=2, SEEDS=[11], SAMPLER_NAMES=['tpe'])
    return ns


@pytest.mark.parametrize('concurrency', [1, 8, 16, 32, 128])
@pytest.mark.parametrize('mode', [0, 3])
def test_optional_sampling_native_command_and_isolation(tmp_path, concurrency, mode):
    ns = namespace(tmp_path)
    first = ns['make_runner']()
    assert not set(DEFAULT_OPTIONAL_PARAMS) & set(first.lab.sampling_menu('vllm', concurrency, 'expanded'))
    for option in ns['optional_params'].values():
        option['enabled'] = True
    runner = ns['make_runner']()
    lab = runner.lab
    assert type(lab) is VLLMLabV3
    assert runner.run_dir != first.run_dir
    menu = lab.sampling_menu('vllm', concurrency, 'expanded')
    values = {k: v[0] for k, v in menu.items()}
    values.update(attention_backend='FLASH_ATTN', graph_mode='FULL_AND_PIECEWISE',
                  compilation_mode=mode, weight_precision='fp8', kv_cache_precision='fp8',
                  enable_prefix_caching=False, async_scheduling=False)
    trial = optuna.trial.FixedTrial(values)
    config = lab.suggest_config(trial, 'vllm', concurrency, 'expanded')
    assert config['quantization'] == 'fp8_per_tensor'
    assert 'calculate_kv_scales' not in config
    assert config['compilation_config']['mode'] == mode
    assert config['compilation_config']['cudagraph_mode'] == (
        'FULL_DECODE_ONLY' if mode == 0 else 'FULL_AND_PIECEWISE')
    assert config['attention_config']['flash_attn_max_num_splits_for_cuda_graph'] == 8
    if concurrency == 1:
        assert 'long_prefill_token_threshold' not in trial.params
        assert 'prefill_schedule_interval' not in trial.params
    command = shlex.split(native_command(lab, config, '/tmp/model', concurrency))
    assert '--disable-prefix-caching' in command
    assert '--no-async-scheduling' in command
    assert '--quantization=fp8_per_tensor' in command
    assert not any(x.startswith('--compilation-mode') for x in command)
    assert 'attention_config' not in SERVER_CONFIGS
    assert not first.lab.optional_params['compilation_mode']['enabled']


def test_inactive_attention_split_and_base_unchanged(tmp_path):
    ns = namespace(tmp_path)
    ns['optional_params']['flash_attn_max_num_splits_for_cuda_graph']['enabled'] = True
    lab = ns['make_runner']().lab
    menu = lab.sampling_menu('vllm', 8, 'expanded')
    values = {k: v[0] for k, v in menu.items()}
    trial = optuna.trial.FixedTrial({**values, 'attention_backend': 'FLASHINFER', 'graph_mode': 'NONE'})
    config = lab.suggest_config(trial, 'vllm', 8, 'expanded')
    assert 'flash_attn_max_num_splits_for_cuda_graph' not in trial.params
    assert 'attention_config' not in config
    assert not set(DEFAULT_OPTIONAL_PARAMS) & set(lab.sampling_menu('vllm', 8, 'baseline'))


def test_official_gauges_and_missing_values():
    body = '\n'.join([
        'vllm:num_requests_running{model_name="m",engine="0"} 3',
        'vllm:num_requests_running{model_name="m",engine="1"} 5',
        'vllm:num_requests_waiting{model_name="m",engine="0"} 0',
        'vllm:num_requests_waiting_by_reason{model_name="m",engine="0",reason="capacity"} 12',
        'vllm:kv_cache_usage_perc{model_name="m",engine="0"} .2',
        'vllm:kv_cache_usage_perc{model_name="m",engine="1"} .6',
    ])
    values = aggregate_prometheus(body)
    assert values['running_requests'] == (8, None)
    assert values['queued_requests'] == (0, None)
    assert values['kv_cache_utilization'] == (.6, None)
    assert values['gpu_utilization'][0] is None
    assert aggregate_prometheus(body + '\nvllm:num_requests_running{worker="a"} 2')['running_requests'][0] is None
    assert unavailable_telemetry('disabled')['metrics']['queued_requests']['peak'] is None


def test_time_weighting_and_expiring_holds():
    result = summarize([(1, {'running_requests': (2, None)}),
                        (2, {'running_requests': (8, None)}),
                        (4, {'running_requests': (None, 'timeout')}),
                        (5, {'running_requests': (4, None)})], 0, 6, 1)
    metric = result['running_requests']
    assert metric['time_weighted_avg'] == 5.5
    assert metric['peak'] == 8
    assert metric['coverage'] == pytest.approx(4 / 6)


def test_invalid_domains():
    options = copy.deepcopy(DEFAULT_OPTIONAL_PARAMS)
    options['watermark']['values'] = [1.0]
    with pytest.raises(ValueError):
        validate_options(options, DEFAULT_TELEMETRY)


@pytest.mark.parametrize("failure", [False, True])
def test_versioned_runtime_persists_and_cleans_up(tmp_path, monkeypatch, failure):
    from llm_optimizer.experiment import runtime_vllm_v3 as runtime
    from llm_optimizer.experiment import telemetry_vllm_v3 as telemetry

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
        "framework": "vllm",
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
    ns = namespace(tmp_path)
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




def test_wandb_callback_receives_flat_telemetry_without_mutating_trial():
    from llm_optimizer.experiment.vllm_policy_v3 import V3Logger

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
