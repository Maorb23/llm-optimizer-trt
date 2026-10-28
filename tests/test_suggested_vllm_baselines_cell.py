"""Check notebook baseline orchestration without starting GPU servers."""
import json
import shlex
import sys
import warnings

import pytest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace


@pytest.mark.parametrize('tracking,fail', [(False, False), (True, False), (True, True)])
def test_suggested_baselines_share_workload_and_isolate_models(
    tmp_path, monkeypatch, tracking, fail,
):
    notebook = json.loads((Path(__file__).resolve().parents[1] /
        'docs/optuna_inference_optimization_lab_v2.ipynb').read_text())
    cell = notebook['cells'][-1]
    assert cell['id'] == 'suggested-vllm-baselines'
    calls = []

    @dataclass
    class Spec:
        server_command: str = 'vllm serve {model} --host {host} --port {port}'

    @dataclass
    class Measurement:
        completed: int = 7

    class Benchmark:
        def __init__(self, framework, snapshot, gpu, concurrency, **kwargs):
            self.workload = kwargs['_context'].workload_for(concurrency, kwargs['phase'])
            self.environment = kwargs['environment']
            self.spec = Spec()
            calls.append((concurrency, kwargs, self))

        def measure(self, framework, native, concurrency):
            assert native['seed'] == 2026
            if fail:
                raise RuntimeError('mock benchmark failure')
            return Measurement()

    monkeypatch.setitem(sys.modules, 'llm_optimizer.predefined', SimpleNamespace(
        SERVER_CONFIGS={'vllm': {'seed': None}}))
    monkeypatch.setitem(sys.modules, 'llm_optimizer.exceptions', SimpleNamespace(
        ServerCleanupError=type('ServerCleanupError', (Exception,), {})))
    wandb_runs = []

    class Run:
        def __init__(self, kwargs):
            self.kwargs = kwargs
            self.id = f'run-{len(wandb_runs)}'
            self.url = f'https://example.test/{self.id}'
            self.summary = {}
            self.logged = []
            self.exit_code = None

        def log(self, payload):
            self.logged.append(payload)

        def finish(self, exit_code):
            self.exit_code = exit_code

    def init(**kwargs):
        assert tracking
        run = Run(kwargs)
        wandb_runs.append(run)
        return run

    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(init=init))
    setups = []

    def configure(context, model, revision, **kwargs):
        assert kwargs['framework'] == 'vllm'
        assert kwargs['enable_dflash'] is False
        context['PHASE_MODELS']['B'] = model
        context['MODEL_SNAPSHOTS']['B'] = str(tmp_path)
        context['REAL_PREFLIGHT'][('B', 'vllm')] = {
            'gpu_type': 'test GPU', 'model_revision': 'resolved-commit',
        }
        setups.append(context)
        return {'target_path': str(tmp_path)}

    monkeypatch.setitem(sys.modules, 'llm_optimizer.experiment.hf_models',
                        SimpleNamespace(configure_hf_models=configure))
    models = {'B': 'Qwen/Qwen3.5-4B'}
    namespace = dict(
        WANDB_ENABLED=tracking, WANDB_PROJECT='project', WANDB_ENTITY='entity',
        WANDB_MODE='offline', WANDB_GROUP='study-group', warnings=warnings,
        PHASE_MODELS=models, MODEL_SNAPSHOTS={'B': str(tmp_path)},
        REAL_PREFLIGHT={('B', 'vllm'): {'gpu_type': 'original GPU'}},
        REAL_FRAMEWORKS=('vllm',), RUN_REAL_GPU=True, CONCURRENCIES=(1, 128),
        RUN_DIR=tmp_path, Path=Path, json=json, shlex=shlex,
        SharedRepositoryBenchmark=Benchmark, display=lambda row: None,
        artifact_value=lambda row: row,
        workload_for=lambda concurrency, phase: dict(
            max_concurrency=concurrency, num_prompts=7, seed=2026,
            dataset_name='random', apply_chat_template=True),
    )
    exec(compile(''.join(cell['source']), 'baseline-cell', 'exec'), namespace)
    assert not calls
    configs = namespace['SUGGESTED_VLLM_CONFIGS']
    rows = namespace['run_suggested_vllm_baselines']()
    assert len(setups) == 5
    assert namespace['REAL_PREFLIGHT'][('B', 'vllm')] == {'gpu_type': 'original GPU'}
    assert len(rows) == len(calls) == 10
    assert len(list(tmp_path.glob('*-results.json'))) == 10
    assert all(row['benchmark_valid'] is not fail for row in rows)
    assert len(wandb_runs) == (10 if tracking else 0)
    for run in wandb_runs:
        assert run.kwargs['group'] == 'study-group'
        assert run.kwargs['mode'] == 'offline'
        assert run.kwargs['job_type'] == 'suggested-vllm-baseline'
        assert run.kwargs['config']['workload']['seed'] == 2026
        assert run.summary['status'] == ('failed' if fail else 'completed')
        assert run.exit_code == int(fail)
        assert run.logged[0]['benchmark_valid'] is not fail
    if tracking:
        assert len({run.kwargs['name'] for run in wandb_runs}) == 10
        assert all('wandb' in row for row in rows)

    assert models == {'B': 'Qwen/Qwen3.5-4B'}
    for concurrency, kwargs, benchmark in calls:
        assert kwargs['num_prompts'] == 7
        assert kwargs['workload_seed'] == 2026
        assert benchmark.workload['max_concurrency'] == concurrency
        assert kwargs['_context'].PHASE_MODELS['B'] in configs
    assert '--language-model-only' in calls[4][2].spec.server_command
    assert configs['Qwen/Qwen3-4B']['max_num_seqs'] == 32
