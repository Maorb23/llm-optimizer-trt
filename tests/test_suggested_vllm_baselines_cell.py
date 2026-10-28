"""Check notebook baseline orchestration without starting GPU servers."""
import json
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace


def test_suggested_baselines_share_workload_and_isolate_models(tmp_path, monkeypatch):
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
            return Measurement()

    monkeypatch.setitem(sys.modules, 'llm_optimizer.predefined', SimpleNamespace(
        SERVER_CONFIGS={'vllm': {'seed': None}}))
    monkeypatch.setitem(sys.modules, 'llm_optimizer.exceptions', SimpleNamespace(
        ServerCleanupError=type('ServerCleanupError', (Exception,), {})))
    models = {'B': 'Qwen/Qwen3.5-4B'}
    namespace = dict(
        PHASE_MODELS=models, MODEL_SNAPSHOTS={'B': str(tmp_path)},
        REAL_PREFLIGHT={}, RUN_REAL_GPU=True, CONCURRENCIES=(1, 128),
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
    namespace['SUGGESTED_BASELINE_SNAPSHOTS'] = {m: str(tmp_path) for m in configs}
    namespace['SUGGESTED_BASELINE_ENVIRONMENTS'] = {
        m: {'gpu_type': 'test GPU'} for m in configs}
    rows = namespace['run_suggested_vllm_baselines']()
    assert len(rows) == len(calls) == 10
    assert len(list(tmp_path.glob('*-results.json'))) == 10
    assert all(row['benchmark_valid'] for row in rows)
    assert models == {'B': 'Qwen/Qwen3.5-4B'}
    for concurrency, kwargs, benchmark in calls:
        assert kwargs['num_prompts'] == 7
        assert kwargs['workload_seed'] == 2026
        assert benchmark.workload['max_concurrency'] == concurrency
        assert kwargs['_context'].PHASE_MODELS['B'] in configs
    assert '--language-model-only' in calls[4][2].spec.server_command
    assert configs['Qwen/Qwen3-4B']['max_num_seqs'] == 32
