"""Exercise the optional final notebook cell without loading a model or GPU."""

import ast
import hashlib
import json
import random
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest


def baseline_cell():
    path = (
        Path(__file__).resolve().parents[1]
        / "docs/optuna_inference_optimization_lab_prune.ipynb"
    )
    notebook = json.loads(path.read_text())
    assert notebook["cells"][-1]["id"] == "transformers-c1-baseline"
    tree = ast.parse("".join(notebook["cells"][-1]["source"]))
    # Users may enable this optional cell in their saved notebook. Unit tests
    # exercise the function explicitly with mocks rather than its live-run toggle.
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "RUN_TRANSFORMERS_BASELINE"
            for target in node.targets
        ):
            node.value = ast.Constant(value=False)
    return ast.unparse(ast.fix_missing_locations(tree))


@pytest.mark.parametrize("fail", [False, True])
def test_sequential_baseline_percentile_warmup_and_failure(tmp_path, monkeypatch, fail):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text('{"architectures": ["FakeModel"]}')
    calls = []
    times = iter([0, .032, 1, 1.1, 2, 2.2, 3, 3.3])

    class Output:
        def __init__(self, count):
            self.count = count

        def __getitem__(self, key):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return [7] * self.count

    class Batch(dict):
        def to(self, device):
            return self

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 1

        def __call__(self, prompt, **kwargs):
            return Batch(input_ids=types.SimpleNamespace(shape=(1, 512)))

        def decode(self, output, **kwargs):
            return "output"

    class Model:
        def to(self, device):
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):
            calls.append(kwargs)
            if fail and len(calls) == 3:
                raise RuntimeError("OOM")
            assert kwargs["min_new_tokens"] == kwargs["max_new_tokens"]
            return Output(kwargs["max_new_tokens"])

    def load_model(path, **kwargs):
        assert Path(path) == snapshot
        assert kwargs["local_files_only"] is True
        return Model()

    fake_transformers = types.SimpleNamespace(
        __version__="test", FakeModel=types.SimpleNamespace(from_pretrained=load_model),
        AutoConfig=types.SimpleNamespace(from_pretrained=lambda *a, **k:
            types.SimpleNamespace(
                architectures=["FakeModel"], _commit_hash="pinned"
            )),
        AutoTokenizer=types.SimpleNamespace(
            from_pretrained=lambda *a, **k: Tokenizer()
        ),
        GenerationConfig=lambda **kwargs: types.SimpleNamespace(**kwargs),
    )
    synchronize = []
    fake_torch = types.SimpleNamespace(
        __version__="test", device=lambda value: value, bfloat16="bfloat16",
        inference_mode=nullcontext, cuda=types.SimpleNamespace(
            is_available=lambda: True, synchronize=synchronize.append,
            empty_cache=lambda: None, get_device_name=lambda device: "mock H100",
        ),
    )
    requests = [
        types.SimpleNamespace(prompt=f"request {i}", prompt_len=512, output_len=128)
        for i in range(3)
    ]
    client = types.SimpleNamespace(
        get_dataset=lambda workload, tokenizer: requests,
        build_workload_manifest=lambda rows, *args: {
            "requests": [vars(row) for row in rows], "sha256": "manifest",
        },
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "llm_optimizer.bench_client", client)
    namespace = {
        "Path": Path, "MODEL_SNAPSHOTS": {"B": str(snapshot)}, "RUN_DIR": tmp_path,
        "PHASE_MODELS": {"B": "phase-b-model"}, "random": random, "np": np,
        "hashlib": hashlib, "json": json, "display": lambda result: None,
        "time": types.SimpleNamespace(perf_counter=lambda: next(times)),
        "workload_for": lambda concurrency, phase, num_prompts: {
            "seed": 2026, "num_prompts": num_prompts, "max_concurrency": concurrency,
        },
    }
    exec(compile(baseline_cell(), "transformers-baseline-cell", "exec"), namespace)
    assert calls == []  # Disabled cell is safe to execute without loading any model.
    if fail:
        with pytest.raises(RuntimeError, match="Invalid Transformers baseline"):
            namespace["run_transformers_c1_baseline"]("B", 3, 1)
    else:
        result = namespace["run_transformers_c1_baseline"]("B", 3, 1)
        assert result["completed"] == 3
        assert result["mean_e2e_latency_ms"] == pytest.approx(200)
        assert result["p95_e2e_latency_ms"] == pytest.approx(290)
        assert result["max_concurrency"] == result["batch_size"] == 1
        assert [call["max_new_tokens"] for call in calls] == [32, 128, 128, 128]
        assert len(synchronize) == 8
    saved = json.loads(next(tmp_path.glob("transformers-*-results.json")).read_text())
    assert saved["benchmark_valid"] is not fail
    if fail:
        assert saved["p95_e2e_latency_ms"] is None
        assert saved["completed"] == 1
        assert saved["failed_or_uncompleted"] == 2
