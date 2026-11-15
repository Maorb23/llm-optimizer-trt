"""CPU checks for notebook-owned FP8 checkpoint selection and validation."""
import ast
import copy
import hashlib
import itertools
import json
from pathlib import Path
import struct
import tempfile
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'docs/phase1_experiments_trt_v2.ipynb'


def definitions(source, names=None):
    tree = ast.parse(source)
    return compile(ast.Module(body=[node for node in tree.body
        if isinstance(node, ast.FunctionDef) and (names is None or node.name in names)],
        type_ignores=[]), '<notebook>', 'exec')


class FP8NotebookTests(unittest.TestCase):
    def setUp(self):
        self.notebook = json.loads(NOTEBOOK.read_text())
        self.cells = [''.join(c['source']) for c in self.notebook['cells']]
        self.ns = dict(json=json, Path=Path, struct=struct, hashlib=hashlib,
                       copy=copy, itertools=itertools)
        for cell in self.cells:
            if 'def validate_fp8_checkpoint' in cell:
                exec(definitions(cell), self.ns)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = Path(self.tmp.name) / 'source'
        self.export = Path(self.tmp.name) / 'export'
        self.source.mkdir(); self.export.mkdir()
        (self.source / 'config.json').write_text(json.dumps({'model_type': 'qwen3_5'}))
        (self.export / 'config.json').write_text(json.dumps({
            'model_type': 'qwen3_5', 'quantization_config': {'quant_method': 'modelopt', 'quant_algo': 'FP8'}}))
        header = json.dumps({'layer.weight': {'dtype': 'F8_E4M3', 'shape': [1], 'data_offsets': [0, 1]}}).encode()
        (self.export / 'model.safetensors').write_bytes(struct.pack('<Q', len(header)) + header + b'\0')
        self.manifest = dict(source_snapshot=str(self.source.resolve()),
            source_config_sha256=self.ns['_file_sha'](self.source / 'config.json'),
            checkpoint=self.ns['_fp8_export_metadata'](self.export))
        (self.export / 'phase1-fp8-export.json').write_text(json.dumps(self.manifest))

    def test_notebook_compiles_and_has_no_saved_outputs(self):
        for c in self.notebook['cells']:
            if c['cell_type'] == 'code':
                compile(''.join(c['source']), '<cell>', 'exec')
                self.assertEqual(c['outputs'], [])
                self.assertIsNone(c['execution_count'])

    def test_accepts_export_and_rejects_changed_source_or_artifact(self):
        validate = self.ns['validate_fp8_checkpoint']
        self.assertEqual(validate(self.export, self.source), self.manifest)
        path = self.export / 'config.json'
        path.write_text(path.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'changed'):
            validate(self.export, self.source)
        (self.source / 'config.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'different pinned'):
            validate(self.export, self.source)

    def test_rejects_metadata_only_fp8_label(self):
        path = self.export / 'model.safetensors'
        header = json.dumps({'weight': {'dtype': 'BF16'}}).encode()
        path.write_bytes(struct.pack('<Q', len(header)) + header)
        with self.assertRaisesRegex(ValueError, 'no FP8-stored tensors'):
            self.ns['_fp8_export_metadata'](self.export)

    def test_precision_hook_preserves_approval_and_source_checks(self):
        original = (ROOT / 'src/llm_optimizer/experiment/phase1.py').read_text()
        exec(definitions(original, {'verify_precision'}), self.ns)
        self.ns['_original_verify_precision'] = self.ns['verify_precision']
        self.ns['TRT140_COMPATIBILITY'] = {'adapter': 'test-fp8'}
        adapter = next(c for c in self.cells if 'def notebook_verify_precision' in c)
        exec(definitions(adapter, {'notebook_verify_precision'}), self.ns)
        values = dict(frameworks=['trtllm'], run_real_gpu=True, phases=['A', 'B'],
            fixed_server={'trtllm': {}}, model_snapshots={'A': str(self.source), 'B': str(self.export)},
            real_preflight={'A': {'trtllm': {}}, 'B': {'trtllm': {
                'notebook_compatibility': {'adapter': 'test-fp8'},
                'weight_artifact': self.manifest, 'verified_precision_combinations': ['fp8/fp8']}}})
        precision = {'trtllm': {'weight_precision': ['fp8'], 'kv_cache_precision': ['fp8']}}
        verify = self.ns['notebook_verify_precision']
        verify(values, precision)
        self.assertEqual(values['model_snapshots']['B'], str(self.export))
        values['real_preflight']['B']['trtllm']['verified_precision_combinations'] = []
        with self.assertRaisesRegex(ValueError, 'verify precision combinations'):
            verify(values, precision)

    def test_factory_selects_real_checkpoint_and_separates_bf16_campaign(self):
        # Exercise the actual factory with a recording runner, without Optuna/GPU imports.
        self.ns['ROOT'] = Path(self.tmp.name)
        for marker in ['CAMPAIGN =', 'TTFT_LIMITS_MS =', 'PREFILL_BUDGETS =', 'PRECISION_CHOICES =']:
            cell = next(''.join(c['source']) for c in self.notebook['cells']
                        if c['cell_type'] == 'code' and marker in ''.join(c['source'])
                        and (marker != 'PRECISION_CHOICES =' or 'TRT_FP8_CALIBRATION =' in ''.join(c['source'])))
            exec(cell, self.ns)
        captured = []
        def runner(settings, *args, **kwargs):
            captured.append(settings)
            return settings
        self.ns.update(Phase1Runner=runner, MODEL_SNAPSHOT=str(self.source),
            MODEL_PROFILES={'trtllm': {}}, TRT_FP8_CHECKPOINT=self.export,
            MODEL_SETUP_IDENTITY=None, MODEL_SETUP_CHOICES=None)
        factory = next(c for c in self.cells if c.startswith('def make_runner('))
        exec(definitions(factory, {'make_runner'}), self.ns)
        self.ns['make_runner']()
        self.assertEqual(captured[-1]['model_snapshots']['A'], str(self.source))
        self.assertEqual(captured[-1]['model_snapshots']['B'], str(self.export.resolve()))
        self.assertIn('@modelopt-fp8-', captured[-1]['phase_models']['B'])
        self.assertIn('weight_artifact', captured[-1]['real_preflight']['B']['trtllm'])
        self.ns['PRECISION_CHOICES']['trtllm']['weight_precision'] = ['bf16']
        self.ns['make_runner']()
        self.assertEqual(captured[-1]['model_snapshots']['B'], str(self.source))
        self.assertNotIn('weight_artifact', captured[-1]['real_preflight']['B']['trtllm'])

    def test_serializer_overrides_inherited_fp8_cache_without_mutation(self):
        class Base:
            def serialize_engine_config(self, framework, concurrency, config): return config
        self.ns['ExplicitLab'] = Base
        adapter = next(c for c in self.cells if 'def notebook_serialize_engine_config' in c)
        exec(definitions(adapter, {'notebook_serialize_engine_config'}), self.ns)
        serialize = self.ns['notebook_serialize_engine_config']
        for cache, expected in [('auto', 'bfloat16'), ('fp8', 'fp8')]:
            config = {'kv_cache_config': {'dtype': cache, 'free_gpu_memory_fraction': .75},
                      'cuda_graph_config': None}
            result = serialize(SimpleNamespace(RUN_REAL_GPU=False), 'trtllm', 8, config)
            self.assertEqual(result['kv_cache_config']['dtype'], expected)
            self.assertEqual(config['kv_cache_config']['dtype'], cache)
            self.assertEqual(result['kv_cache_config']['free_gpu_memory_fraction'], .75)
            self.assertIsNone(result['cuda_graph_config'])


if __name__ == '__main__':
    unittest.main()
