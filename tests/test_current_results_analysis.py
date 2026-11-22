"""Focused report invariants; no experiment execution or writes to artifacts."""
import importlib.util
from pathlib import Path
import unittest
from types import SimpleNamespace

import pandas as pd

spec = importlib.util.spec_from_file_location('current_results_analysis', Path(__file__).resolve().parents[1] / 'docs/current_results_analysis.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


class CurrentResultsTests(unittest.TestCase):
    def test_verified_names_only_and_conflicts(self):
        for name, expected in [('shared-prefix-InVeSt-B-sglang', 'Parameters probe'),
                               ('shared-prefix-B-vllm', 'Initial runs')]:
            self.assertEqual(a.classify_names([{'name': name, 'source': 'ID-matched cache'}])[0], expected)
        self.assertEqual(a.classify_names([])[0], 'unclassified')
        self.assertEqual(a.classify_names([{'name': 'initial'}, {'name': 'invest'}])[0], 'unclassified')
        self.assertEqual(a.classify_names([{'name': 'same'}, {'name': 'same'}])[0], 'Initial runs')

    def test_cached_wandb_latest_full_name_and_checksum(self):
        import tempfile, struct, zlib
        def record(name):
            # Wire fields from W&B RunRecord: run_id=1, display_name=8.
            run = b'\x0a\x08abcdefgh' + b'\x42' + bytes([len(name)]) + name
            payload = b'\x8a\x01' + bytes([len(run)]) + run
            return struct.pack('<IHB', zlib.crc32(b'\x01' + payload), len(payload), 1) + payload
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'runs/20261117-143416/wandb/run-abcdefgh.wandb'
            path.parent.mkdir(parents=True)
            path.write_bytes(b':W&B\xe1\xbe\x00' + record(b'bare-study') + record(b'full-InVeSt-study'))
            self.assertEqual(list(a.cached_wandb_names(path))[-1], ('abcdefgh', 'full-InVeSt-study'))
            studies = [{'attrs': {'wandb': {'id': 'abcdefgh'}}, 'source': 'db'}]
            a.reconcile_names(root, studies, [])
            self.assertEqual(studies[0]['section'], 'Parameters probe')
            self.assertEqual(studies[0]['name_evidence'][0]['prior_names'], ['bare-study'])
            path.write_bytes(path.read_bytes()[:-1] + b'!')
            with self.assertRaises(ValueError):
                list(a.cached_wandb_names(path))

    def test_modelopt_headings_combine_variants_without_changing_identities(self):
        base = 'Qwen/Qwen3-14B'
        variant = base + '@modelopt-fp8-323a067030074b29'
        other_variant = base + '@modelopt-fp8-another-artifact'
        revision = base + '@revision-abc'
        self.assertEqual(a.model_heading(variant), base)
        self.assertEqual(a.model_heading(other_variant), base)
        self.assertEqual(a.model_heading(base + '@modelop'), base)
        self.assertEqual(a.model_heading(revision), revision)
        report = a.Report.__new__(a.Report)
        report.runs = pd.DataFrame({'section': ['Initial runs'] * 5 + ['Parameters probe'],
            'model': [base, variant, other_variant, variant, revision, variant],
            'framework': ['trtllm'] * 6, 'concurrency': [8, 8, 8, 16, 8, 8],
            'context': ['base', 'artifact-a', 'artifact-b', 'c16', 'revision', 'probe']})
        original = report.runs.copy(deep=True)
        selected = report.select('Initial runs', base, 'trtllm', 8)
        self.assertEqual(set(selected.context), {'base', 'artifact-a', 'artifact-b'})
        self.assertEqual(len(report.select('Initial runs', variant, 'trtllm', 8)), 3)
        self.assertEqual(len(report.select('Parameters probe', base, 'trtllm', 8)), 1)
        pd.testing.assert_frame_equal(report.runs, original)
        headings = []
        report.show_model = lambda section, model: headings.append(model)
        original_heading = a.heading
        a.heading = lambda *args: None
        try:
            report.show_section('Initial runs')
        finally:
            a.heading = original_heading
        self.assertEqual(headings, [base, revision])

    def test_objective_groups_combine_studies_across_metadata(self):
        throughput = {'section': 'Initial runs', 'model': 'org/model', 'framework': 'vllm',
                      'directions': ['MAXIMIZE'], 'contract': {'metrics': ['output_throughput'], 'concurrency': 8}}
        variant = {**throughput, 'model': 'org/model@modelopt-fp8-abc', 'notebook_version': 'v3',
                   'contract': {**throughput['contract'], 'phase': 'B', 'sampler': 'GP', 'workload': 'other'}}
        latency = {**variant, 'directions': ['MINIMIZE'],
                   'contract': {**variant['contract'], 'metrics': ['p95_e2e_latency_ms']}}
        self.assertEqual(a.report_group(throughput), a.report_group(variant))
        self.assertNotEqual(a.report_group(variant), a.report_group(latency))
        rows = pd.DataFrame([
            {'context': a.report_group(run), 'model': run['model'], 'study_key': key,
             'trial': trial, 'values': [value]}
            for run, key, trial, value in [(throughput, 'a', 0, 4), (variant, 'b', 0, 7),
                                           (variant, 'b', 1, 8), (latency, 'c', 0, 100)]])
        selected = a.group_rows(rows, a.report_group(throughput))
        ranked = a.best_distinct(selected, ['MAXIMIZE'])
        self.assertEqual(ranked.study_key.tolist(), ['b', 'a'])
        self.assertEqual(ranked.trial.tolist(), [1, 0])
        self.assertEqual(ranked.model.iloc[0], variant['model'])

    def test_model_summary_ranks_across_frameworks_and_counts_sampler_runs(self):
        rows = []
        for key, framework, sampler, metric, scores in [
            ('a', 'vllm', 'GP', 'output_throughput', [8, 9]),
            ('b', 'sglang', 'GP', 'output_throughput', [10]),
            ('c', 'trtllm', 'TPE', 'output_throughput', [7]),
            ('d', 'vllm', 'TPE', 'output_throughput', [None]),
            ('e', 'sglang', 'TPE', 'p95_e2e_latency_ms', [42]),
            ('f', 'vllm', 'GP', 'p95_e2e_latency_ms', [50]),
        ]:
            for trial, score in enumerate(scores):
                rows.append(dict(context=('Initial runs', 'm', framework, 8, metric),
                                 section='Initial runs', model='m', framework=framework,
                                 concurrency=8, sampler=sampler, study_key=key, display_name=key,
                                 trial=trial, values=[score], candidate=score is not None))
        rankings, samplers = a.model_summary_tables(pd.DataFrame(rows))
        self.assertEqual(rankings.study_key.tolist(), ['b', 'a', 'c', 'e', 'f'])
        self.assertEqual(rankings['rank'].tolist(), [1, 2, 3, 1, 2])
        throughput = samplers[samplers.metric.eq('output_throughput')]
        self.assertEqual(throughput.sampler.tolist(), ['GP', 'TPE'])
        self.assertEqual(throughput.runs.tolist(), [2, 2])
        self.assertEqual(throughput.runs_with_result.tolist(), [2, 1])

    def test_native_heading_and_summary_outputs(self):
        from unittest.mock import patch
        frame = pd.DataFrame({'completed': [12]})
        with patch.object(a, 'display') as display:
            a.heading(6, 'Output throughput (maximize)')
            a.table(frame)
        self.assertIsInstance(display.call_args_list[0].args[0], a.Markdown)
        self.assertEqual(display.call_args_list[0].args[0].data, '###### Output throughput (maximize)')
        pd.testing.assert_frame_equal(display.call_args_list[1].args[0], frame)

    def test_duration_boundary(self):
        for value, expected in [(299, False), (300, True), (301, True), (None, False), (float('nan'), False)]:
            self.assertEqual(a.eligible_duration(value), expected)

    def test_distinct_top_five_and_directions(self):
        f = pd.DataFrame([{'study_key': str(i), 'trial': n, 'values': [i * 10 + n]}
                          for i in range(7) for n in range(3)])
        lo = a.best_distinct(f, ['MINIMIZE'])
        hi = a.best_distinct(f, ['MAXIMIZE'])
        self.assertEqual(len(lo), 5); self.assertTrue(lo.study_key.is_unique)
        self.assertEqual(lo.trial.tolist(), [0] * 5)
        self.assertEqual(hi.study_key.iloc[0], '6')
        self.assertEqual(hi.trial.tolist(), [2] * 5)
        self.assertEqual(len(a.best_distinct(f[f.study_key.eq('0')], ['MINIMIZE'])), 1)

    def test_pareto_preserves_pairs(self):
        f = pd.DataFrame({'study_key': ['a', 'b', 'c'], 'trial': [0, 0, 0], 'values': [[1, 9], [9, 1], [10, 10]]})
        self.assertEqual(a.best_distinct(f, ['MINIMIZE', 'MINIMIZE']).study_key.tolist(), ['a', 'b'])

    def test_incomplete_phases_and_overlapping_logs(self):
        e = {'status': 'failed', 'timings_s': {'launch_to_ready': 7, 'benchmark': 3},
             'phase_status': {'launch_to_ready': 'complete', 'benchmark': 'incomplete'},
             'startup_observations': [{'phase': 'compilation', 'seconds': 5}, {'phase': 'compilation', 'seconds': 5}]}
        rows = a.timing_records(e, {'attempt_id': 'x'})
        self.assertEqual([r['status'] for r in rows[:2]], ['complete', 'incomplete'])
        self.assertEqual(len([r for r in rows if r['kind'] == 'engine-log observation']), 2)
        self.assertEqual(rows[0]['value'], 7)  # log observations are never added to launch time

    def test_pressure_flags_coverage_and_zero(self):
        e = {'phase_status': {'benchmark': 'complete'}, 'server_telemetry': {'measured_window_s': 10,
             'metrics': {'running_requests': {'available': False, 'coverage': 1, 'time_weighted_avg': 7},
                         'queued_requests': {'available': True, 'coverage': .5, 'time_weighted_avg': 0, 'peak': 0},
                         'gpu_utilization': {'available': True, 'coverage': 0, 'time_weighted_avg': 99}}}}
        rows = {r['metric']: r for r in a.pressure_records(e, {})}
        self.assertIsNone(rows['running_requests']['time_weighted_avg'])
        self.assertIsNone(rows['gpu_utilization']['time_weighted_avg'])
        self.assertEqual(rows['queued_requests']['time_weighted_avg'], 0)
        self.assertEqual(rows['queued_requests']['weight_s'], 5)
        self.assertIsNone(rows['kv_cache_utilization']['time_weighted_avg'])

    def test_trial_stats_preserve_seed_and_gaps(self):
        f = pd.DataFrame({'context': ['c'] * 3, 'seed': [11, 11, 29], 'trial': [0, 2, 0],
                          'study_key': ['a', 'a', 'b'], 'value': [2., 4., 8.]})
        out = a.statistics(f, ['context', 'seed', 'trial'])
        self.assertEqual(len(out), 3)
        self.assertNotIn(1, out.trial.tolist())
        self.assertTrue(out.sample_variance.isna().all())
        self.assertTrue(out.contributing_runs.eq(1).all())

    def test_hierarchy_selection_and_closed_toggle(self):
        r = a.Report.__new__(a.Report)
        r.runs = pd.DataFrame({'section': ['Initial runs', 'Initial runs', 'Parameters probe'],
                              'model': ['m'] * 3, 'framework': ['vllm'] * 3, 'concurrency': [1, 8, 8]})
        self.assertEqual(len(r.select('Initial runs', 'm', 'vllm', 8)), 1)
        events = []
        original_heading = a.heading
        a.heading = lambda level, text: events.append((level, text))
        r.show_model_summary = lambda section, model: events.append(('summary', model))
        r.show_concurrency = lambda section, model, framework, concurrency: events.append((5, concurrency))
        try:
            r.show_section('Initial runs')
        finally:
            a.heading = original_heading
        self.assertEqual([x[0] for x in events], [2, 3, 'summary', 4, 5, 5])
        output = []; original_display = a.display
        a.display = lambda value: output.append(value.data)
        try:
            a.closed_toggle('Show all runs (2)', '<table></table>')
        finally:
            a.display = original_display
        self.assertIn('<details><summary>', output[0]); self.assertNotIn(' open', output[0])

    def test_compact_details_retain_partial_phases_and_unavailable_flags(self):
        base = dict(context='c', study_key='s', trial=4, attempt_id='a', source='trial.execution')
        execution = {'status': 'failed', 'timings_s': {'launch_to_ready': 6, 'benchmark': 2},
                     'phase_status': {'launch_to_ready': 'complete', 'benchmark': 'partial'},
                     'startup_observations': [{'phase': 'compilation', 'seconds': 1},
                                              {'phase': 'compilation', 'seconds': 1}],
                     'server_telemetry': {'measured_window_s': 2, 'metrics': {
                         'gpu_utilization': {'available': False, 'reason': 'unsupported', 'coverage': 0}}}}
        wall, logs, telemetry = a.compact_attempt_details(
            pd.DataFrame(a.timing_records(execution, base)), pd.DataFrame(a.pressure_records(execution, base)))
        self.assertEqual(len(wall), 1)
        self.assertEqual(wall.iloc[0].recorded_wall_phases['benchmark']['status'], 'partial')
        self.assertEqual(len(logs), 2)
        self.assertEqual(len(telemetry), 1)
        self.assertFalse(telemetry.iloc[0].recorded_metrics['gpu_utilization']['available'])
        self.assertFalse(telemetry.iloc[0].benchmark_complete)

    def test_telemetry_summary_weights_and_excludes_incomplete_windows(self):
        rows = []
        for identity, avg, window, coverage, complete in [('a', 2, 10, .5, True),
                ('b', 10, 2, 1, True), ('partial', 99, 10, 1, False)]:
            base = dict(section='Parameters probe', model='m', framework='vllm', concurrency=8,
                        notebook_version='v3', context='c', study_key=identity)
            execution = {'phase_status': {'benchmark': 'complete' if complete else 'partial'},
                'server_telemetry': {'measured_window_s': window, 'semantics': 'compatible', 'metrics': {
                    'running_requests': {'available': True, 'coverage': coverage, 'time_weighted_avg': avg,
                        'peak': avg + 1, 'sample_count': 2, 'aggregation': 'sum'}}}}
            rows.extend(a.pressure_records(execution, base))
        report = a.Report.__new__(a.Report); report.pressure = pd.DataFrame(rows)
        summary = report._pressure_summary()
        metric = summary[summary.metric.eq('running_requests')].iloc[0]
        self.assertAlmostEqual(metric.time_weighted_avg, 30 / 7)
        self.assertEqual(metric.peak, 11)
        self.assertEqual(metric.usable_attempts, 2)
        self.assertEqual(metric.incomplete_benchmark_attempts, 1)
        self.assertTrue(summary[summary.metric.eq('gpu_utilization')].time_weighted_avg.isna().all())

    def test_importance_dependency_failure_is_an_explicit_skip(self):
        import json
        from unittest.mock import patch
        distribution = json.loads(a.optuna.distributions.distribution_to_json(a.optuna.distributions.FloatDistribution(0, 1)))
        run = {'key': 's', 'context': 'c', 'framework': 'vllm', 'model': 'm', 'notebook_version': 'v2',
               'directions': ['MAXIMIZE'], 'contract': {'metrics': ['output_throughput']},
               'trials': [{'state': 'COMPLETE', 'values': [float(i)], 'attrs': {},
                           'params': {'x': i / 20}, 'distributions': {'x': distribution}} for i in range(12)]}
        short_before = {**run, 'key': 'short-before', 'trials': run['trials'][:2]}
        short_after = {**run, 'key': 'short-after', 'trials': run['trials'][:2]}
        paired = {**run, 'key': 'paired', 'directions': ['MAXIMIZE', 'MINIMIZE'],
                  'contract': {'metrics': ['first', 'second']},
                  'trials': [{**t, 'values': [t['values'][0], 100 - t['values'][0]]} for t in run['trials']]}
        data = SimpleNamespace(main_studies=[short_before, run, paired, short_after], MIN_IMPORTANCE_TRIALS=10, FANOVA_TREES=32, IMPORTANCE_SEED=2026)
        with patch.object(a, 'FanovaImportanceEvaluator', side_effect=ImportError('missing optional dependency')), \
                patch.object(a, 'PedAnovaImportanceEvaluator', side_effect=ImportError('missing optional dependency')):
            importance, statuses, parameters = a.compute_importance(data)
        self.assertTrue(importance.empty)
        self.assertEqual(set(statuses.method), {'fANOVA', 'PED-ANOVA'})
        self.assertTrue(statuses[statuses.study_key.eq('s')].status.str.contains('ImportError: missing optional dependency').all())
        self.assertEqual(len(statuses), 10)
        self.assertEqual(len(statuses[statuses.study_key.eq('paired')]), 4)
        self.assertFalse(statuses.duplicated(['study_key', 'metric', 'method']).any())
        self.assertTrue(statuses[statuses.study_key.str.startswith('short-')].status.eq('insufficient completed candidates').all())

    def test_attempt_id_dedup_and_no_row_order_join(self):
        trial = {'number': 7, 'attrs': {'execution': {'attempt_id': 'one', 'status': 'complete',
            'timings_s': {'benchmark': 1}}, 'measurement': {'execution': {'attempt_id': 'one', 'status': 'complete',
            'timings_s': {'benchmark': 1}}}}}
        run = {'key': 's', 'trials': [trial], 'copies': [], 'contract': {}, 'context': 'c',
               'section': 'Parameters probe', 'model': 'm', 'framework': 'vllm', 'notebook_version': 'v3'}
        timings, pressure, attempts, schema = a.extract_execution(SimpleNamespace(main_studies=[run], unique_attempts=pd.DataFrame(), lookup={'s': run}))
        self.assertEqual(len(attempts), 1)
        self.assertEqual(timings.trial.tolist(), [7])
        self.assertFalse(schema.is_v3.any())  # version-like notebook labels are not schema evidence


if __name__ == '__main__':
    unittest.main()
