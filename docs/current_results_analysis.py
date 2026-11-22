"""Read-only November 17 report analysis and hierarchical notebook presentation.

Collection normalization and compatibility safeguards retained from the original report.
No network access, experiment launches, or writes to source artifacts.
"""
from pathlib import Path
from collections import Counter, defaultdict
from contextlib import closing
import hashlib, html, importlib.metadata, json, math, re, sqlite3, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, HTML
import optuna
from optuna.distributions import json_to_distribution
from optuna.importance import FanovaImportanceEvaluator, PedAnovaImportanceEvaluator


from types import SimpleNamespace

def load_collection(root):
    ROOT = Path(root).resolve()
    RESULTS = ROOT / 'runs/20261117-143416'
    NOTEBOOKS = ROOT / 'experiments/notebooks/20261117-143416'
    MIN_RUN_SECONDS = 300
    # Empty sets mean all. Filters affect analysis, never the raw inventory.
    FRAMEWORK_FILTER = set()  # {'sglang', 'vllm', 'trtllm'}
    MODEL_FILTER = set()      # exact contract target_model, including any @artifact
    VERSION_FILTER = set()   # resolved version labels shown in inventory
    MIN_IMPORTANCE_TRIALS = 10
    FANOVA_TREES = 32
    IMPORTANCE_SEED = 2026
    # Provide only externally verified mappings; collection folder names are not VM evidence.
    VERIFIED_VM_BY_ORIGINAL_RUN = {}  # original run ID: {'vm': '...', 'evidence': '...'}

    def canonical(x):
        return json.dumps(x, sort_keys=True, separators=(',', ':'), default=str)
    def digest(x):
        return hashlib.sha256(canonical(x).encode()).hexdigest()
    def relative(p):
        return str(p.relative_to(ROOT))
    def load_json(p):
        try:
            return json.loads(p.read_text())
        except (OSError, ValueError) as exc:
            issues.append({'source': relative(p), 'issue': str(exc)})
            return {}
    def sanitize(model):
        # Forward-only match, identical to rename_local_runs; never invert hyphens.
        return re.sub(r'[^A-Za-z0-9._-]+', '-', str(model).split('@', 1)[0]).strip('-')
    def seconds(x):
        try:
            v = float(x)
            return v if math.isfinite(v) and v >= 0 else None
        except (ValueError, TypeError):
            return None

    def duration_for(attrs, trials):
        duration = seconds(attrs.get('optimization_wall_seconds'))
        if duration is not None:
            return duration, 'study optimization_wall_seconds'
        # Same conservative fallback as optuna_run_metadata: all trials must be terminal.
        if trials and all(t['state'] in {'COMPLETE', 'FAIL', 'PRUNED'} for t in trials):
            starts = [pd.Timestamp(t['start']) for t in trials if t['start']]
            ends = [pd.Timestamp(t['end']) for t in trials if t['end']]
            if starts and ends:
                return seconds((max(ends) - min(starts)).total_seconds()), 'terminal trial timestamp span'
        return None, 'unknown: no reliable optimization duration or terminal span'
    issues, raw_studies, folder_rows, notebook_rows = [], [], [], []
    notebook_texts = {}
    for p in sorted(NOTEBOOKS.glob('*.ipynb')):
        nb = load_json(p)
        outputs = '\n'.join(canonical(o) for c in nb.get('cells', []) for o in c.get('outputs', []))
        notebook_texts[relative(p)] = outputs
        notebook_rows.append({'notebook': relative(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                              'code_cells': sum(c['cell_type']=='code' for c in nb.get('cells', [])),
                              'executed_cells': sum(c.get('execution_count') is not None for c in nb.get('cells', []))})

    def notebook_version(original, contract):
        matches = [p for p, output in notebook_texts.items() if original in output]
        # A saved output is evidence of execution, not proof of a VM identity.
        if len(matches) == 1:
            return Path(matches[0]).stem, 'original run ID in saved output', matches
        env = contract.get('environment', {})
        adapter = env.get('notebook_compatibility', {}).get('adapter', '')
        for fw in ['sglang', 'vllm', 'trtllm']:
            if env.get(fw + '_v3') or original.startswith('phase1-' + fw + '-v3-'):
                return fw + '_v3 [contract/campaign]', adapter or 'campaign ID', matches
        if adapter:
            return adapter + ' [adapter; notebook file unconfirmed]', 'contract adapter', matches
        return 'unknown notebook version', 'no unique execution/adapter evidence', matches

    paths = sorted(set(RESULTS.rglob('*.sqlite3')) | set(RESULTS.rglob('*.db')) | set(RESULTS.rglob('*.sqlite')))
    folders = sorted({p.parent for p in paths} | {p.parent for p in RESULTS.rglob('config.json')})
    source_hashes = {relative(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    for folder in folders:
        collection = folder.relative_to(RESULTS).parts[0]
        original, marker, suffix = folder.name.partition('__model-')
        cfg = load_json(folder / 'config.json') if (folder / 'config.json').exists() else {}
        folder_rows.append({'folder': relative(folder), 'collection': collection, 'original_run': original,
                            'suffix': suffix or '(legacy: no suffix)', 'config_present': bool(cfg),
                            'databases': sum(p.parent == folder for p in paths),
                            'config_models': sorted(set(cfg.get('settings',{}).get('phase_models',{}).values())),
                            'config_frameworks': cfg.get('settings',{}).get('frameworks',[])})
        for database in [p for p in paths if p.parent == folder]:
            try:
                with closing(sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True)) as db:
                    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                    if not {'studies', 'trials', 'study_user_attributes'} <= tables:
                        issues.append({'source': relative(database), 'issue': 'not an Optuna database'})
                        continue
                    def attrs(table, column, identity):
                        if table not in tables:
                            return {}
                        return {k: json.loads(v) for k,v in db.execute(
                            f'SELECT key,value_json FROM {table} WHERE {column}=? ORDER BY key', (identity,))}
                    for sid, name in db.execute('SELECT study_id,study_name FROM studies ORDER BY study_id'):
                        a = attrs('study_user_attributes', 'study_id', sid)
                        contract = a.get('contract', {})
                        ts = []
                        for tid, number, state, start, end in db.execute(
                            'SELECT trial_id,number,state,datetime_start,datetime_complete FROM trials WHERE study_id=? ORDER BY number', (sid,)):
                            params, distributions = {}, {}
                            for key, value, dist_json in db.execute('SELECT param_name,param_value,distribution_json FROM trial_params WHERE trial_id=? ORDER BY param_name', (tid,)):
                                dist = json_to_distribution(dist_json)
                                params[key] = dist.to_external_repr(value)
                                distributions[key] = json.loads(dist_json)
                            values = [v for _,v in db.execute('SELECT objective,value FROM trial_values WHERE trial_id=? ORDER BY objective', (tid,))]
                            intermediates = list(db.execute('SELECT step,intermediate_value FROM trial_intermediate_values WHERE trial_id=? ORDER BY step', (tid,)))
                            ts.append(dict(number=number, state=state, start=start, end=end, params=params,
                                distributions=distributions, values=values, attrs=attrs('trial_user_attributes','trial_id',tid),
                                system=attrs('trial_system_attributes','trial_id',tid), intermediates=intermediates))
                        directions = [v for _,v in db.execute('SELECT objective,direction FROM study_directions WHERE study_id=? ORDER BY objective',(sid,))]
                        env = contract.get('environment', {})
                        model = contract.get('target_model') or env.get('target_model')
                        config_model = cfg.get('settings',{}).get('phase_models',{}).get(contract.get('phase'), cfg.get('target_model'))
                        if not model:
                            model = config_model or 'unresolved suffix:' + (suffix or 'legacy')
                        mismatch = []
                        if config_model and config_model != model:
                            mismatch.append('config phase model differs from contract')
                        if suffix and sanitize(model) != suffix:
                            mismatch.append('suffix differs from sanitized contract base model')
                        if env.get('target_model') and env['target_model'] != model:
                            mismatch.append('environment target_model differs from contract')
                        if contract.get('directions') != directions:
                            mismatch.append('contract directions differ from SQLite directions')
                        fw = contract.get('framework') or env.get('framework') or 'unknown'
                        if env.get('framework') and env['framework'] != fw:
                            mismatch.append('environment framework differs from contract')
                        version, version_source, notebooks = notebook_version(original, contract)
                        vm_fields = {k:env[k] for k in ['source_vm','vm_name','hostname','instance_id'] if env.get(k)}
                        verified = VERIFIED_VM_BY_ORIGINAL_RUN.get(original)
                        vm = verified['vm'] if verified else canonical(vm_fields) if vm_fields else 'unknown VM (unverified)'
                        vm_evidence = verified['evidence'] if verified else canonical(vm_fields) if vm_fields else 'no VM identifier in metadata; collection is not VM evidence'
                        duration, duration_source = duration_for(a, ts)
                        provenance = a.get('wandb',{}).get('url') or a.get('wandb',{}).get('id') or original
                        snapshot = digest({'provenance':provenance, 'contract':contract, 'attrs':a,
                                           'study_system':attrs('study_system_attributes','study_id',sid),
                                           'trials':ts,'directions':directions})
                        raw_studies.append(dict(source=relative(database), folder=relative(folder), collection=collection,
                            original_run=original, suffix=suffix or '(legacy)', study_name=name, study_id=sid,
                            contract=contract, attrs=a, trials=ts, directions=directions, model=model,
                            config_model=config_model, framework=fw, notebook_version=version, version_source=version_source,
                            notebooks=notebooks, vm=vm, vm_evidence=vm_evidence, duration_seconds=duration,
                            duration_source=duration_source, mismatch='; '.join(mismatch), provenance=provenance,
                            snapshot=snapshot, key=digest([snapshot,suffix])[:16]))
            except (sqlite3.Error, ValueError, KeyError, TypeError) as exc:
                issues.append({'source':relative(database), 'issue':repr(exc)})

    # Exact duplicates collapse across collections, retaining all source references.
    by_key = defaultdict(list)
    for r in raw_studies:
        by_key[r['key']].append(r)
    unique_studies = []
    for key, copies in by_key.items():
        r = dict(copies[0])
        r['copies'] = copies
        r['sources'] = sorted({c['source'] for c in copies})
        r['collections'] = sorted({c['collection'] for c in copies})
        unique_studies.append(r)
    by_provenance = defaultdict(list)
    for r in unique_studies:
        by_provenance[(r['provenance'],r['suffix'])].append(r)
    for group in by_provenance.values():
        for r in group:
            r['snapshot_conflict'] = len(group) > 1

    inventory = pd.DataFrame([{k:r[k] for k in ['key','source','folder','collection','original_run','suffix',
        'study_name','model','config_model','framework','notebook_version','version_source','vm','vm_evidence',
        'duration_seconds','duration_source','mismatch','provenance']} | {'trials':len(r['trials'])} for r in raw_studies])
    folder_inventory = pd.DataFrame(folder_rows)
    folder_inventory['study_records'] = folder_inventory.folder.map(Counter(r['folder'] for r in raw_studies)).fillna(0).astype(int)
    folder_inventory['actual_frameworks'] = folder_inventory.folder.map(
        lambda f: sorted({r['framework'] for r in raw_studies if r['folder']==f}))
    folder_inventory['inventory_status'] = folder_inventory.study_records.map(
        lambda count: 'study data available' if count else 'no Optuna studies: no analyzable experiment or verified duration')

    print(f'Normalized {len(raw_studies)} raw studies in {len(paths)} databases; streaming measured artifacts…', flush=True)
    # Stream full records; retain hashes and scalar metadata rather than large per-request arrays.
    attempt_rows = []
    for folder in folders:
        local_studies = [r for r in raw_studies if r['folder']==relative(folder)]
        for p in sorted(folder.glob('*measured*.jsonl')):
            candidates = [r for r in local_studies if p.name.startswith(r['study_name'] + '-')]
            linked = max(candidates, key=lambda r:len(r['study_name'])) if candidates else None
            occurrences = Counter()
            with p.open() as stream:
                for line_no,line in enumerate(stream,1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        fingerprint = digest(record)
                        occurrences[fingerprint] += 1
                        metadata = record.get('metadata',{})
                        results = record.get('results',{})
                        # Preserve repeated identical rows inside a file using an occurrence index.
                        identity = digest([linked['key'] if linked else relative(p), fingerprint, occurrences[fingerprint]])
                        attempt_rows.append({'source':relative(p),'line':line_no,'study_key':linked['key'] if linked else None,
                            'attempt_key':identity,'valid':results.get('benchmark_valid'),
                            'trial_number':metadata.get('trial_number'),'finished_at':metadata.get('finished_at'),
                            'execution':metadata.get('execution') or record.get('execution') or results.get('execution'),
                            'framework':record.get('config',{}).get('framework'), 'parse_error':False})
                    except (ValueError, TypeError) as exc:
                        issues.append({'source':relative(p),'issue':f'line {line_no}: {exc}'})
                        attempt_rows.append({'source':relative(p),'line':line_no,'study_key':None,
                            'attempt_key':relative(p)+':'+str(line_no),'valid':None,'parse_error':True})
    attempts = pd.DataFrame(attempt_rows, columns=['source','line','study_key','attempt_key','valid','trial_number','finished_at','framework','parse_error','execution'])
    unique_attempts = attempts.drop_duplicates('attempt_key')

    def counts(records):
        ts = [t for r in records for t in r['trials']]
        return {'studies / experiment runs':len(records),'trials':len(ts),
                'completed trials':sum(t['state']=='COMPLETE' for t in ts)}
    raw_counts, dedup_counts = counts(raw_studies), counts(unique_studies)
    # Empty databases/directories have no study provenance proving an independent run
    # or a duplicate download. Keep each separately in artifact counts.
    folder_signatures = []
    for f in folders:
        records = [r for r in raw_studies if r['folder']==relative(f)]
        if not records:
            folder_signatures.append(('unverified empty artifact',relative(f)))
            continue
        config = load_json(f/'config.json') if (f/'config.json').exists() else None
        folder_signatures.append(('verified content',digest({
            'original':f.name.partition('__model-')[0], 'suffix':f.name.partition('__model-')[2],
            'config':config, 'studies':sorted(r['key'] for r in records),
            'attempts':sorted(a['attempt_key'] for a in attempt_rows if Path(a['source']).parent==f.relative_to(ROOT))})))
    database_signatures = []
    for p in paths:
        keys = sorted(r['key'] for r in raw_studies if r['source']==relative(p))
        database_signatures.append(('verified studies',tuple(keys)) if keys else ('unverified empty artifact',relative(p)))
    coverage = pd.DataFrame([{'stage':'raw','run folders':len(folders),'databases':len(paths),**raw_counts,
        'measured records':len(attempts)}, {'stage':'deduplicated snapshots',
        'run folders':len(set(folder_signatures)), 'databases':len(set(database_signatures)),
        **dedup_counts,'measured records':len(unique_attempts)}])

    def duration_status(r):
        if r['duration_seconds'] is None:
            return 'unknown duration'
        return 'under 300 seconds' if r['duration_seconds'] < MIN_RUN_SECONDS else 'at least 300 seconds'
    def extra_reasons(r):
        c = r['contract']; env = c.get('environment',{})
        reasons = []
        if r['snapshot_conflict']: reasons.append('conflicting/overlapping snapshots of same provenance')
        if r['mismatch']: reasons.append('unresolved metadata mismatch')
        if env.get('measurement') != 'MEASURED': reasons.append('synthetic or measurement status unverified')
        if r['framework'] not in {'sglang','vllm','trtllm'}: reasons.append('unknown framework')
        if not c.get('metrics') or not c.get('workload') or len(c.get('metrics',[])) != len(r['directions']):
            reasons.append('missing workload/objective contract')
        if FRAMEWORK_FILTER and r['framework'] not in FRAMEWORK_FILTER: reasons.append('framework filter')
        if MODEL_FILTER and r['model'] not in MODEL_FILTER: reasons.append('model filter')
        if VERSION_FILTER and r['notebook_version'] not in VERSION_FILTER: reasons.append('version filter')
        return reasons
    for r in unique_studies:
        r['exclusions'] = ([duration_status(r)] if duration_status(r) != 'at least 300 seconds' else []) + extra_reasons(r)

    excluded = [r for r in unique_studies if r['exclusions']]
    included = [r for r in unique_studies if not r['exclusions']]
    exclusion_inventory = pd.DataFrame([{'key':r['key'],'study':r['study_name'],'model':r['model'],
        'framework':r['framework'],'duration_seconds':r['duration_seconds'],'duration_source':r['duration_source'],
        'reasons':'; '.join(r['exclusions']),'sources':r['sources']} for r in excluded])

    print(f'Reconciled {len(unique_studies)} studies and {len(unique_attempts)} measured records; matching W&B names…', flush=True)
    reconcile_names(ROOT, unique_studies, issues)
    CONTEXT_FIELDS = ['phase','concurrency','workload','target','metrics','directions','constrained',
        'ttft_limit_ms','e2e_limit_ms','space_kind','phase_a_space','expanded_choices','sampling_menu',
        'pruner_class','pruner_settings','budget_trials','budget_gpu_seconds','checkpoint_requests']
    ENV_FIELDS = ['gpu_type','gpu_count','model_revision','tokenizer_revision','framework_version','cuda_version',
        'driver_version','notebook_compatibility','fixed_server','weight_artifact','verified_precision_combinations',
        'dataset_identity','row_indices','requests_sha256','package_versions','sglang_v3','vllm_v3','trtllm_v3']
    contexts, trial_rows = {}, []
    for r in included:
        c = r['contract']; env = c.get('environment',{})
        observed = defaultdict(set)
        for t in r['trials']:
            for p,d in t['distributions'].items(): observed[p].add(canonical(d))
        context = {'section':r['section'],'model':r['model'],'suffix':r['suffix'],'framework':r['framework'],
            'notebook_version':r['notebook_version'],'vm':r['vm'],
            'unknown_vm_isolation':r['original_run'] if r['vm'].startswith('unknown') else None,
            'contract':{k:c.get(k) for k in CONTEXT_FIELDS},
            'environment':{k:env.get(k) for k in ENV_FIELDS},
            'observed_distributions':{k:sorted(v) for k,v in observed.items()}}
        ctx = report_group(r)
        r['context'] = ctx
        contexts[r['key']] = context
        for t in r['trials']:
            measurement = t['attrs'].get('measurement',{})
            if not isinstance(measurement,dict): measurement = {}
            metrics = measurement.get('results',measurement)
            row = dict(study_key=r['key'],context=ctx,model=r['model'],suffix=r['suffix'],framework=r['framework'],
                notebook_version=r['notebook_version'],vm=r['vm'],study=r['study_name'],trial=t['number'],
                state=t['state'],sampler=c.get('sampler'),seed=c.get('sampler_settings',{}).get('seed'),
                params=t['params'],config=t['attrs'].get('config'),values=t['values'],
                feasible=t['attrs'].get('slo_feasible'),duration_seconds=r['duration_seconds'])
            for m in ['output_throughput','request_throughput','p95_e2e_latency_ms','p95_ttft_ms']:
                row[m] = metrics.get(m, t['attrs'].get(m, measurement.get('execution', {}).get(m)))
            row['trial_duration_seconds'] = seconds((pd.Timestamp(t['end']) - pd.Timestamp(t['start'])).total_seconds()) if t['start'] and t['end'] else None
            row['candidate'] = (t['state']=='COMPLETE' and len(t['values'])==len(r['directions'])
                and all(isinstance(v,(int,float)) and math.isfinite(v) for v in t['values'])
                and (not c.get('constrained') or row['feasible'] is True))
            trial_rows.append(row)
    trials = pd.DataFrame(trial_rows)
    assert not trials.empty, 'No eligible trials; inspect exclusions or filters'
    main_studies = included
    lookup = {r['key']:r for r in main_studies}
    context_inventory = pd.DataFrame([{'study_key':key,'model':v['model'],'suffix':v['suffix'],'framework':v['framework'],
        'notebook_version':v['notebook_version'],'vm':v['vm'],'gpu':v['environment']['gpu_type'],
        'revision':v['environment']['model_revision'],'concurrency':v['contract']['concurrency'],
        'workload':v['contract']['workload'],'objective':v['contract']['target'],
        'metrics':v['contract']['metrics'],'directions':v['contract']['directions'],
        'definition':canonical(v)} for key,v in contexts.items()])

    return SimpleNamespace(**locals())


def compute_importance(data):
    main_studies = data.main_studies
    MIN_IMPORTANCE_TRIALS = data.MIN_IMPORTANCE_TRIALS
    FANOVA_TREES = data.FANOVA_TREES
    IMPORTANCE_SEED = data.IMPORTANCE_SEED
    importance_rows, importance_status, parameter_audit = [], [], []
    SUPPORTED = {'FloatDistribution','IntDistribution','CategoricalDistribution'}
    for r in main_studies:
        ts = [t for t in r['trials'] if t['state']=='COMPLETE' and len(t['values'])==len(r['directions'])
            and all(isinstance(v,(int,float)) and math.isfinite(v) for v in t['values'])
            and (not r['contract'].get('constrained') or t['attrs'].get('slo_feasible') is True)]
        names = sorted({p for t in ts for p in t['params']})
        usable=[]
        for p in names:
            present=[t for t in ts if p in t['params']]
            ds={canonical(t['distributions'][p]) for t in present}
            status='included'
            if len(present)!=len(ts): status='conditional: absent in some completed candidates'
            elif len(ds)!=1: status='changing distributions'
            elif any(t['distributions'][p]['name'] not in SUPPORTED for t in present):status='unsupported distribution'
            elif len({canonical(t['params'][p]) for t in present})<2:status='constant observed values'
            elif json_to_distribution(next(iter(ds))).single():status='constant distribution'
            if status=='included':usable.append(p)
            parameter_audit.append({'study_key':r['key'],'context':r['context'],'parameter':p,
                'present':len(present),'completed_candidates':len(ts),'status':status})
        for i,metric in enumerate(r['contract']['metrics']):
            base={'study_key':r['key'],'context':r['context'],'framework':r['framework'],'model':r['model'],
                'notebook_version':r['notebook_version'],'metric':metric,'candidates':len(ts)}
            reason = ('insufficient completed candidates' if len(ts)<MIN_IMPORTANCE_TRIALS else
                      'no stable common varying parameters' if not usable else
                      'constant objective' if len({t['values'][i] for t in ts})<2 else None)
            if reason:
                for method in ['fANOVA','PED-ANOVA']:importance_status.append({**base,'method':method,'status':reason})
                continue
            temporary=optuna.create_study(direction='minimize')
            sign=1 if r['directions'][i]=='MINIMIZE' else -1
            temporary.add_trials([optuna.trial.create_trial(
                params={p:t['params'][p] for p in usable},
                distributions={p:json_to_distribution(canonical(t['distributions'][p])) for p in usable},
                value=sign*t['values'][i]) for t in ts])
            factories = {'fANOVA': lambda: FanovaImportanceEvaluator(n_trees=FANOVA_TREES, seed=IMPORTANCE_SEED),
                     'PED-ANOVA': lambda: PedAnovaImportanceEvaluator(target_quantile=.1, evaluate_on_local=True)}
            for method, factory in factories.items():
                try:
                    with warnings.catch_warnings(record=True) as captured:
                        warnings.simplefilter('always')
                        evaluator = factory()
                        scores = optuna.importance.get_param_importances(temporary, evaluator=evaluator, params=usable)
                    if not scores or not all(math.isfinite(v) for v in scores.values()) or sum(scores.values()) <= 0:
                        raise ValueError('degenerate importance; no usable finite normalized estimate')
                    importance_status.append({**base, 'method': method, 'status': 'computed',
                        'warnings': '; '.join(str(w.message) for w in captured)})
                    for p, score in scores.items():
                        importance_rows.append({**base, 'method': method, 'parameter': p, 'importance': score})
                except (ValueError,RuntimeError,ImportError,TypeError,ZeroDivisionError) as exc:
                    importance_status.append({**base,'method':method,'status':f'{type(exc).__name__}: {exc}'})
    importance=pd.DataFrame(importance_rows)
    importance_status=pd.DataFrame(importance_status)
    parameter_audit=pd.DataFrame(parameter_audit)

    return importance, importance_status, parameter_audit

def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), default=str)

def seconds(value):
    try:
        value = float(value)
        return value if math.isfinite(value) and value >= 0 else None
    except (ValueError, TypeError):
        return None


def classify_names(evidence):
    """Only complete, ID-matched names constitute classification evidence."""
    names = {e['name'].strip() for e in evidence if isinstance(e.get('name'), str) and e['name'].strip()}
    if len(names) != 1:
        return 'unclassified', None, 'conflicting full names' if names else 'full name unavailable'
    name = next(iter(names))
    return ('Parameters probe' if 'invest' in name.lower() else 'Initial runs'), name, None


def reconcile_names(root, studies, issues):
    by_id = defaultdict(list)
    files = [root / 'experiment_results/wandb_run_snapshot.json', *sorted((root / 'results/sorted_results').glob('*.json'))]
    for path in files:
        if not path.exists():
            continue
        try:
            export = json.loads(path.read_text())
            for row in export.get('runs', []):
                if row.get('id') and row.get('name'):
                    by_id[row['id']].append({'name': row['name'], 'source': str(path.relative_to(root)),
                        'exported_at': export.get('exported_at'), 'url': row.get('url')})
        except (ValueError, TypeError) as exc:
            issues.append({'source': str(path.relative_to(root)), 'issue': str(exc)})
    wanted = {r['attrs'].get('wandb', {}).get('id') for r in studies}
    # These are the November collection's actual cached W&B names, rather than
    # older wandb.ai exports. Match the parsed RunRecord ID, not the directory name.
    for path in sorted((root / 'runs/20261117-143416').rglob('*.wandb')):
        if path.stem.removeprefix('run-') not in wanted:
            continue
        try:
            history = defaultdict(list)
            for identity, name in cached_wandb_names(path):
                history[identity].append(name)
            for identity, names in history.items():
                if identity in wanted:
                    by_id[identity].append({'name': names[-1], 'source': str(path.relative_to(root)) + ':latest RunRecord.display_name',
                                           'prior_names': names[:-1], 'url': None})
        except (OSError, ValueError) as exc:
            issues.append({'source': str(path.relative_to(root)), 'issue': str(exc)})
    for record in studies:
        wb = record['attrs'].get('wandb', {})
        identity = wb.get('id') or (wb.get('url') or '').rstrip('/').split('/')[-1]
        evidence = [json.loads(item) for item in sorted({canonical(e) for e in by_id.get(identity, [])})]
        # Explicit saved W&B display name is valid; study_name and RUN_NAME are not.
        if wb.get('name'):
            evidence.append({'name': wb['name'], 'source': record['source'] + ':study.wandb.name'})
        section, name, reason = classify_names(evidence)
        record.update(section=section, display_name=name, classification_reason=reason,
                      name_evidence=evidence, name_source=sorted({e['source'] for e in evidence}),
                      wandb_id=identity, wandb_url=wb.get('url') or next((e['url'] for e in evidence if e.get('url')), None))


UNITS = {'output_throughput': 'output tokens/s', 'request_throughput': 'requests/s',
         'p95_e2e_latency_ms': 'ms', 'p95_ttft_ms': 'ms'}
PRESSURE = {'running_requests': 'requests', 'queued_requests': 'requests',
            'client_requests_in_flight': 'requests', 'kv_cache_utilization': 'fraction',
            'gpu_utilization': 'percent'}


def eligible_duration(value):
    value = seconds(value)
    return value is not None and value >= 300


def frontier(frame, directions):
    if frame.empty:
        return frame
    values = np.array(frame['values'].tolist(), dtype=float)
    loss = values * np.array([1 if d == 'MINIMIZE' else -1 for d in directions])
    return frame.loc[[not np.any(np.all(loss <= row, axis=1) & np.any(loss < row, axis=1)) for row in loss]]


def best_distinct(frame, directions, limit=5):
    if frame.empty:
        return frame
    if len(directions) > 1:
        return frontier(frame, directions)  # paired, never scalar ranked
    frame = frame.assign(objective=frame['values'].map(lambda v: v[0]))
    return (frame.sort_values(['objective', 'study_key', 'trial'],
                             ascending=[directions[0] == 'MINIMIZE', True, True])
            .drop_duplicates('study_key').head(limit))


def timing_records(execution, base):
    """Recorded wall phases and individual, potentially overlapping log observations."""
    rows = []
    incomplete = execution.get('incomplete_phases', [])
    for phase, value in execution.get('timings_s', {}).items():
        status = execution.get('phase_status', {}).get(phase)
        # Runner/allocation/orchestration are totals and have no individual phase flag.
        if status is None:
            status = 'complete' if execution.get('status') == 'complete' else 'unknown'
        if phase in incomplete:
            status = 'incomplete'
        rows.append({**base, 'kind': 'wall phase', 'phase': phase, 'value': seconds(value),
                     'status': status, 'unit': 's', 'definition': 'execution.timings_s.' + phase})
    for index, observation in enumerate(execution.get('startup_observations', [])):
        rows.append({**base, 'kind': 'engine-log observation', 'phase': observation.get('phase'),
                     'value': seconds(observation.get('seconds')), 'status': 'complete', 'unit': 's',
                     'definition': observation.get('parser', 'unknown parser'), 'observation': index,
                     'line_number': observation.get('line_number'), 'log_source': observation.get('source')})
    return rows


def pressure_records(execution, base):
    telemetry = execution.get('server_telemetry') or {}
    metrics = dict(telemetry.get('metrics') or {})
    if telemetry.get('client_requests_in_flight'):
        client = telemetry['client_requests_in_flight']
        # Analytic coroutine integral: no scrape count or scrape coverage flag is recorded.
        metrics['client_requests_in_flight'] = {**client, 'available': client.get('time_weighted_avg') is not None,
            'aggregation': client.get('meaning'), 'coverage': None, 'sample_count': None}
    rows = []
    for metric in sorted(set(PRESSURE) | set(metrics)):
        item = metrics.get(metric, {})
        client = metric == 'client_requests_in_flight'
        available = item.get('available') is True
        coverage = seconds(item.get('coverage'))
        window = seconds(telemetry.get('measured_window_s'))
        # A gauge must have recorded positive coverage; analytic client integrals use their window.
        usable = available and window is not None and window > 0 and (client or (coverage is not None and coverage > 0))
        reason = item.get('reason') or (None if usable else 'server telemetry not recorded' if not telemetry else 'availability/coverage insufficient')
        rows.append({**base, 'metric': metric, 'available': available, 'usable': usable,
            'unit': item.get('unit', PRESSURE.get(metric)), 'sample_count': item.get('sample_count'),
            'measured_window_s': window, 'coverage_fraction': coverage,
            'weight_s': window * (1 if client else coverage) if usable else None,
            'aggregation': item.get('aggregation'), 'semantics': telemetry.get('semantics'),
            'method': 'analytic client integral' if client else 'sampled gauge, left hold up to 2 intervals',
            'time_weighted_avg': seconds(item.get('time_weighted_avg')) if usable else None,
            'peak': seconds(item.get('peak')) if usable else None, 'missing_reason': reason,
            'execution_complete': execution.get('phase_status', {}).get('benchmark') == 'complete',
            'collection_errors': telemetry.get('errors'), 'raw_measurement': item})
    return rows


def extract_execution(data):
    """Join by attempt ID or explicit trial identity; never use JSONL row order."""
    executions, identities, schema = {}, {}, []
    for r in data.main_studies:
        for t in r['trials']:
            attrs = t['attrs']; measurement = attrs.get('measurement') or {}
            for source, e in [('trial.execution', attrs.get('execution')), ('trial.measurement.execution', measurement.get('execution'))]:
                if not isinstance(e, dict):
                    continue
                aid = e.get('attempt_id') or 'trial-' + str(t['number'])
                identities[(r['key'], aid)] = t['number']
                executions[(r['key'], aid)] = (r, t['number'], source, e)
    # Reuse the streamed, deduplicated JSONL records from collection normalization.
    for row in data.unique_attempts.to_dict('records'):
        if row['study_key'] not in data.lookup:
            continue
        r = data.lookup[row['study_key']]; e = row.get('execution')
        if not isinstance(e, dict):
            continue
        aid = e.get('attempt_id')
        number = identities.get((r['key'], aid), e.get('trial_number'))
        if number is None and pd.notna(row.get('trial_number')):
            number = row['trial_number']
        if aid and number is not None:
            executions.setdefault((r['key'], aid), (r, number, row['source'], e))
    # Supplement only scoped, reconciled studies from matching attempt artifacts.
    for r in data.main_studies:
        for folder in sorted({data.ROOT / c['folder'] for c in r['copies']}):
            # attempt directory prefix records study identity; a trial still needs an ID join.
            for directory in folder.glob(r['study_name'] + '-*measured-attempts'):
                for p in directory.glob('*/execution.json'):
                    e = json.loads(p.read_text()); aid = e.get('attempt_id')
                    number = identities.get((r['key'], aid), e.get('trial_number'))
                    if aid and number is not None:
                        executions.setdefault((r['key'], aid), (r, number, str(p.relative_to(data.ROOT)), e))
                    elif aid:
                        data.issues.append({'source': str(p.relative_to(data.ROOT)), 'issue': 'attempt not assigned: no recorded trial identity or matching attempt ID'})
    timings, pressure, details = [], [], []
    for (key, aid), (r, number, source, e) in executions.items():
        c = r['contract']
        base = dict(study_key=key, trial=number, attempt_id=aid, source=source, context=r['context'],
                    section=r['section'], model=r['model'], framework=r['framework'], concurrency=c.get('concurrency'),
                    notebook_version=r['notebook_version'], is_v3=bool(c.get('environment', {}).get(r['framework'] + '_v3')), sampler=c.get('sampler'),
                    seed=c.get('sampler_settings', {}).get('seed'))
        timings.extend(timing_records(e, base)); pressure.extend(pressure_records(e, base))
        details.append({**base, 'status': e.get('status'), 'incomplete_phases': e.get('incomplete_phases'),
                        'sampled_config': e.get('sampled_config'), 'attempt_artifact': e.get('attempt_artifact'),
                        'request_errors': e.get('request_errors'),
                        'samples': (e.get('server_telemetry') or {}).get('samples', 'individual scrape samples not persisted')})
        schema.append({**base, 'execution_keys': sorted(e), 'timing_fields': sorted(e.get('timings_s', {})),
                       'has_server_telemetry': bool(e.get('server_telemetry'))})
    return pd.DataFrame(timings), pd.DataFrame(pressure), pd.DataFrame(details), pd.DataFrame(schema)


def statistics(frame, groups, value='value'):
    if frame.empty:
        return pd.DataFrame(columns=[*groups, 'mean', 'sample_variance', 'observations', 'contributing_runs'])
    return frame.groupby(groups, dropna=False).agg(mean=(value, 'mean'),
        sample_variance=(value, 'var'), observations=(value, 'count'), contributing_runs=('study_key', 'nunique')).reset_index()


def model_heading(model):
    """Group ModelOpt artifact variants for presentation; retain exact data IDs."""
    base, marker, variant = model.partition('@')
    return base if marker and variant.lower().startswith('modelop') else model


def objective_heading(metric):
    return {'output_throughput': 'Output throughput (maximize)',
            'p95_e2e_latency_ms': 'p95 E2E latency (minimize)'}.get(metric, metric)


def report_group(run):
    contract = run['contract']
    metric, = contract['metrics']
    direction, = run['directions']
    expected = {'output_throughput': 'MAXIMIZE', 'p95_e2e_latency_ms': 'MINIMIZE'}
    if metric in expected and direction != expected[metric]:
        raise ValueError(f'Unexpected direction for {metric}: {direction}')
    return (run['section'], model_heading(run['model']), run['framework'],
            contract.get('concurrency'), metric)


def group_rows(frame, group):
    return frame.loc[frame.context.map(lambda value: value == group).astype(bool)]


def model_summary_tables(trials):
    """Rank study winners across frameworks and count each eligible study once."""
    data = trials.assign(base_model=trials.model.map(model_heading),
                         metric=trials.context.map(lambda group: group[-1]))
    rankings, samplers = [], []
    groups = ['section', 'base_model', 'concurrency', 'metric']
    for _, rows in data.groupby(groups, dropna=False):
        metric = rows.metric.iloc[0]
        direction = 'MINIMIZE' if metric == 'p95_e2e_latency_ms' else 'MAXIMIZE'
        winners = best_distinct(rows[rows.candidate], [direction], limit=rows.study_key.nunique())
        winners = winners.reindex(columns=[*groups, 'study_key', 'display_name', 'model',
                                           'framework', 'trial', 'sampler', 'objective'])
        winners['unit'] = UNITS.get(metric, 'unit unavailable')
        top = winners.head(5).copy()
        top['rank'] = range(1, len(top) + 1)
        rankings.append(top)
        counts = rows.groupby('sampler', dropna=False).study_key.nunique().rename('runs')
        eligible_counts = winners.groupby('sampler', dropna=False).study_key.nunique().rename('runs_with_result')
        leaders = winners.drop_duplicates('sampler').set_index('sampler')
        sampler_rows = (counts.to_frame().join(eligible_counts).join(leaders[
            ['objective', 'framework', 'display_name']].rename(columns={'objective': 'best_result'}))
            .reset_index().sort_values('best_result', ascending=direction == 'MINIMIZE', kind='stable'))
        sampler_rows['runs_with_result'] = sampler_rows.runs_with_result.fillna(0).astype(int)
        for column in groups:
            sampler_rows[column] = rows[column].iloc[0]
        sampler_rows['unit'] = UNITS.get(metric, 'unit unavailable')
        samplers.append(sampler_rows)
    return pd.concat(rankings, ignore_index=True), pd.concat(samplers, ignore_index=True)


class Report:
    """Compute once, then pass complete section/model/framework/concurrency selections."""
    def __init__(self, root):
        print('Reading and reconciling November 17 collection…', flush=True)
        self.data = d = load_collection(root)
        print(f'Eligible: {len(d.main_studies)} studies; extracting telemetry…', flush=True)
        self.runs = pd.DataFrame([{k: r.get(k) for k in ['key', 'section', 'model', 'framework', 'notebook_version',
            'context', 'display_name', 'name_source', 'name_evidence', 'classification_reason', 'wandb_id', 'wandb_url',
            'study_name', 'duration_seconds', 'duration_source', 'vm', 'vm_evidence', 'sources', 'mismatch']} |
            {'concurrency': r['contract'].get('concurrency'), 'phase': r['contract'].get('phase'),
             'sampler': r['contract'].get('sampler'), 'pruner': r['contract'].get('pruner_class'),
             'seed': r['contract'].get('sampler_settings', {}).get('seed'), 'contract': r['contract']}
            for r in d.main_studies])
        self.trials = d.trials.merge(self.runs[['key', 'section', 'concurrency', 'pruner', 'display_name', 'wandb_url']],
                                    left_on='study_key', right_on='key', validate='many_to_one')
        objective = []
        for row in self.trials[self.trials.candidate].to_dict('records'):
            r = d.lookup[row['study_key']]
            for i, (metric, direction) in enumerate(zip(r['contract']['metrics'], r['directions'])):
                objective.append({**row, 'metric': metric, 'direction': direction,
                                  'unit': UNITS.get(metric, 'unit not recorded'), 'value': row['values'][i]})
        self.objectives = pd.DataFrame(objective, columns=[*self.trials.columns, 'metric', 'direction', 'unit', 'value'])
        self.objective_stats = statistics(self.objectives, ['context', 'metric', 'direction', 'unit', 'sampler', 'seed'])
        self.objective_overall = statistics(self.objectives, ['context', 'metric', 'direction', 'unit'])
        serving = self.trials.melt(id_vars=['context', 'study_key'],
            value_vars=['output_throughput', 'request_throughput', 'p95_e2e_latency_ms', 'p95_ttft_ms'],
            var_name='metric', value_name='value')
        serving['value'] = pd.to_numeric(serving.value, errors='coerce')
        serving = serving[np.isfinite(serving.value)]
        serving['unit'] = serving.metric.map(UNITS)
        self.serving_stats = statistics(serving, ['context', 'metric', 'unit'])
        self.mean_by_trial = statistics(self.objectives, ['context', 'metric', 'direction', 'unit', 'sampler', 'seed', 'trial'])
        self.best = {}
        for ctx, frame in self.trials.groupby('context'):
            self.best[ctx] = best_distinct(frame[frame.candidate], d.lookup[frame.study_key.iloc[0]]['directions'])
        ranking_columns = ['context', 'display_name', 'study_key', 'model', 'trial', 'objective',
                           'sampler', 'seed', 'duration_seconds']
        self.rankings = pd.concat(list(self.best.values())).reindex(columns=ranking_columns)
        self.rankings['unit'] = self.rankings.context.map(lambda group: UNITS.get(group[-1], 'unit unavailable'))
        self.rankings.insert(1, 'rank', self.rankings.groupby('context').cumcount() + 1)
        outcomes = self.trials.groupby('context').state.value_counts().unstack(fill_value=0)
        self.outcomes = outcomes.assign(
            runs=self.trials.groupby('context').study_key.nunique(),
            trials=outcomes.sum(axis=1),
            eligible_completed=self.trials.groupby('context').candidate.sum(),
        ).reset_index()
        self.model_rankings, self.model_samplers = model_summary_tables(self.trials)
        self.timings, self.pressure, self.execution_details, self.schema = extract_execution(d)
        good = self.timings[self.timings.status.eq('complete') & self.timings.value.notna()]
        coverage_rows = []
        for ctx, g in self.trials.groupby('context'):
            attempts = group_rows(self.execution_details, ctx)
            group_timings = group_rows(self.timings, ctx)
            complete_launch = group_rows(good, ctx).query("kind == 'wall phase' and phase == 'launch_to_ready'")
            joined_trials = len(attempts[['study_key', 'trial']].drop_duplicates())
            coverage_rows.append({'context': ctx, 'trials': len(g), 'trials_with_execution': joined_trials,
                'execution_attempts': len(attempts), 'trials_without_execution': len(g) - joined_trials,
                'complete_launch_attempts': len(complete_launch),
                'incomplete_or_unknown_phase_observations': int((group_timings.kind.eq('wall phase') & ~group_timings.status.eq('complete')).sum())})
        self.timing_coverage = pd.DataFrame(coverage_rows)
        self.wall_details, self.log_details, self.pressure_details = compact_attempt_details(self.timings, self.pressure)
        self.timing_stats = statistics(good, ['context', 'kind', 'phase', 'definition', 'unit'])
        launch = good[good.kind.eq('wall phase') & good.phase.eq('launch_to_ready')]
        # First average repeated attempts within one trial, then equal weight each trial.
        launch = launch.groupby(['context', 'sampler', 'seed', 'trial', 'study_key'], dropna=False).value.mean().reset_index()
        self.launch_by_trial = statistics(launch, ['context', 'sampler', 'seed', 'trial'])
        print('Computing fANOVA and PED-ANOVA once per study/objective…', flush=True)
        self.importance, self.importance_status, self.parameter_audit = compute_importance(d)
        if self.parameter_audit.empty:
            self.parameter_audit = pd.DataFrame(columns=['context'])
        self.importance_counts = (self.importance_status.groupby(['context', 'method', 'status'])
                                  .size().rename('studies').reset_index())
        importance_columns = ['context', 'study_key', 'metric', 'candidates', 'method', 'status', 'warnings']
        self.importance_reasons = self.importance_status.reindex(columns=importance_columns).pivot(
            index=['context', 'study_key', 'metric', 'candidates'], columns='method',
            values=['status', 'warnings'])
        self.importance_reasons.columns = [f'{method} {field}' for field, method in self.importance_reasons.columns]
        self.importance_reasons = self.importance_reasons.reset_index()
        self.importance_pairs = (self.importance.pivot(
            index=['context', 'study_key', 'metric', 'parameter'], columns='method', values='importance')
            .reset_index() if not self.importance.empty else pd.DataFrame(columns=['context']))
        self.telemetry_summary = self._pressure_summary()
        self.section_coverage = pd.DataFrame([{'section': section, **d.counts([r for r in d.main_studies if r['section'] == section])}
            for section in ['Initial runs', 'Parameters probe', 'unclassified']])
        self.reference = pd.DataFrame([{'stage': stage, **d.counts(records),
            'reference_studies': refs[0], 'reference_trials': refs[1], 'reference_completed': refs[2]}
            for stage, records, refs in [('Raw', d.raw_studies, (423, 8137, 6410)),
                ('Deduplicated', d.unique_studies, (388, 7602, 6000)), ('Eligible', d.main_studies, (371, 7417, 5868))]])

        self._compute_audit()

    def _compute_audit(self):
        self.exclusion_counts = self.data.exclusion_inventory.groupby('reasons', dropna=False).size().rename('runs').reset_index()
        self.schema_counts = self.schema.groupby(['framework', 'notebook_version', 'has_server_telemetry'], dropna=False).size().rename('execution attempts').reset_index()
        # Empirical schema difference, using all eligible populated records per version/framework.
        differences = []
        for fw in sorted(self.schema.framework.unique()):
            g = self.schema[self.schema.framework.eq(fw)]
            v3 = g[g.is_v3]; v2 = g[~g.is_v3]
            keys2 = set(k for keys in v2.execution_keys for k in keys)
            keys3 = set(k for keys in v3.execution_keys for k in keys)
            timing2 = set(k for keys in v2.timing_fields for k in keys)
            timing3 = set(k for keys in v3.timing_fields for k in keys)
            differences.append({'framework': fw, 'v2_attempts': len(v2), 'v3_attempts': len(v3),
                'additional_v3_execution_keys': sorted(keys3 - keys2), 'additional_v3_timing_fields': sorted(timing3 - timing2),
                'shared_timing_fields': sorted(timing3 & timing2)})
        self.schema_differences = pd.DataFrame(differences)
        self.pressure_audit = (self.pressure.groupby(
            ['framework', 'metric', 'available', 'usable', 'execution_complete', 'missing_reason'], dropna=False
        ).size().rename('attempt metrics').reset_index())
        self.importance_audit = self.importance_status.groupby(['method', 'status']).size().rename('studies').reset_index()

    def _pressure_summary(self):
        rows = []
        groups = ['context', 'metric', 'unit', 'aggregation', 'method', 'semantics']
        for keys, g in self.pressure.groupby(groups, dropna=False):
            valid = g[g.usable & g.execution_complete & g.time_weighted_avg.notna()]
            weight = valid.weight_s.sum()
            rows.append(dict(zip(groups, keys)) | {'attempts': len(g), 'available_attempts': int(g.available.sum()),
                'incomplete_benchmark_attempts': int((~g.execution_complete).sum()),
                'usable_attempts': len(valid), 'sample_count': g.sample_count.sum(min_count=1),
                'measured_window_s': g.measured_window_s.sum(min_count=1),
                'coverage_fraction': ((g.coverage_fraction * g.measured_window_s).sum(min_count=1) / g.loc[g.coverage_fraction.notna(), 'measured_window_s'].sum())
                    if g.loc[g.coverage_fraction.notna(), 'measured_window_s'].sum() and g.metric.iloc[0] != 'client_requests_in_flight' else np.nan,
                'time_weighted_avg': (valid.time_weighted_avg * valid.weight_s).sum() / weight if weight else np.nan,
                'peak': valid.peak.max(), 'missing_reason': '; '.join(sorted(set(g.missing_reason.dropna()))),
                'summary_method': 'covered-window weighted attempt means; maximum recorded peak'})
        return pd.DataFrame(rows)

    def select(self, section, model=None, framework=None, concurrency=None):
        f = self.runs[self.runs.section.eq(section)]
        for column, value in [('model', model), ('framework', framework), ('concurrency', concurrency)]:
            if value is not None:
                f = f[f[column].map(model_heading).eq(model_heading(value))] if column == 'model' else f[f[column].eq(value)]
        return f

    def show_section(self, section):
        heading(2, section)
        selected = self.select(section)
        if selected.empty:
            display(Markdown('No classified eligible runs.'))
        for model in sorted(selected.model.map(model_heading).unique()):
            self.show_model(section, model)

    def show_model(self, section, model):
        model = model_heading(model)
        heading(3, 'Model: ' + model)
        self.show_model_summary(section, model)
        for framework in sorted(self.select(section, model).framework.unique()):
            self.show_framework(section, model, framework)

    def show_model_summary(self, section, model):
        display(Markdown('**Top five study results per concurrency and objective across frameworks**'))
        for frame, columns in [
            (self.model_rankings, ['concurrency', 'metric', 'rank', 'objective', 'unit',
                                  'framework', 'sampler', 'display_name', 'model', 'study_key', 'trial']),
            (self.model_samplers, ['concurrency', 'metric', 'sampler', 'runs', 'runs_with_result',
                                  'best_result', 'unit', 'framework', 'display_name']),
        ]:
            if frame is self.model_samplers:
                display(Markdown('**Samplers ranked by their best study result within each concurrency and objective**  \n'
                                 'Run counts include every eligible study. Runs with a result have an eligible completed trial.'))
            selected = frame.loc[frame.section.eq(section) & frame.base_model.eq(model), columns]
            with pd.option_context('display.max_rows', None):
                table(selected.assign(metric=selected.metric.map(objective_heading)))

    def show_framework(self, section, model, framework):
        heading(4, 'Framework: ' + framework)
        for concurrency in sorted(self.select(section, model, framework).concurrency.unique()):
            self.show_concurrency(section, model, framework, concurrency)

    def show_concurrency(self, section, model, framework, concurrency):
        selected = self.select(section, model, framework, concurrency)
        heading(5, 'Concurrency: ' + str(concurrency))
        for group in sorted(set(selected.context)):
            heading(6, objective_heading(group[-1]))
            self.show_objective(group)

    def show_objective(self, group):
        for label, frame in [
            ('Trial outcomes', self.outcomes),
            ('Objective statistics', self.objective_overall),
            ('Objective statistics by sampler and seed', self.objective_stats),
            ('Best distinct runs', self.rankings),
            ('Execution and timing coverage', self.timing_coverage),
            ('Startup and serving timings (variance in s²)', self.timing_stats),
            ('Recorded serving metrics', self.serving_stats),
            ('Benchmark-window telemetry coverage', self.telemetry_summary),
            ('Per-study importance coverage', self.importance_counts),
        ]:
            display(Markdown(f'**{label}**'))
            table(group_rows(frame, group).drop(columns='context'))
        self.plot_trial_behavior(group)
        self.plot_overall(group)
        self.show_details(group)

    def plot_trial_behavior(self, group):
        values = group_rows(self.objectives, group)
        means = group_rows(self.mean_by_trial, group)
        launch = group_rows(self.launch_by_trial, group)
        fig, (objective_ax, launch_ax) = plt.subplots(1, 2, figsize=(12, 3.5))
        for (sampler, seed), points in values.groupby(['sampler', 'seed'], dropna=False):
            average = means[means.sampler.eq(sampler) if pd.notna(sampler) else means.sampler.isna()]
            average = average[average.seed.eq(seed) if pd.notna(seed) else average.seed.isna()]
            mean_line, = objective_ax.plot(average.trial, average['mean'], '.',
                                          label=f'{sampler} / seed {seed} mean')
            annotate_counts(objective_ax, average)
            for _, study in points.groupby('study_key'):
                study = study.sort_values('trial')
                incumbent = study.value.cummin() if study.direction.iloc[0] == 'MINIMIZE' else study.value.cummax()
                objective_ax.scatter(study.trial, study.value, s=7, alpha=.3, color=mean_line.get_color())
                objective_ax.plot(study.trial, incumbent, '_', alpha=.6, color=mean_line.get_color())
        objective_ax.set(title='Trial values, incumbents and means', xlabel='Original trial number',
                         ylabel=UNITS.get(group[-1], 'unit unavailable'))
        if not values.empty:
            objective_ax.legend(fontsize=6)
        else:
            objective_ax.text(.1, .5, 'Eligible completed trials unavailable', transform=objective_ax.transAxes)
        for (sampler, seed), points in launch.groupby(['sampler', 'seed'], dropna=False):
            launch_ax.plot(points.trial, points['mean'], '.', label=f'{sampler} / seed {seed}')
            annotate_counts(launch_ax, points)
        launch_ax.set(title='Mean server launch-to-ready', xlabel='Original trial number', ylabel='s')
        if not launch.empty:
            launch_ax.legend(fontsize=6)
        else:
            launch_ax.text(.1, .5, 'Complete startup timing unavailable', transform=launch_ax.transAxes)
        fig.suptitle('Count labels = observations/contributing runs')
        finish_plot(fig)

    def plot_overall(self, ctx):
        values = group_rows(self.objectives, ctx)
        if not values.empty:
            groups = list(values.groupby(['sampler', 'seed'], dropna=False))
            fig, ax = plt.subplots(figsize=(max(6, len(groups) * 1.2), 3))
            ax.boxplot([rows.value for _, rows in groups],
                       tick_labels=[f'{sampler} / seed {seed}' for (sampler, seed), _ in groups])
            ax.tick_params(axis='x', rotation=20)
            ax.set(title=objective_heading(ctx[-1]), ylabel=values.unit.iloc[0])
            finish_plot(fig)
        telemetry = group_rows(self.telemetry_summary, ctx)
        usable = telemetry[telemetry.time_weighted_avg.notna()]
        if not usable.empty:
            fig, axes = plt.subplots(1, len(usable), figsize=(max(6, len(usable)*3), 3), squeeze=False)
            for ax, (_, row) in zip(axes[0], usable.iterrows()):
                ax.bar(['Time-weighted mean', 'Peak'], [row.time_weighted_avg, row.peak])
                ax.tick_params(axis='x', rotation=30)
                ax.set(title=f'{row.metric}\n{row.method} / {row.aggregation}', ylabel=row.unit)
            fig.suptitle('Benchmark window')
            finish_plot(fig)

    def show_details(self, group):
        selected = group_rows(self.runs, group)
        trials = group_rows(self.trials, group)
        frames = [
            ('Run inventory, configurations and provenance', selected),
            ('Individual trial results', trials),
            ('Per-trial objective means and counts', group_rows(self.mean_by_trial, group)),
            ('Per-trial launch means and counts', group_rows(self.launch_by_trial, group)),
            ('Wall phase timings and phase status', group_rows(self.wall_details, group)),
            ('Engine-log observations (may overlap)', group_rows(self.log_details, group)),
            ('Telemetry measurements and collection errors', group_rows(self.pressure_details, group)),
            ('Execution records', group_rows(self.execution_details, group)),
            ('fANOVA and PED-ANOVA status and skip reasons', group_rows(self.importance_reasons, group)),
            ('Parameter inclusion and exclusion audit', group_rows(self.parameter_audit, group)),
            ('fANOVA and PED-ANOVA estimates', group_rows(self.importance_pairs, group)),
        ]
        closed_toggle(f'Show all runs ({len(selected)} runs, {len(trials)} trials)',
                      ''.join(details_table(title, frame.drop(columns='context')) for title, frame in frames))

    def show_audit(self, include_heading=True):
        if include_heading:
            heading(2, 'Audit and provenance')
        table(self.reference); table(self.section_coverage)
        display(Markdown(f"Excluded studies: **{len(self.data.excluded)}**; empty databases/folders: **{sum(self.data.folder_inventory.study_records.eq(0))}**. "
                         'Classification partitions eligible studies without changing their total. Unknown VM identities are recorded with their original campaign.'))
        table(self.exclusion_counts)
        table(self.runs[self.runs.section.eq('unclassified')][['key', 'study_name', 'wandb_id', 'classification_reason', 'name_evidence', 'sources']])
        table(self.schema_counts)
        table(self.schema_differences)
        table(self.pressure_audit)
        table(self.importance_audit)
        closed_toggle(f'Show audit details ({len(self.data.raw_studies)} raw runs)', ''.join([
            details_table('Excluded runs', self.data.exclusion_inventory),
            details_table('Raw inventory', self.data.inventory), details_table('Folders', self.data.folder_inventory),
            details_table('Executed notebooks', pd.DataFrame(self.data.notebook_rows)),
            details_table('Study metadata and observed distributions', self.data.context_inventory),
            details_table('Measured attempts', self.data.unique_attempts.drop(columns='execution')),
            details_table('Source issues', pd.DataFrame(self.data.issues))]))

    def validate(self):
        assert sum(self.section_coverage['studies / experiment runs']) == len(self.data.main_studies)
        assert self.runs.duration_seconds.map(eligible_duration).all()
        assert (len(self.runs), len(self.trials), self.trials.state.eq('COMPLETE').sum()) == (371, 7417, 5868)
        for ctx, g in self.runs.groupby('context'):
            assert all(report_group(self.data.lookup[key]) == ctx for key in g.key)
            if len(self.data.lookup[g.key.iloc[0]]['directions']) == 1:
                assert self.best[ctx].study_key.is_unique and len(self.best[ctx]) <= 5
        for fw in ['sglang', 'vllm', 'trtllm']:
            records = self.pressure[(self.pressure.framework == fw) & self.pressure.is_v3
                & self.pressure.section.eq('Parameters probe') & self.pressure.source.str.startswith('trial')
                & self.pressure.raw_measurement.map(bool)]
            representative = records.iloc[0]
            original = self.data.lookup[representative.study_key]
            trial = next(t for t in original['trials'] if t['number'] == representative.trial)
            execution = trial['attrs'].get('measurement', {}).get('execution') or trial['attrs'].get('execution')
            assert execution and execution.get('attempt_id') == representative.attempt_id
            if execution:
                expected = {r['metric']: r for r in pressure_records(execution, {})}
                for row in records[records.attempt_id.eq(representative.attempt_id)].to_dict('records'):
                    e = expected[row['metric']]
                    assert row['available'] == e['available']
                    assert row['sample_count'] == e['sample_count'] or (pd.isna(row['sample_count']) and e['sample_count'] is None)
                    if e['time_weighted_avg'] is None: assert pd.isna(row['time_weighted_avg'])
                    else: assert row['time_weighted_avg'] == e['time_weighted_avg']
        assert {'sglang', 'vllm', 'trtllm'} <= set(self.schema[self.schema.is_v3].framework)
        assert self.pressure.loc[~self.pressure.usable, 'time_weighted_avg'].isna().all()
        assert set(self.importance_status.method) == {'fANOVA', 'PED-ANOVA'}
        assert not self.importance_status.duplicated(['study_key', 'metric', 'method']).any()
        assert len(self.importance_status) == 2 * sum(len(r['contract']['metrics']) for r in self.data.main_studies)
        eligible_keys = set(self.runs.key)
        for frame in [self.timings, self.pressure, self.importance_status, self.objectives]:
            assert set(frame.study_key) <= eligible_keys
        for path, expected in self.data.source_hashes.items():
            assert hashlib.sha256((self.data.ROOT / path).read_bytes()).hexdigest() == expected, path
        return {'eligible_runs': len(self.runs), 'trials': len(self.trials),
                'completed_trials': int(self.trials.state.eq('COMPLETE').sum()), 'timing_attempts': len(self.execution_details),
                'complete_wall_timing_observations': len(self.timings[(self.timings.kind == 'wall phase') & (self.timings.status == 'complete')]),
                'source_database_hashes_verified': len(self.data.source_hashes)}


def heading(level, text):
    display(Markdown(f'{"#" * level} {text}'))


def frame_html(frame):
    formatters = {column: lambda value: json.dumps(value, indent=2, default=str) if isinstance(value, (dict, list, tuple)) else str(value)
                  for column in frame.select_dtypes(include='object').columns}
    return '<div class="report-table">' + frame.to_html(index=False, escape=True, render_links=True, formatters=formatters) + '</div>'


def table(frame):
    display(frame.reset_index(drop=True) if not frame.empty else Markdown('Unavailable: no recorded observations.'))


def details_table(title, frame):
    return '<h6>' + html.escape(title) + '</h6>' + (frame_html(frame) if not frame.empty else '<p>Unavailable</p>')


def closed_toggle(label, content):
    display(HTML('<details><summary>' + html.escape(label) + '</summary>' + content + '</details>'))


def annotate_counts(ax, frame):
    for row in frame.itertuples():
        ax.annotate(f'{row.observations}/{row.contributing_runs}', (row.trial, row.mean), fontsize=5, alpha=.6)


def finish_plot(fig):
    fig.tight_layout(); plt.show(); plt.close(fig)


def protobuf_fields(payload):
    """Read protobuf wire fields without importing the W&B client or initializing it."""
    pos = 0
    def varint():
        nonlocal pos
        value = 0
        for shift in range(0, 70, 7):
            if pos >= len(payload): raise ValueError('truncated protobuf varint')
            byte = payload[pos]; pos += 1; value |= (byte & 127) << shift
            if byte < 128: return value
        raise ValueError('invalid protobuf varint')
    while pos < len(payload):
        tag = varint(); number, wire = tag >> 3, tag & 7
        if wire == 0: value = varint()
        elif wire == 2:
            length = varint(); value = payload[pos:pos + length]; pos += length
        elif wire in (1, 5):
            width = 8 if wire == 1 else 4; value = payload[pos:pos + width]; pos += width
        else: raise ValueError('unsupported protobuf wire type')
        if pos > len(payload): raise ValueError('truncated protobuf field')
        yield number, wire, value


def cached_wandb_names(path):
    """Extract verified RunRecord IDs/names from local W&B transaction logs.

    W&B log v0: seven-byte :W&B header; 32 KiB blocks; CRC32/uint16/type
    fragments (FULL=1, FIRST=2, MIDDLE=3, LAST=4). Protobuf Record.run=17;
    RunRecord.run_id=1 and display_name=8, verified against installed W&B schema.
    Validate framing/checksums; never search arbitrary binary strings for a name.
    """
    import struct
    import zlib
    with path.open('rb') as stream:
        if stream.read(7) != b':W&B\xe1\xbe\x00':
            raise ValueError('unsupported W&B datastore header')
        fragments = None
        while True:
            remaining = 32768 - stream.tell() % 32768
            if remaining < 7:
                stream.read(remaining)
            header = stream.read(7)
            if not header: break
            if len(header) != 7: raise ValueError('truncated W&B log header')
            crc, length, kind = struct.unpack('<IHB', header)
            if kind == 0 and length == 0:
                stream.read(32768 - stream.tell() % 32768); continue
            chunk = stream.read(length)
            if len(chunk) != length or zlib.crc32(bytes([kind]) + chunk) != crc:
                raise ValueError('truncated or checksum-invalid W&B log record')
            if kind == 1: payload = chunk
            elif kind == 2: fragments = bytearray(chunk); continue
            elif kind in (3, 4) and fragments is not None:
                fragments.extend(chunk)
                if kind == 3: continue
                payload = bytes(fragments); fragments = None
            else: raise ValueError('invalid W&B log fragment sequence')
            for number, wire, value in protobuf_fields(payload):
                if number == 17 and wire == 2:
                    run = {n: v for n, w, v in protobuf_fields(value) if w == 2 and n in (1, 8)}
                    if run.get(1) and run.get(8):
                        yield run[1].decode('utf-8'), run[8].decode('utf-8')


def compact_attempt_details(timings, pressure):
    """One row per attempt for wall phases/gauges; keep every engine-log observation.

    Avoid repeating long identities nine times per attempt or emitting five
    artificial metric-detail rows for executions without any pressure telemetry.
    Original frames remain the inputs to all statistics and validation.
    """
    identity = ['context', 'study_key', 'trial', 'attempt_id', 'source']
    wall_rows = []
    wall = timings[timings.kind.eq('wall phase')]
    for keys, group in wall.groupby(identity, dropna=False, sort=False):
        phases = {row.phase: {'seconds': row.value, 'status': row.status,
                             'definition': row.definition} for row in group.itertuples()}
        wall_rows.append(dict(zip(identity, keys)) | {'recorded_wall_phases': phases, 'unit': 's'})
    logs = timings[timings.kind.eq('engine-log observation')]
    log_columns = identity + ['phase', 'value', 'unit', 'definition', 'observation', 'line_number', 'log_source']
    logs = logs.reindex(columns=log_columns)
    pressure_rows = []
    recorded = pressure[pressure.raw_measurement.map(bool)]
    for keys, group in recorded.groupby(identity, dropna=False, sort=False):
        first = group.iloc[0]
        pressure_rows.append(dict(zip(identity, keys)) | {
            'measured_window_s': first.measured_window_s,
            'benchmark_complete': first.execution_complete,
            'recorded_metrics': {row.metric: row.raw_measurement for row in group.itertuples()},
            'collection_errors': first.collection_errors, 'semantics': first.semantics})
    return (pd.DataFrame(wall_rows, columns=identity + ['recorded_wall_phases', 'unit']), logs,
            pd.DataFrame(pressure_rows, columns=identity + ['measured_window_s', 'benchmark_complete',
                'recorded_metrics', 'collection_errors', 'semantics']))
