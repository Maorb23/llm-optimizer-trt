"""Compose frozen chat workloads with the unchanged v3 campaign and lifecycle."""

import builtins
import hashlib
import importlib
import json
import time
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path
from types import FunctionType, SimpleNamespace

from .datasets import (
    Sampling,
    adapt,
    canonical,
    digest,
    freeze,
    held_out,
    load_hf,
    sample,
)
from .experiment import DatasetExperimentV4, ExperimentSettings, check_scorer_runtime
from .runtime import Completion, RuntimeV4
from .scorers import aggregate_scores, score


def _clone(function, overrides, imports=None):
    # v3 has no dependency injection for its local imports. Rebind only the
    # private copy's dependencies, including in its execution thread.
    namespace = {**function.__globals__, **overrides}
    if imports:

        def import_dependency(name, globals=None, locals=None, fromlist=(), level=0):
            if name in imports:
                return imports[name]
            return builtins.__import__(name, globals, locals, fromlist, level)

        namespace["__builtins__"] = {**vars(builtins), "__import__": import_dependency}
    bound = FunctionType(
        function.__code__,
        namespace,
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    bound.__kwdefaults__ = function.__kwdefaults__
    return bound


def prepare_legacy(path, revision):
    """Pin the untouched v3 ShareGPT sampler's input bytes."""
    from .datasets import pinned_revision

    if path is None:
        from huggingface_hub import hf_hub_download

        revision = pinned_revision(revision)
        if len(revision) != 40:
            raise ValueError("Legacy Hub dataset requires a commit SHA")
        path = hf_hub_download(
            "anon8231489123/ShareGPT_Vicuna_unfiltered",
            "ShareGPT_V3_unfiltered_cleaned_split.json",
            repo_type="dataset",
            revision=revision,
        )
    else:
        revision = None
    source = Path(path).resolve()
    with source.open("rb") as stream:
        sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    return source, {"source_sha256": sha256, "dataset_revision": revision}


class CampaignWorkloads:
    """Freeze all measured workloads before opening any study."""

    def __init__(
        self,
        *,
        source,
        source_format,
        dataset_id,
        revision,
        split,
        dataset_config,
        generation,
        scorer,
        thresholds,
        framework,
        models,
        revisions,
        snapshots,
        requests,
        repeat_count,
        repeat_seeds,
        held_out_count,
        sampling_seed,
        buckets,
        thinking_verified,
        fixture_count=0,
        score_quality=True,
    ):
        self.scorer = scorer
        self.thresholds = ExperimentSettings(
            quality_thresholds=thresholds
        ).quality_thresholds
        self.score_quality = score_quality
        if self.thresholds and not score_quality:
            raise ValueError("Thresholds require quality scoring")
        self.framework = framework
        self.training, self.validation, self.repeats, self.lengths = {}, {}, {}, {}
        if source == "fixture":
            if source_format != "extraction":
                raise ValueError("The offline fixture uses extraction")
            rows = [
                {
                    "instruction": "Extract id as JSON",
                    "text": f"id={i}",
                    "schema": {
                        "type": "object",
                        "required": ["id"],
                        "properties": {"id": {"type": "integer"}},
                    },
                    "expected": {"id": i},
                }
                for i in range(fixture_count)
            ]
            dataset_id, revision = "offline-extraction", "sha256:" + digest(rows)
        elif source == "local":
            path = Path(dataset_id)
            content = path.read_bytes()
            if path.suffix == ".jsonl":
                rows = [
                    json.loads(line) for line in content.splitlines() if line.strip()
                ]
            else:
                rows = json.loads(content)
            revision = "sha256:" + hashlib.sha256(content).hexdigest()
        elif source != "hf":
            raise ValueError("Choose fixture, local, hf, or legacy_random")
        examples = (
            load_hf(
                dataset_id,
                revision=revision,
                source_format=source_format,
                split=split,
                config=dataset_config,
                seed=sampling_seed,
            )
            if source == "hf"
            else tuple(adapt(row, source_format, seed=sampling_seed) for row in rows)
        )
        tuning, holdout = held_out(
            examples, validation_count=held_out_count, seed=sampling_seed
        )

        def prepare_phase(phase, model):
            runtime = RuntimeV4(
                model,
                runtime_version="v3 managed",
                thinking_template_verified=thinking_verified,
            )
            runtime.framework = framework
            runtime.validate_generation(generation)
            counter = None
            tokenizer_settings = {
                "add_generation_prompt": True,
                "thinking": generation.thinking,
            }
            if source != "fixture":
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    snapshots[phase], local_files_only=True, trust_remote_code=True
                )
                tokenizer_settings["chat_template_sha256"] = digest(
                    tokenizer.chat_template
                )
                tokenizer_settings["transformers_version"] = version("transformers")
                options = (
                    {}
                    if generation.thinking == "default"
                    else {"enable_thinking": generation.thinking == "on"}
                )

                def counter(messages):
                    return len(
                        tokenizer.apply_chat_template(
                            messages,
                            tokenize=True,
                            return_dict=False,
                            add_generation_prompt=True,
                            **options,
                        )
                    )

            def make(pool, count, seed, role, quota=()):
                settings = Sampling(
                    "token_bucket" if quota else "natural", count, seed, quota
                )
                selected = sample(pool, settings, counter)
                manifest = freeze(
                    selected,
                    dataset_id=str(dataset_id),
                    revision=revision,
                    source_format=source_format,
                    split=split,
                    dataset_config=dataset_config,
                    model=model,
                    model_revision=revisions[phase],
                    tokenizer_revision=revisions[phase],
                    generation=generation,
                    sampling=settings,
                    evaluation=scorer.identity,
                    role=role,
                    tokenizer_settings=tokenizer_settings,
                )
                self.lengths[manifest.signature] = {
                    e.example_id: counter(e.payload["messages"]) if counter else 0
                    for e in selected
                }
                return manifest

            valid = make(holdout, held_out_count, sampling_seed, "validation")
            self.validation[phase] = valid
            for concurrency, count in requests.items():
                train = make(
                    tuning, count, sampling_seed, "train", buckets.get(concurrency, ())
                )
                DatasetExperimentV4(
                    runtime,
                    train,
                    validation_manifest=valid,
                    scorer=scorer,
                    settings=ExperimentSettings(quality_thresholds=self.thresholds),
                    validate_scorer=score_quality,
                )
                self.training[phase, concurrency] = train
            for seed in repeat_seeds:
                self.repeats[phase, seed] = make(tuning, repeat_count, seed, "train")

        for phase, model in models.items():
            prepare_phase(phase, model)

        self.identity = {
            "version": "managed-datasets-v4.1",
            "source": source,
            "training": {
                f"{p}:{c}": m.signature for (p, c), m in self.training.items()
            },
            "held_out": {p: m.signature for p, m in self.validation.items()},
            "repeats": {f"{p}:{s}": m.signature for (p, s), m in self.repeats.items()},
            "score_quality": score_quality,
            "thresholds": self.thresholds,
            "scorer": scorer.identity,
            "thinking_template_verified": thinking_verified,
            "implementation": {
                name: digest(Path(__file__).with_name(name).read_text())
                for name in ("campaign.py", "datasets.py", "runtime.py", "scorers.py")
            },
        }

    def save(self, directory):
        for manifest in (
            *self.training.values(),
            *self.validation.values(),
            *self.repeats.values(),
        ):
            manifest.save(directory)


class CampaignBenchmark:
    """Factory for per-workload v3 benchmark subclasses."""

    @staticmethod
    def bind(factory, workloads, *, held_out_run=False):
        parent = factory.func
        framework = workloads.framework
        runtime_module = importlib.import_module(parent.__module__)
        telemetry_module = importlib.import_module(
            f"llm_optimizer.experiment.telemetry_{framework}_v3"
        )
        from llm_optimizer import bench_client

        from ..lab import RepositoryBenchmark

        def create(*args, **kwargs):
            phase, concurrency = kwargs["phase"], args[3]
            if held_out_run:
                manifest = workloads.validation[phase]
            elif kwargs.get("workload_seed") is not None:
                manifest = workloads.repeats[phase, kwargs["workload_seed"]]
            else:
                manifest = workloads.training[phase, concurrency]
            doc = manifest.document
            examples = {str(i): e for i, e in enumerate(doc["examples"])}
            lengths = workloads.lengths[manifest.signature]
            generation = doc["generation"]

            def prepare(_):
                return {
                    "version": "frozen-chat-request-index-v1",
                    "manifest": manifest.signature,
                    "requests": [
                        {
                            "prompt": key,
                            "prompt_len": lengths[e["example_id"]],
                            "output_len": generation["output_tokens"],
                        }
                        for key, e in examples.items()
                    ],
                }, {"evaluation_manifest": manifest.signature}

            init = _clone(
                RepositoryBenchmark.__init__,
                {},
                {
                    "llm_optimizer.bench_client": SimpleNamespace(
                        prepare_workload_manifest=prepare
                    ),
                },
            )

            evaluation_kind = (
                "held_out"
                if held_out_run
                else "candidate_repeat"
                if kwargs.get("workload_seed") is not None
                else "tuning"
            )

            class Benchmark(parent):
                def __init__(self, *args, **kwargs):
                    parent_init = _clone(
                        parent.__init__,
                        {"super": lambda: SimpleNamespace(__init__=init.__get__(self))},
                    )
                    parent_init(self, *args, **kwargs)

                def measure(self, framework, config, concurrency, **measure_kwargs):
                    from .runtime import Generation

                    gen = Generation(**generation)
                    runtime = RuntimeV4(
                        doc["model"],
                        runtime_version="v3 managed",
                        thinking_template_verified=workloads.identity[
                            "thinking_template_verified"
                        ],
                    )
                    runtime.framework = framework
                    completions = {}
                    is_measured = False

                    def get_request(*args, **kwargs):
                        nonlocal is_measured
                        is_measured = True
                        return bench_client.get_request(*args, **kwargs)

                    async def request(request_func_input, pbar=None, **_):
                        item = request_func_input
                        key = item.prompt
                        payload = runtime.payload(
                            examples[key],
                            replace(gen, output_tokens=item.output_len),
                            served_model=item.model,
                        )
                        output, completion = await chat_request(
                            item, examples[key]["example_id"], payload, pbar
                        )
                        if is_measured:
                            completions[key] = completion
                        return output

                    client_namespace = {
                        **vars(bench_client),
                        "get_request": get_request,
                    }
                    client_namespace["ASYNC_REQUEST_FUNCS"] = dict.fromkeys(
                        bench_client.ASYNC_REQUEST_FUNCS, request
                    )
                    client_namespace["benchmark"] = _clone(
                        bench_client.benchmark, client_namespace
                    )
                    client_namespace["run_benchmark"] = _clone(
                        bench_client.run_benchmark, client_namespace
                    )

                    def run_client(args):
                        nonlocal is_measured
                        is_measured = False
                        completions.clear()
                        return telemetry_run(args)

                    telemetry_run = _clone(
                        telemetry_module.run_benchmark,
                        {},
                        {
                            "llm_optimizer": SimpleNamespace(
                                bench_client=SimpleNamespace(**client_namespace)
                            )
                        },
                    )
                    lifecycle = _clone(
                        runtime_module.run_single_configuration,
                        {},
                        {
                            "": SimpleNamespace(
                                **{
                                    f"telemetry_{framework}_v3": SimpleNamespace(
                                        run_benchmark=run_client
                                    )
                                }
                            ),
                        },
                    )
                    measure = _clone(
                        parent.measure, {"run_single_configuration": lifecycle}
                    )
                    from ..lab import BenchmarkFailure, optuna

                    failure = None
                    try:
                        measured = measure(
                            self, framework, config, concurrency, **measure_kwargs
                        )
                    except (BenchmarkFailure, optuna.TrialPruned) as exc:
                        if not completions:
                            raise
                        failure, measured = exc, None
                    execution = (
                        measured.execution
                        if measured
                        else getattr(failure, "execution", {})
                    ) or {}
                    if measured is not None and len(completions) != len(examples):
                        raise RuntimeError(
                            "Runtime accounting differs from frozen workload"
                        )
                    # No GPU lifecycle, inference timer or telemetry collector is
                    # active here. Warmups are never included in the scoring set.
                    started = time.perf_counter()
                    refs = manifest.references
                    full_workload = len(completions) == len(examples)
                    rows = [
                        score(
                            doc["task"]
                            if workloads.score_quality and full_workload
                            else "conversation",
                            completions[key],
                            refs[e["example_id"]],
                            workloads.scorer,
                        )
                        for key, e in examples.items()
                        if key in completions
                    ]
                    accounting = aggregate_scores(rows)
                    feasible = not workloads.thresholds or (
                        not accounting["generation_failures"]
                        and not accounting["scorer_failures"]
                        and all(
                            accounting["quality"].get(k, -1) >= v
                            for k, v in workloads.thresholds
                        )
                    )
                    evaluation = {
                        "manifest": manifest.signature,
                        "role": doc["role"],
                        "kind": evaluation_kind,
                        "planned_requests": len(examples),
                        "full_workload": full_workload,
                        "task": doc["task"],
                        "quality_available": bool(accounting["quality"]),
                        "evaluation": doc["evaluation"],
                        "quality_feasible": feasible,
                        "scoring_seconds": time.perf_counter() - started,
                        **accounting,
                    }
                    result = {
                        "evaluation": evaluation,
                        "scores": rows,
                        "measurement": asdict(measured) if measured else None,
                        "attempt": execution.get("attempt_id"),
                        "config": config,
                        "concurrency": concurrency,
                    }
                    with self.output_path.with_suffix(".quality.jsonl").open(
                        "a"
                    ) as stream:
                        stream.write(
                            canonical(self._context.artifact_value(result)) + "\n"
                        )
                    execution = {**execution, "dataset_v4": evaluation}
                    if failure:
                        failure.execution = execution
                        raise failure
                    return replace(measured, execution=execution)

            return Benchmark(*args, **kwargs, **factory.keywords)

        return create


def attach_campaign(runner, workloads):
    """Keep v3 objective values and performance checkpoints intact."""
    import optuna

    lab = runner.lab
    if lab.RUN_REAL_GPU and workloads.score_quality:
        check_scorer_runtime(
            next(iter(workloads.validation.values())).document["task"], workloads.scorer
        )
    with runner.lock():
        workloads.save(runner.run_dir / "workloads")
    factory = lab.RepositoryBenchmark
    lab.RepositoryBenchmark = CampaignBenchmark.bind(factory, workloads)
    record = lab.record_measurement

    def record_measurement(trial, measured, concurrency):
        record(trial, measured, concurrency)
        evaluation = (measured.execution or {}).get("dataset_v4")
        if evaluation is None:
            return
        trial.set_user_attr("dataset_v4", evaluation)
        metrics = trial.user_attrs["measurement"]
        metrics.update(
            {f"quality/{key}": value for key, value in evaluation["quality"].items()}
        )
        for key in (
            "scorer_failures",
            "generation_failures",
            "truncated",
            "scoring_seconds",
            "quality_feasible",
        ):
            metrics[f"quality/{key}"] = evaluation[key]
        trial.set_user_attr("measurement", metrics)
        if not evaluation["quality_feasible"]:
            # Final rejection makes the trial ineligible for scalar or Pareto
            # selection without changing sampler dimensionality or objective.
            raise optuna.TrialPruned("Final task quality feasibility gate")

    lab.record_measurement = record_measurement
    runner.held_out_benchmark = CampaignBenchmark.bind(
        factory, workloads, held_out_run=True
    )
    runner.dataset_workloads = workloads
    return runner


async def chat_request(item, example_id, payload, pbar):
    """Adapt OpenAI chat streaming to the existing v3 request accounting."""
    import aiohttp

    from llm_optimizer.bench_client import (
        RequestFuncOutput,
        classify_error,
        get_auth_headers,
    )

    output = RequestFuncOutput.init_new(item)
    started = time.perf_counter()
    first = last = None
    text, reasoning = [], []
    finish = tokens = None
    url = item.api_url.split("/v1/")[0].split("/v2/")[0].removesuffix("/generate")
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=600)
        ) as session:
            async with session.post(
                url + "/v1/chat/completions", json=payload, headers=get_auth_headers()
            ) as response:
                response.raise_for_status()
                async for line in response.content:
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        break
                    event = json.loads(data)
                    if "error" in event:
                        raise RuntimeError(str(event["error"]))
                    usage = event.get("usage") or {}
                    if usage.get("completion_tokens") is not None:
                        tokens = int(usage["completion_tokens"])
                    for choice in event.get("choices", []):
                        delta = choice.get("delta") or {}
                        content = delta.get("content") or ""
                        thought = (
                            delta.get("reasoning_content")
                            or delta.get("reasoning")
                            or ""
                        )
                        if content or thought:
                            now = time.perf_counter()
                            if first is None:
                                first = now - started
                            else:
                                output.itl.append(now - last)
                            last = now
                        text.append(content)
                        reasoning.append(thought)
                        finish = choice.get("finish_reason") or finish
        if (
            finish not in {"stop", "length", "max_tokens"}
            or tokens is None
            or tokens < 0
        ):
            raise RuntimeError(
                "Missing finish reason or authoritative completion token usage"
            )
        if first is None:
            raise RuntimeError("The stream contained no text or reasoning tokens")
        output.success = True
        output.ttft = first
        output.output_len = tokens
        output.usage = {"completion_tokens": tokens}
    except (
        aiohttp.ClientError,
        TimeoutError,
        RuntimeError,
        ValueError,
        TypeError,
    ) as exc:
        output.error = f"{type(exc).__name__}: {exc}"
        output.error_category = classify_error(
            output.error, getattr(exc, "status", None)
        )
    finally:
        if pbar is not None:
            pbar.update(1)
    output.latency = time.perf_counter() - started
    output.generated_text = "".join(text)
    return output, Completion(
        example_id,
        output.generated_text,
        "".join(reasoning),
        finish,
        tokens,
        output.latency,
        first,
        output.error or None,
    )


def evaluate_held_out(runner, study, *, pareto_trial=None):
    """Evaluate a selected winner under the original arm's v3 server settings."""
    import optuna

    from ..lab import BenchmarkFailure

    lab = runner.lab
    expected_storage = lab.STORAGE.url
    study = optuna.load_study(study_name=study.study_name, storage=expected_storage)
    contract = study.user_attrs["contract"]
    eligible = lab.completed(study, feasible_only=contract["constrained"])
    if len(study.directions) == 1:
        if pareto_trial is not None:
            raise ValueError("Specify a trial number only for a Pareto study")
        winner = lab.choose_single(study, feasible_only=contract["constrained"])
    else:
        front = lab.pareto_front(eligible, study.directions)
        winner = next((trial for trial in front if trial.number == pareto_trial), None)
    if winner is None:
        raise ValueError(
            "Select an eligible scalar winner or nominate an eligible Pareto trial"
        )
    phase, framework, concurrency = (
        contract[key] for key in ("phase", "framework", "concurrency")
    )
    manifest = runner.dataset_workloads.validation[phase]
    with runner.lock():
        # A shallow copy retains the precise v3 notebook policy and leaves the
        # optimization lab's request factory untouched.
        from copy import copy

        held_out_lab = copy(lab)
        held_out_lab.RepositoryBenchmark = runner.held_out_benchmark
        benchmark = held_out_lab.benchmark_for(
            phase,
            framework,
            concurrency,
            f"held-out-{study.study_name}-{winner.number}",
            num_prompts=len(manifest.document["examples"]),
        )
        try:
            measured = benchmark.measure(
                framework, winner.user_attrs["config"], concurrency
            )
        except BenchmarkFailure as failure:
            outcome = {
                "measurement": None,
                "failure": str(failure),
                "execution": getattr(failure, "execution", {}),
                "request_accounting": getattr(failure, "request_accounting", {}),
            }
        else:
            outcome = {"measurement": asdict(measured)}
        result = {
            "study": study.study_name,
            "trial": winner.number,
            "contract": contract,
            "config": winner.user_attrs["config"],
            "manifest": manifest.signature,
            **outcome,
        }
        path = runner.run_dir / "held-out-results.jsonl"
        with path.open("a") as stream:
            stream.write(canonical(lab.artifact_value(result)) + "\n")
        evaluation_run = SimpleNamespace(
            study_name="held-out-" + study.study_name,
            set_user_attr=lambda *_: None,
        )
        logger = lab.NotebookWandbLogger(evaluation_run, contract)
        try:
            if logger.tracker is not None and logger.tracker.run is not None:
                logger.tracker.run.summary["held_out"] = lab.artifact_value(result)
                logger.tracker.artifacts(
                    [path, benchmark.output_path.with_suffix(".quality.jsonl")],
                    name="held-out-" + study.study_name,
                )
        except Exception:
            import warnings

            warnings.warn("Held-out W&B logging failed", stacklevel=2)
        finally:
            if logger.tracker is not None:
                logger.tracker.finish()
    return result
