"""Optuna optimization with frozen evaluation identity and one held-out check."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .datasets import canonical, digest
from .scorers import ScorerSettings, aggregate_scores, score


def check_scorer_runtime(task, settings):
    """Fail before allocating a GPU when an executable scorer is unavailable."""
    if task not in {"humaneval", "humaneval_plus"}:
        return
    import shutil
    import subprocess

    if shutil.which("docker") is None:
        raise RuntimeError("Docker is required for executable task scoring")
    inspected = subprocess.run(
        ["docker", "image", "inspect", settings.image],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=10,
        check=False,
    )
    if inspected.returncode:
        raise RuntimeError("Pinned CPU scorer image is unavailable locally")


@dataclass(frozen=True)
class ExperimentSettings:
    seed: int = 0
    trials: int = 10
    quality_thresholds: tuple = ()
    wandb_enabled: bool = False
    gpu_identity: str = "unspecified"

    def __post_init__(self):
        object.__setattr__(
            self,
            "quality_thresholds",
            tuple(sorted(tuple(p) for p in self.quality_thresholds)),
        )
        if self.trials < 1 or len(dict(self.quality_thresholds)) != len(
            self.quality_thresholds
        ):
            raise ValueError("Invalid trials/duplicate quality thresholds")
        if any(
            not isinstance(k, str) or not math.isfinite(v) or not 0 <= v <= 1
            for k, v in self.quality_thresholds
        ):
            raise ValueError("Quality thresholds must lie in [0,1]")


class DatasetExperimentV4:
    def __init__(
        self,
        runtime,
        manifest,
        *,
        scorer=None,
        settings=None,
        validation_manifest=None,
        output_dir="v4-results",
        search_space=None,
        fixed_config=None,
        managed=True,
        validate_scorer=True,
    ):
        self.runtime = runtime
        self.manifest = manifest.verify()
        self.validation_manifest = validation_manifest
        self.scorer = scorer or ScorerSettings()
        self.settings = settings or ExperimentSettings()
        self.output_dir = Path(output_dir)
        self.search_space = json.loads(canonical(search_space or {}))
        self.fixed_config = json.loads(canonical(fixed_config or {}))
        self.managed = managed
        self.validate_scorer = validate_scorer
        if set(self.fixed_config) & set(self.search_space):
            raise ValueError("Fixed and sampled configuration keys overlap")
        if not managed and (self.search_space or self.fixed_config):
            raise ValueError("Unmanaged runs cannot tune server settings")
        if any(
            not isinstance(values, list) or not values
            for values in self.search_space.values()
        ):
            raise ValueError("Search space values must be nonempty categorical lists")
        self._preflight(manifest)
        allowed = {
            "conversation": set(),
            "gpqa_diamond": {"accuracy"},
            "humaneval": {"pass@1"},
            "humaneval_plus": {"pass@1"},
            "extraction": {"json_validity", "schema_compliance", "field_accuracy"},
        }
        if self.settings.quality_thresholds and not self.validate_scorer:
            raise ValueError("Thresholds require quality scoring")
        if (
            set(dict(self.settings.quality_thresholds))
            - allowed[manifest.document["task"]]
        ):
            raise ValueError("Quality threshold does not apply to this task")
        if manifest.document["role"] != "train":
            raise ValueError("Optimization requires a training manifest")
        if validation_manifest is not None:
            self._preflight(validation_manifest)
            train, valid = manifest.document, validation_manifest.document
            if valid["role"] != "validation":
                raise ValueError("Held-out manifest must have validation role")
            for key in (
                "task",
                "dataset_id",
                "revision",
                "dataset_config",
                "model",
                "model_revision",
                "tokenizer_id",
                "tokenizer_revision",
                "tokenizer_settings",
                "generation",
                "evaluation",
            ):
                if train[key] != valid[key]:
                    raise ValueError(f"Held-out settings differ: {key}")
            if {e["example_id"] for e in train["examples"]} & {
                e["example_id"] for e in valid["examples"]
            }:
                raise ValueError("Training and held-out example IDs overlap")
            # Also catch identical payloads assigned different IDs.
            if {e["payload_sha256"] for e in train["examples"]} & {
                e["payload_sha256"] for e in valid["examples"]
            }:
                raise ValueError("Training and held-out payloads overlap")
        settings_identity = asdict(self.settings)
        settings_identity.pop("trials")  # Extending the same study is allowed.
        self.identity = {
            "version": "experiments-v4.1",
            "workload": manifest.signature,
            "dataset": {
                key: manifest.document[key]
                for key in (
                    "dataset_id",
                    "revision",
                    "dataset_config",
                    "split",
                    "source_format",
                    "task",
                    "generation",
                    "sampling",
                    "model_revision",
                    "tokenizer_revision",
                )
            },
            "validation": validation_manifest.signature
            if validation_manifest
            else None,
            "runtime": runtime.identity,
            "scorer": self.scorer.identity,
            "settings": settings_identity,
            "search_space": self.search_space,
            "fixed_config": self.fixed_config,
            "managed": managed,
            "validate_scorer": validate_scorer,
        }
        self.signature = digest(self.identity)
        self.study_name = f"datasets-v4-{runtime.framework}-{self.signature}"
        self.directory = self.output_dir / self.study_name

    def _preflight(self, manifest):
        self.runtime.preflight(manifest)
        if manifest.document["evaluation"] != self.scorer.identity:
            raise ValueError(
                "Scorer settings/version differ from the frozen evaluation"
            )
        if not self.validate_scorer:
            return
        if (
            manifest.document["task"] in {"humaneval", "humaneval_plus"}
            and self.scorer.image is None
        ):
            raise ValueError(
                "HumanEval requires a digest-pinned scorer container image"
            )
        if manifest.document["task"] == "extraction":
            from jsonschema import Draft202012Validator

            for reference in manifest.references.values():
                Draft202012Validator.check_schema(reference["schema"])

    def evaluate(self, manifest, config, *, label):
        self._preflight(manifest)
        if self.validate_scorer:
            check_scorer_runtime(manifest.document["task"], self.scorer)
        measured = self.runtime.generate(
            manifest,
            native_config=config,
            output_dir=self.directory,
            managed=self.managed,
        )
        doc, references = manifest.document, manifest.references
        outputs = measured.pop("completions")
        expected_ids = {e["example_id"] for e in doc["examples"]}
        if (
            len(outputs) != len(expected_ids)
            or {c.example_id for c in outputs} != expected_ids
        ):
            raise ValueError("Runtime completions do not match frozen examples")
        # No inference clock or telemetry collector is running during scoring.
        started = time.perf_counter()
        rows = [
            score(
                doc["task"] if self.validate_scorer else "conversation",
                c,
                references[c.example_id],
                self.scorer,
            )
            for c in outputs
        ]
        aggregate = aggregate_scores(rows)
        scorer_seconds = time.perf_counter() - started
        quality = aggregate["quality"]
        feasible = (
            not aggregate["scorer_failures"]
            and not aggregate["generation_failures"]
            and all(
                quality.get(k, -1) >= v for k, v in self.settings.quality_thresholds
            )
        )
        result = {
            "label": label,
            "study_signature": self.signature,
            "manifest": manifest.signature,
            "dataset_id": doc["dataset_id"],
            "task": doc["task"],
            "role": doc["role"],
            "generation": doc["generation"],
            "native_config": config,
            **measured,
            "scores": rows,
            "quality": quality,
            "accounting": aggregate,
            "scoring_seconds": scorer_seconds,
            "quality_feasible": feasible,
            "completions": [asdict(c) for c in outputs],
        }
        self.directory.mkdir(parents=True, exist_ok=True)
        # Append-only evidence also preserves repeated/resumed validation attempts.
        with (self.directory / "results.jsonl").open("a") as stream:
            stream.write(canonical(result) + "\n")
        return result

    def run(self, *, storage=None):
        import optuna

        from llm_optimizer.wandb_tracking import WandbTracker

        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "identity.json").write_text(canonical(self.identity))
        self.manifest.save(self.directory / "workloads")
        if self.validation_manifest:
            self.validation_manifest.save(self.directory / "workloads")
        tracker = (
            WandbTracker.start(
                study_name=self.study_name,
                framework=self.runtime.framework,
                model=self.runtime.model,
                gpu=self.settings.gpu_identity,
                run_id=self.signature,
                objective="output_tokens_per_second",
            )
            if self.settings.wandb_enabled
            else WandbTracker()
        )
        if tracker.run is not None:
            try:
                tracker.run.config.update(self.identity)
            except Exception:
                pass  # Tracking failures cannot fail optimization.
        study = optuna.create_study(
            study_name=self.study_name,
            storage=storage or f"sqlite:///{(self.directory / 'study.db').resolve()}",
            load_if_exists=True,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.settings.seed),
        )
        existing = study.user_attrs.get("v4_identity")
        if existing is not None and canonical(existing) != canonical(self.identity):
            raise ValueError("Study identity mismatch")
        study.set_user_attr("v4_identity", self.identity)

        def objective(trial):
            config = {
                **self.fixed_config,
                **{
                    k: trial.suggest_categorical(k, values)
                    for k, values in self.search_space.items()
                },
            }
            try:
                result = self.evaluate(
                    self.manifest, config, label=f"trial-{trial.number}"
                )
            except RuntimeError as exc:
                trial.set_user_attr("runtime_error", f"{type(exc).__name__}: {exc}")
                raise
            flattened = {
                **result["metrics"],
                **{f"quality/{k}": v for k, v in result["quality"].items()},
                "scorer_failures": result["accounting"]["scorer_failures"],
                "scoring_seconds": result["scoring_seconds"],
            }
            trial.set_user_attr("metrics", flattened)
            trial.set_user_attr("quality_feasible", result["quality_feasible"])
            trial.set_user_attr("execution", result["execution"])
            trial.set_user_attr("native_config", config)
            if not result["quality_feasible"]:
                raise optuna.TrialPruned(
                    "Quality threshold or generation/scorer failure"
                )
            value = result["metrics"]["output_tokens_per_second"]
            if value is None or not math.isfinite(value):
                raise optuna.TrialPruned("Missing measured output token usage")
            return value

        def callback(study, trial):
            tracker.trial(trial, trial.user_attrs.get("metrics", {}))

        validation = None
        try:
            study.optimize(
                objective,
                n_trials=self.settings.trials,
                callbacks=[callback],
                catch=(RuntimeError,),
            )
            complete = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))
            if complete and self.validation_manifest is not None:
                # Select solely on training results. Validation never feeds suggestions.
                best = study.best_trial
                validation = self.evaluate(
                    self.validation_manifest,
                    best.user_attrs["native_config"],
                    label=f"held-out-trial-{best.number}",
                )
                study.set_user_attr(
                    "held_out",
                    {
                        "trial": best.number,
                        "quality": validation["quality"],
                        "metrics": validation["metrics"],
                        "feasible": validation["quality_feasible"],
                    },
                )
                if tracker.run is not None:
                    try:
                        tracker.run.summary["held_out"] = study.user_attrs["held_out"]
                    except Exception:
                        pass
            tracker.artifacts(
                [self.directory / "identity.json", self.directory / "results.jsonl"],
                name="v4-results-" + self.signature[:16],
            )
        finally:
            tracker.finish()
        return study, validation
