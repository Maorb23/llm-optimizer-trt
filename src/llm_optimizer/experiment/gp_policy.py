"""Reject failed and pruned effective configurations without changing samplers."""

import hashlib
import json
from pathlib import Path

from optuna import TrialPruned
from optuna.trial import TrialState


class BlockedGPProposal(Exception):
    """A proposal was rejected before benchmark execution."""


def canonical_config(config):
    """Compare YAML contents, not generated YAML filenames."""
    result = dict(config)
    for key in ("config", "extra_llm_api_options"):
        if isinstance(result.get(key), str) and Path(result[key]).is_file():
            import yaml

            result[key] = yaml.safe_load(Path(result[key]).read_text())

    def normalize(value):
        if isinstance(value, float):
            return round(value, 12)
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        return value

    return json.loads(json.dumps(normalize(result), sort_keys=True, allow_nan=False))


class GPFailurePolicy:
    def __init__(
        self, study, run_dir, identity, fixed_server=None, max_rejections=10, scope=None
    ):
        if (
            isinstance(max_rejections, bool)
            or not isinstance(max_rejections, int)
            or max_rejections < 1
        ):
            raise ValueError("GP max_rejections must be a positive integer")
        self.study = study
        self.identity = canonical_config(identity)
        contract = study.user_attrs.get("contract", {})
        contract = {
            key: value
            for key, value in contract.items()
            if key not in {"budget_trials", "budget_gpu_seconds", "quick_demo"}
        }
        self.scope = canonical_config(scope if scope is not None else contract)
        self.fixed_server = fixed_server or {}
        self.scope = {
            "experiment": self.scope,
            "fixed_server": canonical_config(self.fixed_server),
            "pruner_class": type(study.pruner).__name__,
            "pruner_parameters": {
                key: repr(value)
                for key, value in vars(study.pruner).items()
                if isinstance(value, (str, int, float, bool, type(None)))
            },
        }
        execution_identity = {"identity": self.identity, "scope": self.scope}
        previous_identity = study.user_attrs.get("gp_rejection_identity")
        if previous_identity is not None and previous_identity != execution_identity:
            raise ValueError("GP rejection history has a different experiment scope")
        self.max_rejections = max_rejections
        label = hashlib.sha256(study.study_name.encode()).hexdigest()[:16]
        self.path = Path(run_dir) / f"gp-failures-{label}.json"
        self.data = (
            json.loads(self.path.read_text())
            if self.path.exists()
            else {
                "study": study.study_name,
                "identity": self.identity,
                "scope": self.scope,
                "pruned": [],
                "max_consecutive_rejections": max_rejections,
                "failures": [],
                "rejections": [],
                "stops": [],
            }
        )
        if self.data["identity"] != self.identity:
            raise ValueError("GP failure log has a different execution identity")
        if "scope" in self.data and self.data["scope"] != self.scope:
            raise ValueError("GP rejection log has a different experiment scope")
        # Legacy logs inherit the study's already validated execution contract.
        self.data["scope"] = self.scope
        self.data.setdefault("pruned", [])
        self.data["max_consecutive_rejections"] = max_rejections
        self.blocked_sources = {}
        # Recompute legacy fingerprints with the scoped identity on resume.
        for category in ("failures", "pruned"):
            for row in self.data[category]:
                _, row["fingerprint"] = self.fingerprint(row["effective_config"])
                if row.get("blocked", True):
                    self.blocked_sources.setdefault(
                        row["fingerprint"],
                        {
                            "original_trial_number": row["trial_number"],
                            "original_state": "PRUNED"
                            if category == "pruned"
                            else "FAIL",
                        },
                    )
        self.blocked = set(self.blocked_sources)
        recorded = {
            row["trial_number"]
            for category in ("failures", "pruned")
            for row in self.data[category]
        }
        for trial in study.trials:
            if (
                trial.state in (TrialState.FAIL, TrialState.PRUNED)
                and (
                    trial.user_attrs.get("effective_config")
                    or trial.user_attrs.get("config")
                )
                and not trial.user_attrs.get("gp_rejected")
                and (
                    trial.state == TrialState.PRUNED
                    or not trial.user_attrs.get("gp_non_config_failure")
                )
                and trial.number not in recorded
            ):
                # Prefer the resolved snapshot: generated YAML files may no longer
                # exist after moving a study or restoring it on another worker.
                config = trial.user_attrs.get("effective_config")
                if config is None:
                    config = trial.user_attrs["config"]
                effective, fingerprint = self.fingerprint(config)
                recorder = (
                    self.pruned if trial.state == TrialState.PRUNED else self.failure
                )
                recorder(
                    trial.number,
                    effective,
                    fingerprint,
                    trial.user_attrs.get("failure_reason")
                    or (
                        "Previously pruned configuration"
                        if trial.state == TrialState.PRUNED
                        else "Previously failed trial"
                    ),
                )
        study.set_user_attr("gp_rejection_identity", execution_identity)
        self.save()

    def fingerprint(self, config):
        effective = canonical_config({**self.fixed_server, **config})
        payload = json.dumps([self.identity, self.scope, effective], sort_keys=True)
        return effective, hashlib.sha256(payload.encode()).hexdigest()

    def save(self):
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.data, indent=2, allow_nan=False))
        temporary.replace(self.path)

    def check(self, trial, config):
        effective, fingerprint = self.fingerprint(config)
        trial.set_user_attr("effective_config", effective)
        trial.set_user_attr("effective_config_fingerprint", fingerprint)
        if fingerprint in self.blocked:
            source = self.blocked_sources[fingerprint]
            reason = (
                "Previously pruned configuration"
                if source["original_state"] == "PRUNED"
                else "Previously failed configuration"
            )
            trial.set_user_attr("gp_rejected", True)
            trial.set_user_attr("gp_rejection_reason", reason)
            trial.set_user_attr(
                "gp_original_trial_number", source["original_trial_number"]
            )
            trial.set_user_attr("benchmark_executed", False)
            self.data["rejections"].append(
                {
                    "trial_number": trial.number,
                    "fingerprint": fingerprint,
                    "effective_config": effective,
                    "reason": reason,
                    **source,
                    "server_launched": False,
                }
            )
            self.save()
            raise BlockedGPProposal(fingerprint)

    def failure(self, number, effective, fingerprint, reason, blocked=True):
        if blocked:
            self.blocked.add(fingerprint)
            self.blocked_sources.setdefault(
                fingerprint,
                {
                    "original_trial_number": number,
                    "original_state": "FAIL",
                },
            )
        self.data["failures"].append(
            {
                "trial_number": number,
                "fingerprint": fingerprint,
                "effective_config": effective,
                "reason": reason,
                "blocked": blocked,
            }
        )
        self.save()

    def pruned(
        self, number, effective, fingerprint, reason="Previously pruned configuration"
    ):
        self.blocked.add(fingerprint)
        self.blocked_sources.setdefault(
            fingerprint,
            {
                "original_trial_number": number,
                "original_state": "PRUNED",
            },
        )
        if not any(row["trial_number"] == number for row in self.data["pruned"]):
            self.data["pruned"].append(
                {
                    "trial_number": number,
                    "fingerprint": fingerprint,
                    "effective_config": effective,
                    "reason": reason,
                }
            )
        self.save()

    def optimize(self, objective, n_trials, failure_type, callbacks):
        self.study.set_user_attr("gp_stop_reason", None)
        executed = 0
        rejected = 0
        latest = None

        def record(study, frozen):
            nonlocal latest
            latest = frozen
            if frozen.user_attrs.get("gp_rejected"):
                return
            if frozen.state == TrialState.FAIL and frozen.user_attrs.get(
                "effective_config_fingerprint"
            ):
                self.failure(
                    frozen.number,
                    frozen.user_attrs["effective_config"],
                    frozen.user_attrs["effective_config_fingerprint"],
                    frozen.user_attrs.get("failure_reason", "Benchmark failed"),
                    blocked=not frozen.user_attrs.get("gp_non_config_failure", False),
                )
            if frozen.state == TrialState.PRUNED and frozen.user_attrs.get(
                "effective_config_fingerprint"
            ):
                self.pruned(
                    frozen.number,
                    frozen.user_attrs["effective_config"],
                    frozen.user_attrs["effective_config_fingerprint"],
                )
            for callback in callbacks:
                callback(study, frozen)

        def guarded_objective(trial):
            try:
                return objective(trial)
            except (BlockedGPProposal, failure_type, TrialPruned):
                raise
            except BaseException:
                trial.set_user_attr("gp_non_config_failure", True)
                raise

        while executed < n_trials:
            try:
                self.study.optimize(
                    guarded_objective,
                    n_trials=1,
                    n_jobs=1,
                    catch=(BlockedGPProposal, failure_type),
                    callbacks=[record],
                )
            except BaseException as exc:
                failed = [t for t in self.study.trials if t.state != TrialState.WAITING]
                if failed:
                    trial = failed[-1]
                    if trial.user_attrs.get("effective_config_fingerprint") and not any(
                        row["trial_number"] == trial.number
                        for row in self.data["failures"]
                    ):
                        self.failure(
                            trial.number,
                            trial.user_attrs["effective_config"],
                            trial.user_attrs["effective_config_fingerprint"],
                            f"{type(exc).__name__}: {exc}",
                            blocked=False,
                        )
                self.data["stops"].append({"reason": f"{type(exc).__name__}: {exc}"})
                self.save()
                raise
            frozen = latest
            if frozen.user_attrs.get("gp_rejected"):
                rejected += 1
                if rejected >= self.max_rejections:
                    diagnostic = (
                        f"GP stopped after {rejected} consecutive blocked proposals; "
                        f"no fallback sampler used. See {self.path}"
                    )
                    self.data["stops"].append(
                        {"trial_number": frozen.number, "reason": diagnostic}
                    )
                    self.study.set_user_attr("gp_stop_reason", diagnostic)
                    self.save()
                    print(diagnostic)
                    break
            else:
                executed += 1
                rejected = 0
            if self.study._stop_flag:
                break
