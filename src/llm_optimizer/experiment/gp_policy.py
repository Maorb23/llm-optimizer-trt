"""Reject previously failed effective configurations without changing samplers."""

import hashlib
import json
from pathlib import Path

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
    def __init__(self, study, run_dir, identity, fixed_server=None, max_rejections=10):
        if (
            isinstance(max_rejections, bool)
            or not isinstance(max_rejections, int)
            or max_rejections < 1
        ):
            raise ValueError("GP max_rejections must be a positive integer")
        self.study = study
        self.identity = identity
        self.fixed_server = fixed_server or {}
        self.max_rejections = max_rejections
        label = hashlib.sha256(study.study_name.encode()).hexdigest()[:16]
        self.path = Path(run_dir) / f"gp-failures-{label}.json"
        self.data = (
            json.loads(self.path.read_text())
            if self.path.exists()
            else {
                "study": study.study_name,
                "identity": identity,
                "max_consecutive_rejections": max_rejections,
                "failures": [],
                "rejections": [],
                "stops": [],
            }
        )
        if self.data["identity"] != identity:
            raise ValueError("GP failure log has a different execution identity")
        self.blocked = {
            row["fingerprint"]
            for row in self.data["failures"]
            if row.get("blocked", True)
        }
        recorded = {row["trial_number"] for row in self.data["failures"]}
        for trial in study.trials:
            if (
                trial.state == TrialState.FAIL
                and (
                    trial.user_attrs.get("effective_config")
                    or trial.user_attrs.get("config")
                )
                and not trial.user_attrs.get("gp_rejected")
                and not trial.user_attrs.get("gp_non_config_failure")
                and trial.number not in recorded
            ):
                # Prefer the resolved snapshot: generated YAML files may no longer
                # exist after moving a study or restoring it on another worker.
                config = trial.user_attrs.get("effective_config")
                if config is None:
                    config = trial.user_attrs["config"]
                effective, fingerprint = self.fingerprint(config)
                self.failure(
                    trial.number,
                    effective,
                    fingerprint,
                    trial.user_attrs.get("failure_reason", "Previously failed trial"),
                )
        self.save()

    def fingerprint(self, config):
        effective = canonical_config({**self.fixed_server, **config})
        payload = json.dumps([self.identity, effective], sort_keys=True)
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
            trial.set_user_attr("gp_rejected", True)
            self.data["rejections"].append(
                {
                    "trial_number": trial.number,
                    "fingerprint": fingerprint,
                    "effective_config": effective,
                    "reason": "Previously failed configuration",
                    "server_launched": False,
                }
            )
            self.save()
            raise BlockedGPProposal(fingerprint)

    def failure(self, number, effective, fingerprint, reason, blocked=True):
        if blocked:
            self.blocked.add(fingerprint)
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

    def optimize(self, objective, n_trials, failure_type, callbacks):
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
            for callback in callbacks:
                callback(study, frozen)

        def guarded_objective(trial):
            try:
                return objective(trial)
            except (BlockedGPProposal, failure_type):
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
