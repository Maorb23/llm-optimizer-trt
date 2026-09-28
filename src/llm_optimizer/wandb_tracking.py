"""Optional, failure-isolated Weights & Biases tracking."""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def stable_run_id(run_id: str, study_name: str, framework: str) -> str:
    """Return the same W&B-safe ID for every attempt of an optimization job."""
    # The remote worker uploads its final manifest after the CLI process exits and
    # cannot reliably reconstruct an implicit signature-derived study name.  The
    # durable controller run ID plus framework is the job's stable identity.
    value = f"{run_id}\0{framework}".encode()
    return hashlib.sha256(value).hexdigest()[:24]


def _tags() -> list[str]:
    return [
        tag.strip()
        for tag in os.getenv("WANDB_TAGS", "").split(",")
        if tag.strip()
    ]


class WandbTracker:
    """Small wrapper that guarantees telemetry failures cannot fail a search."""

    def __init__(self, run: Any = None):
        self.run = run

    @classmethod
    def start(
        cls,
        *,
        study_name: str,
        framework: str,
        model: str,
        gpu: str,
        run_id: str,
        objective: str,
    ) -> WandbTracker:
        project = os.getenv("WANDB_PROJECT")
        if not project:
            return cls()
        try:
            import wandb  # type: ignore[import-not-found]

            wandb_id = stable_run_id(run_id, study_name, framework)
            run = wandb.init(
                project=project,
                entity=os.getenv("WANDB_ENTITY") or None,
                name=os.getenv("WANDB_RUN_NAME") or None,
                tags=_tags() or None,
                mode=os.getenv("WANDB_MODE") or None,
                id=wandb_id,
                resume="allow",
                config={
                    "run_id": run_id,
                    "study_name": study_name,
                    "framework": framework,
                    "model": model,
                    "gpu": gpu,
                    "objective": objective,
                },
                settings=wandb.Settings(silent=True),
            )
            return cls(run)
        except Exception:
            # SDK exceptions may contain request metadata. Never risk logging a key.
            logger.warning("W&B initialization failed; continuing without tracking")
            return cls()

    def trial(self, trial: Any, metrics: dict[str, Any] | None = None) -> None:
        if self.run is None:
            return
        try:
            payload: dict[str, Any] = {
                "trial/number": trial.number,
                "trial/state": trial.state.name,
            }
            if trial.value is not None:
                payload["trial/objective"] = trial.value
            payload.update(
                {f"parameters/{key}": value for key, value in trial.params.items()}
            )
            for key, value in (metrics or {}).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    payload[f"benchmark/{key}"] = value
            self.run.log(payload, step=trial.number)
        except Exception:
            logger.warning("W&B trial logging failed; continuing optimization")

    def artifacts(self, paths: Iterable[Path], *, name: str) -> None:
        if self.run is None:
            return
        try:
            import wandb  # type: ignore[import-not-found]

            artifact = wandb.Artifact(name=name, type="optimization-results")
            added = False
            for path in paths:
                if path.is_file():
                    artifact.add_file(str(path), name=path.name)
                    added = True
            if added:
                self.run.log_artifact(artifact)
        except Exception:
            logger.warning("W&B artifact upload failed; continuing")

    def finish(self) -> None:
        if self.run is None:
            return
        try:
            self.run.finish()
        except Exception:
            logger.warning("W&B shutdown failed; continuing")
