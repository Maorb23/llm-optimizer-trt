"""Persistent, non-GPU controller with explicit reconciliation ticks."""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .adapters import NebiusAdapter, NebiusCommandError, S3Artifacts
from .contracts import OptimizationRequest
from .service import controller_lease

TERMINAL = {"completed", "failed", "cancelled", "dry_run"}
logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Controller:
    """Call tick periodically; API calls return immediately."""

    def __init__(
        self,
        database: str | Path,
        nebius: Any = None,
        s3_factory: Any = S3Artifacts,
        concurrency: int = 2,
    ):
        if concurrency < 1:
            raise ValueError("concurrency must be positive")
        self.database = str(database)
        self.nebius = nebius or NebiusAdapter()
        self.s3_factory = s3_factory
        self.concurrency = concurrency
        self._thread_lock = threading.RLock()
        self._owns_lease = False
        with self._db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, idempotency_key TEXT UNIQUE,
                    request TEXT NOT NULL,
                    created_at TEXT NOT NULL, cancelled INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    run_id TEXT NOT NULL, framework TEXT NOT NULL, state TEXT NOT NULL,
                    nebius_id TEXT, error TEXT, progress TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, framework),
                    FOREIGN KEY (run_id) REFERENCES runs(id)
                );
            """)
            if "progress" not in {
                column[1] for column in db.execute("PRAGMA table_info(jobs)")
            }:
                db.execute("ALTER TABLE jobs ADD COLUMN progress TEXT")
            # Older versions stored CalledProcessError text, which included
            # --env credentials. Replace it when opening an existing database.
            db.execute(
                "UPDATE jobs SET error=? WHERE error LIKE ? OR error LIKE ? "
                "OR error LIKE ? OR error LIKE ? OR error LIKE ?",
                (
                    "Previous Nebius submission error redacted; inspect job by name",
                    "%--env%",
                    "%AWS_%",
                    "%HF_TOKEN=%",
                    "%LLM_OPTIMIZER_PAYLOAD_B64=%",
                    "%WANDB_API_KEY=%",
                ),
            )

    def _db(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.database, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    @staticmethod
    def _safe_error(exc: Exception) -> str:
        if isinstance(exc, NebiusCommandError):
            return str(exc)
        # Untrusted exceptions can embed subprocess arguments or credentials.
        return f"{type(exc).__name__} during controller operation"

    @staticmethod
    def _public_failure_line(value: str) -> str | None:
        line = value.strip().splitlines()[-1][:500] if value.strip() else ""
        if not line or any(
            marker in line.lower()
            for marker in (
                "aws_",
                "hf_token",
                "wandb_api_key",
                "--env",
                "password",
                "secret",
                "://",
            )
        ):
            return None
        for key in (
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "HF_TOKEN",
            "WANDB_API_KEY",
        ):
            secret = os.getenv(key)
            if secret and secret in line:
                return None
        return (
            line
            if re.match(r"^[\w.]+(?:Error|Exception|Exit|Interrupt): ", line)
            else None
        )

    def _failed_job_error(
        self, row: sqlite3.Row, request: dict[str, Any], state: str
    ) -> str:
        error = f"Nebius state: {state}"
        try:
            store = self.s3_factory(request["bucket"], request["prefix"])
            manifest = store.manifest(row["run_id"], row["framework"])
            if not manifest or (manifest.get("run_id"), manifest.get("framework")) != (
                row["run_id"],
                row["framework"],
            ):
                return error
            worker_error = self._public_failure_line(str(manifest.get("error", "")))
            if worker_error:
                error += f"; worker: {worker_error}"
            log_tail = getattr(store, "execution_log_tail", lambda *args: None)(
                row["run_id"], row["framework"]
            )
            if log_tail:
                cause = next(
                    (
                        line
                        for raw in reversed(log_tail.splitlines())
                        if (line := self._public_failure_line(raw))
                    ),
                    None,
                )
                if cause and cause != worker_error:
                    error += f"; cause: {cause}"
        except Exception as exc:
            logger.warning(
                "Could not read failure artifacts for %s/%s: %s",
                row["run_id"],
                row["framework"],
                self._safe_error(exc),
            )
        return error

    def submit(
        self, data: dict[str, Any], idempotency_key: str | None = None
    ) -> dict[str, Any]:
        request = OptimizationRequest.parse(data)
        run_id = uuid.uuid4().hex
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            if idempotency_key:
                previous = db.execute(
                    "SELECT id, request FROM runs WHERE idempotency_key=?",
                    (idempotency_key,),
                ).fetchone()
                if previous:
                    if json.loads(previous["request"]) != request.to_dict():
                        raise ValueError(
                            "idempotency key belongs to a different request"
                        )
                    return self.status(previous["id"])
            active = db.execute(
                "SELECT run_id FROM jobs WHERE state NOT IN "
                "('completed','failed','cancelled','dry_run') LIMIT 1"
            ).fetchone()
            if active:
                raise ValueError(f"Optimizer study {active['run_id']} is still active; no new study submitted")
            db.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, 0)",
                (run_id, idempotency_key, json.dumps(request.to_dict()), _now()),
            )
            db.executemany(
                "INSERT INTO jobs (run_id, framework, state, nebius_id, error, "
                "progress, updated_at) VALUES (?, ?, 'queued', NULL, NULL, NULL, ?)",
                [(run_id, name, _now()) for name in request.frameworks],
            )
        return self.status(run_id)

    def _update(
        self,
        run_id: str,
        framework: str,
        state: str,
        nebius_id: str | None = None,
        error: str | None = None,
    ) -> None:
        with self._db() as db:
            db.execute(
                "UPDATE jobs SET state=?, nebius_id=COALESCE(?, nebius_id), "
                "error=?, updated_at=? WHERE run_id=? AND framework=?",
                (state, nebius_id, error, _now(), run_id, framework),
            )

    def status(self, run_id: str) -> dict[str, Any]:
        with self._db() as db:
            run = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            if run is None:
                raise KeyError(run_id)
            jobs = [
                dict(row)
                for row in db.execute(
                    "SELECT * FROM jobs WHERE run_id=? ORDER BY framework", (run_id,)
                )
            ]
        for job in jobs:
            job["progress"] = json.loads(job["progress"]) if job["progress"] else None
        states = {job["state"] for job in jobs}
        state = (
            "cancelling"
            if run["cancelled"] and not states <= TERMINAL
            else "cancelled"
            if run["cancelled"]
            else "dry_run"
            if states == {"dry_run"}
            else "completed"
            if states == {"completed"}
            else "cancelled"
            if states == {"cancelled"}
            else "partial"
            if states <= TERMINAL and "completed" in states
            else "failed"
            if states <= TERMINAL
            else "running"
        )
        return {
            "run_id": run_id,
            "state": state,
            "created_at": run["created_at"],
            "jobs": jobs,
        }

    @contextmanager
    def exclusive(self):
        with self._thread_lock:
            if self._owns_lease:
                yield
            else:
                with controller_lease(self.database):
                    self._owns_lease = True
                    try:
                        yield
                    finally:
                        self._owns_lease = False

    def tick(self) -> None:
        """Submit and poll jobs under the same lease used by serve."""
        with self.exclusive():
            self._tick()

    def _tick(self) -> None:
        with self._db() as db:
            uncertain = db.execute(
                "SELECT run_id, framework, state FROM jobs WHERE state IN "
                "('submitting','submission_unknown','cancelling')"
            ).fetchall()
        for row in uncertain:
            try:
                finder = getattr(self.nebius, "find_by_name", None)
                job_id = finder(row["run_id"], row["framework"]) if finder else None
                if job_id:
                    if row["state"] == "cancelling":
                        self.nebius.cancel(job_id)
                        self._update(
                            row["run_id"],
                            row["framework"],
                            "cancelled",
                            nebius_id=job_id,
                        )
                    else:
                        self._update(
                            row["run_id"],
                            row["framework"],
                            "submitted",
                            nebius_id=job_id,
                        )
                elif row["state"] == "submitting":
                    self._update(row["run_id"], row["framework"], "submission_unknown")
            except Exception as exc:
                error = self._safe_error(exc)
                logger.error(
                    "Nebius reconciliation failed for %s/%s: %s",
                    row["run_id"],
                    row["framework"],
                    error,
                )
                self._update(
                    row["run_id"],
                    row["framework"],
                    "submission_unknown",
                    error=error,
                )
        with self._db() as db:
            active = db.execute(
                "SELECT COUNT(*) FROM jobs WHERE state IN "
                "('submitting','submitted','running','submission_unknown')"
            ).fetchone()[0]
            queued = db.execute(
                "SELECT j.run_id, j.framework, r.request FROM jobs j "
                "JOIN runs r ON r.id=j.run_id WHERE j.state='queued' "
                "AND r.cancelled=0 ORDER BY r.created_at, j.framework"
            ).fetchall()
        for row in queued[: max(0, self.concurrency - active)]:
            run_id, framework = row["run_id"], row["framework"]
            request = json.loads(row["request"])
            with self._db() as db:
                claimed = db.execute(
                    "UPDATE jobs SET state='submitting', updated_at=? "
                    "WHERE run_id=? AND framework=? AND state='queued'",
                    (_now(), run_id, framework),
                ).rowcount
            if not claimed:
                continue
            try:
                job_id = self.nebius.submit(
                    {"run_id": run_id, "framework": framework, "request": request}
                )
            except Exception as exc:
                # The provider may have accepted the job before the response failed.
                error = self._safe_error(exc)
                logger.error(
                    "Nebius submission uncertain for %s/%s: %s. "
                    "Check Nebius for job name %s before retrying.",
                    run_id,
                    framework,
                    error,
                    NebiusAdapter.name(run_id, framework),
                )
                with self._db() as db:
                    db.execute(
                        "UPDATE jobs SET state='submission_unknown', error=?, "
                        "updated_at=? WHERE run_id=? AND framework=? "
                        "AND state='submitting'",
                        (error, _now(), run_id, framework),
                    )
            else:
                with self._db() as db:
                    recorded = db.execute(
                        "UPDATE jobs SET state=?, nebius_id=?, updated_at=? "
                        "WHERE run_id=? AND framework=? AND state='submitting'",
                        (
                            "dry_run" if request["dry_run"] else "submitted",
                            job_id,
                            _now(),
                            run_id,
                            framework,
                        ),
                    ).rowcount
                if not recorded:
                    self.nebius.cancel(job_id)
                    self._update(run_id, framework, "cancelled", nebius_id=job_id)
        with self._db() as db:
            active_jobs = db.execute(
                "SELECT j.*, r.request FROM jobs j JOIN runs r ON r.id=j.run_id "
                "WHERE j.state IN ('submitted','running')"
            ).fetchall()
        for row in active_jobs:
            try:
                state = self.nebius.status(row["nebius_id"])
                request = json.loads(row["request"])
                try:
                    progress = self.s3_factory(
                        request["bucket"], request["prefix"]
                    ).progress(row["run_id"], row["framework"])
                    if progress:
                        with self._db() as db:
                            db.execute(
                                "UPDATE jobs SET progress=? WHERE run_id=? "
                                "AND framework=?",
                                (json.dumps(progress), row["run_id"], row["framework"]),
                            )
                except Exception:
                    pass
                if state in {"succeeded", "success", "done", "completed"}:
                    store = self.s3_factory(request["bucket"], request["prefix"])
                    manifest = store.manifest(row["run_id"], row["framework"])
                    if not manifest:
                        self._update(
                            row["run_id"],
                            row["framework"],
                            "failed",
                            error="missing framework result manifest",
                        )
                        continue
                    if (
                        manifest.get("run_id") != row["run_id"]
                        or manifest.get("framework") != row["framework"]
                    ):
                        self._update(
                            row["run_id"],
                            row["framework"],
                            "failed",
                            error="invalid framework result manifest",
                        )
                        continue
                    if manifest.get("status") != "completed":
                        self._update(
                            row["run_id"],
                            row["framework"],
                            "failed",
                            error=manifest.get("error", "worker failed"),
                        )
                    else:
                        self._update(row["run_id"], row["framework"], "completed")
                elif state in {"failed", "error", "timed_out", "timeout"}:
                    error = self._failed_job_error(row, request, state)
                    self._update(
                        row["run_id"],
                        row["framework"],
                        "failed",
                        error=error,
                    )
                    logger.error(
                        "Job %s/%s failed: %s", row["run_id"], row["framework"], error
                    )
                elif state in {"cancelled", "canceled"}:
                    self._update(row["run_id"], row["framework"], "cancelled")
                else:
                    self._update(row["run_id"], row["framework"], "running")
            except Exception as exc:
                # Transient control-plane or S3 errors are retried on the next tick.
                error = self._safe_error(exc)
                logger.error(
                    "Nebius tracking failed for %s/%s: %s",
                    row["run_id"],
                    row["framework"],
                    error,
                )
                self._update(row["run_id"], row["framework"], row["state"], error=error)
        with self._db() as db:
            run_rows = db.execute("SELECT id, request FROM runs").fetchall()
        for row in run_rows:
            request = json.loads(row["request"])
            if request["dry_run"]:
                continue
            status = self.status(row["id"])
            if all(job["state"] in TERMINAL for job in status["jobs"]):
                try:
                    report = self.results(row["id"])
                    self.s3_factory(request["bucket"], request["prefix"]).put_report(
                        row["id"], report
                    )
                except Exception:
                    # Reporting can retry on the next tick without changing jobs.
                    pass

    def results(self, run_id: str) -> dict[str, Any]:
        status = self.status(run_id)
        with self._db() as db:
            request = json.loads(
                db.execute("SELECT request FROM runs WHERE id=?", (run_id,)).fetchone()[
                    0
                ]
            )
        store = (
            None
            if request["dry_run"]
            else self.s3_factory(request["bucket"], request["prefix"])
        )
        manifests = {}
        for job in status["jobs"]:
            if store and job["state"] in {"completed", "failed", "running"}:
                try:
                    manifest = store.manifest(run_id, job["framework"])
                    if (
                        manifest
                        and manifest.get("run_id") == run_id
                        and manifest.get("framework") == job["framework"]
                    ):
                        manifests[job["framework"]] = manifest
                except Exception:
                    pass
        signatures = {
            (
                m.get("model"),
                m.get("resolved_revision") or m.get("revision"),
                m.get("gpu_type"),
                m.get("gpu_count"),
                json.dumps(m.get("workload"), sort_keys=True),
                m.get("workload_sha256"),
                json.dumps(m.get("objectives"), sort_keys=True),
                m.get("constraints"),
            )
            for m in manifests.values()
            if m.get("status") == "completed"
        }
        report = {
            **status,
            "manifests": manifests,
            "comparable": len(signatures) <= 1 and bool(signatures),
            "aggregate": {
                name: manifest.get("best_configurations")
                for name, manifest in manifests.items()
                if manifest.get("status") == "completed"
            },
        }
        if not request["dry_run"] and all(
            job["state"] in TERMINAL for job in status["jobs"]
        ):
            report["report_uri"] = (
                f"s3://{request['bucket']}/{request['prefix'].strip('/')}/"
                f"{run_id}/aggregate.json"
            )
        return report

    def cancel(self, run_id: str, framework: str | None = None) -> dict[str, Any]:
        status = self.status(run_id)
        if framework and framework not in {job["framework"] for job in status["jobs"]}:
            raise KeyError(framework)
        with self._db() as db:
            if framework is None:
                db.execute("UPDATE runs SET cancelled=1 WHERE id=?", (run_id,))
        for job in status["jobs"]:
            if (
                framework is not None
                and job["framework"] != framework
                or job["state"] in TERMINAL
            ):
                continue
            if job["nebius_id"]:
                self.nebius.cancel(job["nebius_id"])
            state = (
                "cancelling"
                if job["state"] in {"submitting", "submission_unknown"}
                and not job["nebius_id"]
                else "cancelled"
            )
            self._update(run_id, job["framework"], state)
        return self.status(run_id)
