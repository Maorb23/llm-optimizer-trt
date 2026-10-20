"""Single-writer controller lease and optional host-managed startup handshake."""

from __future__ import annotations

import fcntl
import json
import logging
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


def sidecar(database, suffix):
    return Path(str(Path(database).resolve()) + suffix)


@contextmanager
def controller_lease(database):
    """The inode is shared even when host/container mount paths differ."""
    with sidecar(database, ".controller.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                "Another controller owns this database; do not run tick alongside serve"
            ) from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _locked(database):
    try:
        with sidecar(database, ".controller.lock").open("r") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(lock, fcntl.LOCK_UN)
            return False
    except FileNotFoundError:
        return False


def _read(path):
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (FileNotFoundError, ValueError):
        return {}


def controller_health(database, max_age=15):
    health = _read(sidecar(database, ".controller-health.json"))
    age = time.time() - health.get("updated_at", 0)
    ready = (
        health.get("schema_version") == 1
        and health.get("state") == "ready"
        and 0 <= age <= max_age
        and _locked(database)
    )
    return {**health, "healthy": ready}


def ensure_running(database):
    """Request a systemd .path activation; never spawn from the agent container."""
    registration_path = sidecar(database, ".controller-managed.json")
    registration = _read(registration_path)
    if registration_path.exists() and (
        registration.get("schema_version", 1) != 1
        or not isinstance(registration.get("managed"), bool)
    ):
        raise ValueError(
            "Invalid managed controller registration; nothing was submitted"
        )
    if not registration.get("managed"):
        return {"managed": False, "healthy": False}
    timeout = float(registration.get("startup_timeout_seconds", 20))
    if not 0 < timeout <= 60:
        raise ValueError("Controller startup timeout must be between 0 and 60 seconds")

    def healthy():
        health = controller_health(database)
        return (
            health
            if health["healthy"]
            and health.get("concurrency") == registration.get("concurrency", 3)
            else None
        )

    health = healthy()
    if health:
        return {**health, "managed": True}
    sidecar(database, ".controller-start").touch(mode=0o600, exist_ok=True)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        health = healthy()
        if health:
            return {**health, "managed": True}
        time.sleep(0.1)
    raise RuntimeError(
        "Managed optimizer controller did not become healthy; nothing was submitted. "
        "Check the hf-model-watch-optimizer user service logs"
    )


def run_forever(controller, interval=15, *, stop=None):
    """Hold the lease for the entire serve lifetime, including provider waits."""
    if interval <= 0:
        raise ValueError("Controller interval must be positive")
    stop = stop or threading.Event()
    health_path = sidecar(controller.database, ".controller-health.json")
    state = {
        "schema_version": 1,
        "pid": os.getpid(),
        "state": "ready",
        "concurrency": controller.concurrency,
        "last_tick_completed_at": None,
    }
    state_lock = threading.Lock()

    def publish():
        with state_lock:
            payload = {**state, "updated_at": time.time()}
        fd, name = tempfile.mkstemp(
            prefix=".controller-health-", dir=health_path.parent
        )
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(payload, stream)
            os.replace(name, health_path)
        finally:
            Path(name).unlink(missing_ok=True)

    def heartbeat():
        while not stop.wait(2):
            publish()

    with controller.exclusive():
        publish()
        sidecar(controller.database, ".controller-start").unlink(missing_ok=True)
        thread = threading.Thread(
            target=heartbeat, name="optimizer-heartbeat", daemon=True
        )
        thread.start()
        try:
            while not stop.is_set():
                try:
                    controller.tick()
                except Exception as exc:
                    # Provider exceptions may contain credentials. Only log type.
                    logger.warning("Controller tick failed (%s)", type(exc).__name__)
                    with state_lock:
                        state["state"] = "degraded"
                        state["error_type"] = type(exc).__name__
                else:
                    with state_lock:
                        state["state"] = "ready"
                        state.pop("error_type", None)
                        state["last_tick_completed_at"] = time.time()
                publish()
                stop.wait(interval)
        finally:
            stop.set()
            thread.join(timeout=3)
            health_path.unlink(missing_ok=True)
