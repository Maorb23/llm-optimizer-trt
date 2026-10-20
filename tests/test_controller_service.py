import json
import threading
import time

import pytest
from llm_optimizer.remote.controller import Controller
from llm_optimizer.remote.service import (
    controller_health,
    ensure_running,
    run_forever,
    sidecar,
)


def test_serve_lease_blocks_other_ticks_even_during_provider_wait(tmp_path):
    path = tmp_path / "controller.sqlite"
    owner = Controller(path)
    other = Controller(path)
    with owner.exclusive():
        owner.tick()  # nested lease on the owning controller is allowed
        with pytest.raises(RuntimeError, match="Another controller"):
            other.tick()
    other.tick()


def test_process_lock_uses_shared_inode_through_alias_paths(tmp_path):
    path = tmp_path / "controller.sqlite"
    owner = Controller(path)
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    with owner.exclusive(), pytest.raises(RuntimeError, match="Another controller"):
        Controller(alias / "controller.sqlite").tick()


def test_managed_startup_waits_for_background_service_and_stays_healthy(tmp_path):
    path = tmp_path / "controller.sqlite"
    controller = Controller(path, concurrency=3)
    sidecar(path, ".controller-managed.json").write_text(
        json.dumps({"managed": True, "concurrency": 3, "startup_timeout_seconds": 2})
    )
    stop = threading.Event()

    def host_service():
        deadline = time.monotonic() + 2
        while (
            not sidecar(path, ".controller-start").exists()
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        run_forever(controller, 0.05, stop=stop)

    thread = threading.Thread(target=host_service)
    thread.start()
    try:
        first = ensure_running(path)
        assert first["managed"] and first["healthy"]
        assert ensure_running(path)["pid"] == first["pid"]
        assert controller_health(path)["healthy"]
        with pytest.raises(RuntimeError, match="Another controller"):
            Controller(path).tick()
    finally:
        stop.set()
        thread.join(timeout=3)
    assert not controller_health(path)["healthy"]
    assert not sidecar(path, ".controller-health.json").exists()


def test_stale_health_without_lease_is_not_healthy_and_timeout_is_explicit(tmp_path):
    path = tmp_path / "controller.sqlite"
    Controller(path)
    sidecar(path, ".controller-managed.json").write_text(
        json.dumps({"managed": True, "startup_timeout_seconds": 0.05})
    )
    sidecar(path, ".controller-health.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state": "ready",
                "updated_at": time.time(),
                "concurrency": 3,
            }
        )
    )
    assert not controller_health(path)["healthy"]
    with pytest.raises(RuntimeError, match="nothing was submitted"):
        ensure_running(path)


def test_unmanaged_clients_preserve_existing_behavior(tmp_path):
    assert ensure_running(tmp_path / "controller.sqlite") == {
        "managed": False,
        "healthy": False,
    }


def test_malformed_registration_does_not_silently_allow_unmanaged_submission(tmp_path):
    path = tmp_path / "controller.sqlite"
    sidecar(path, ".controller-managed.json").write_text("{invalid")
    with pytest.raises(ValueError, match="nothing was submitted"):
        ensure_running(path)


def test_second_process_is_rejected_under_serve_lease(tmp_path):
    import subprocess
    import sys

    path = tmp_path / "controller.sqlite"
    owner = Controller(path)
    code = (
        "from llm_optimizer.remote.controller import Controller; Controller("
        + repr(str(path))
        + ").tick()"
    )
    with owner.exclusive():
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=10
        )
    assert result.returncode != 0
    assert "Another controller owns this database" in result.stderr
