import socket
import sys
import time
from unittest.mock import Mock, patch

import psutil
import pytest

from llm_optimizer.exceptions import ServerCleanupError
from llm_optimizer.server_utils import (
    _assert_port_available,
    _TRTLifecycle,
    start_server,
)


def test_occupied_port_refuses_launch():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        with pytest.raises(ServerCleanupError, match="occupied"):
            _assert_port_available(port)


def test_cleanup_tracks_detached_worker_and_kills_it():
    import subprocess

    script = (
        "import subprocess,sys,time; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'], "
        "start_new_session=True); print(p.pid,flush=True); time.sleep(60)"
    )
    process = psutil.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE)
    lifecycle = _TRTLifecycle(process, "http://127.0.0.1:0/health")
    worker = psutil.Process(int(process.stdout.readline()))
    try:
        deadline = time.monotonic() + 3
        while worker.pid not in lifecycle.owned and time.monotonic() < deadline:
            time.sleep(0.05)
        assert worker.pid in lifecycle.owned
        lifecycle.cleanup(timeout=0.1)
        assert process.poll() is not None
        assert not worker.is_running() or worker.status() == psutil.STATUS_ZOMBIE
    finally:
        lifecycle.stop.set()
        for owned in (process, worker):
            try:
                owned.kill()
            except psutil.NoSuchProcess:
                pass
        process.wait()
        process.stdout.close()


def test_startup_interrupt_cleans_up():
    process = Mock(pid=123)
    with (
        patch("llm_optimizer.server_utils.psutil.Popen", return_value=process),
        patch("llm_optimizer.server_utils._TRTLifecycle.track"),
        patch(
            "llm_optimizer.server_utils.wait_for_server", side_effect=KeyboardInterrupt
        ),
        patch("llm_optimizer.server_utils.terminate_process_top_down") as cleanup,
        pytest.raises(KeyboardInterrupt),
    ):
        start_server("trtllm-serve model", {}, "http://127.0.0.1:0/health", True)
    cleanup.assert_called_once_with(process)


def test_signal_error_does_not_skip_other_workers_and_survivors_stop_run():
    parent = Mock(pid=100)
    child = Mock(pid=101)
    parent.children.return_value = [child]
    child.children.return_value = []
    child.terminate.side_effect = OSError(22, "Invalid argument")
    parent.status.return_value = child.status.return_value = psutil.STATUS_RUNNING
    with (
        patch("llm_optimizer.server_utils._TRTLifecycle.track"),
        patch(
            "llm_optimizer.server_utils.psutil.wait_procs",
            side_effect=lambda processes, timeout: ([], processes),
        ),
    ):
        lifecycle = _TRTLifecycle(parent, "http://127.0.0.1:0/health")
        with pytest.raises(ServerCleanupError, match="surviving PIDs"):
            lifecycle.cleanup(timeout=0)
    parent.kill.assert_called_once()
    child.kill.assert_called_once()


def test_readiness_rejects_unowned_listener():
    from llm_optimizer.server_utils import wait_for_server

    process = Mock(pid=100)
    process.poll.return_value = None
    with patch("llm_optimizer.server_utils._TRTLifecycle.track"):
        lifecycle = _TRTLifecycle(process, "http://127.0.0.1:0/health")
    process._trt_lifecycle = lifecycle
    with (
        patch.object(lifecycle, "owns_listener", return_value=False),
        patch(
            "llm_optimizer.server_utils.requests.get",
            return_value=Mock(status_code=200),
        ),
        pytest.raises(ServerCleanupError, match="unowned server"),
    ):
        wait_for_server("http://127.0.0.1:0/health", process=process)
