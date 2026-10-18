import os
import shlex
import socket
import subprocess
import threading
import time
import typing as t
from urllib.parse import urlparse

import psutil
import requests

from llm_optimizer.exceptions import ServerCleanupError, ServerNotReadyError
from llm_optimizer.logging import get_logger
from llm_optimizer.predefined.trtllm import required_gpu_count

logger = get_logger("server_utils")


class _TRTLifecycle:
    """Retain worker handles before MPI can detach or reparent them."""

    def __init__(self, process, ready_url, isolated=False):
        self.process = process
        address = urlparse(ready_url)
        self.port = address.port
        self.host = address.hostname
        self.discovery_errors = []
        self.group_id = process.pid if isolated else None
        self.owned = {process.pid: process}
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.track, daemon=True)
        self.thread.start()

    def discover(self):
        with self.lock:
            for process in list(self.owned.values()):
                try:
                    for child in process.children(recursive=True):
                        self.owned.setdefault(child.pid, child)
                except psutil.NoSuchProcess:
                    pass
                except (psutil.Error, OSError) as exc:
                    self.discovery_errors.append(f"PID {process.pid}: {exc}")
            # MPI helpers may be reparented but remain in our isolated group.
            if self.group_id is not None:
                for candidate in psutil.process_iter():
                    try:
                        if os.getpgid(candidate.pid) == self.group_id:
                            self.owned.setdefault(candidate.pid, candidate)
                    except (ProcessLookupError, PermissionError):
                        pass
            return list(self.owned.values())

    def track(self):
        while not self.stop.is_set():
            try:
                self.discover()
            except psutil.Error:
                logger.exception("Unable to discover TRT-LLM worker processes")
            self.stop.wait(0.05)

    def owns_listener(self):
        for process in self.discover():
            try:
                if any(
                    connection.status == psutil.CONN_LISTEN
                    and connection.laddr.port == self.port
                    for connection in process.net_connections(kind="inet")
                ):
                    return True
            except psutil.NoSuchProcess:
                pass
        return False

    def cleanup(self, timeout):
        errors = []

        def signal(process, force=False):
            try:
                if force:
                    process.kill()
                else:
                    process.terminate()
            except psutil.NoSuchProcess:
                pass
            except (psutil.Error, OSError) as exc:
                errors.append(f"PID {process.pid}: {exc}")

        try:
            self.discover()
            # Let the serving parent coordinate MPI/executor shutdown first.
            signal(self.process)
            psutil.wait_procs([self.process], timeout=timeout)
            processes = self.discover()
            _, alive = psutil.wait_procs(processes, timeout=0)
            for process in alive:
                signal(process)
            _, alive = psutil.wait_procs(alive, timeout=timeout)
            # Include workers discovered during graceful shutdown.
            _, alive = psutil.wait_procs(self.discover(), timeout=0)
            for process in alive:
                signal(process, force=True)
            _, alive = psutil.wait_procs(alive, timeout=timeout)
            survivors = []
            for process in alive:
                try:
                    if process.status() != psutil.STATUS_ZOMBIE:
                        survivors.append(process.pid)
                except psutil.NoSuchProcess:
                    pass
            if survivors:
                raise ServerCleanupError(
                    f"TRT-LLM cleanup failed; surviving PIDs: {survivors}; {errors}"
                )
            # A surviving untracked listener must also prevent another trial.
            _assert_port_available(self.port, self.host)
            if self.discovery_errors:
                raise ServerCleanupError(
                    f"Cannot verify all TRT-LLM workers exited: {self.discovery_errors}"
                )
            if errors:
                logger.warning(
                    "TRT-LLM cleanup recovered from signal errors: %s", errors
                )
        except ServerCleanupError:
            raise
        except Exception as exc:
            # Attempt every retained handle even if discovery or waiting fails.
            with self.lock:
                retained = list(self.owned.values())
            for process in retained:
                signal(process, force=True)
            raise ServerCleanupError(
                f"Cannot verify TRT-LLM cleanup completed: {exc}; {errors}"
            ) from exc
        finally:
            self.stop.set()
            self.thread.join(timeout=1)


def _assert_port_available(port, host="127.0.0.1"):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError as exc:
            raise ServerCleanupError(
                f"TRT-LLM serving port {host}:{port} is occupied; refusing next trial"
            ) from exc


def wait_for_server(
    url: str,
    max_retries: int = 60,
    delay: int = 5,
    request_timeout: int = 5,
    process: t.Optional[psutil.Popen] = None,
) -> bool:
    """Repeatedly try to access the server until it's available."""
    logger.info(f"Waiting for server to be ready at {url}...")
    for attempt in range(max_retries):
        if process is not None and process.poll() is not None:
            logger.error(
                "Server process exited before readiness "
                f"(exit code={process.returncode})."
            )
            return False
        try:
            response = requests.get(url, timeout=request_timeout)
            if response.status_code == 200:
                lifecycle = getattr(process, "_trt_lifecycle", None)
                if (
                    isinstance(lifecycle, _TRTLifecycle)
                    and not lifecycle.owns_listener()
                ):
                    raise ServerCleanupError(
                        "Health response belongs to an unowned server"
                    )
                logger.info(f"Server is up after {attempt + 1} attempts.")
                return True
        except requests.RequestException:
            pass

        logger.debug(
            f"Server not ready yet (attempt {attempt + 1}/{max_retries}). "
            f"Retrying in {delay}s."
        )
        time.sleep(delay)
    return False


def start_server(
    server_cmd: str, server_envs: dict[str, str], ready_url: str, mute: bool
) -> psutil.Process:
    """Starts the server and waits for it to become ready."""
    cmd = shlex.split(server_cmd)
    logger.info(f"Starting server with command: {server_cmd}")
    server_process = None
    envs = os.environ.copy()
    envs.update(server_envs)
    is_trt = os.path.basename(cmd[0]) == "trtllm-serve"
    if is_trt:
        address = urlparse(ready_url)
        _assert_port_available(address.port, address.hostname)

    try:
        stdout_pipe = subprocess.DEVNULL if mute else None
        server_process = psutil.Popen(
            cmd,
            text=True,
            stdout=stdout_pipe,
            stderr=subprocess.STDOUT,
            env=envs,
            **({"start_new_session": True} if is_trt else {}),
        )
        if is_trt:
            server_process._trt_lifecycle = _TRTLifecycle(
                server_process, ready_url, isolated=True
            )
        logger.debug(f"Started server with PID: {server_process.pid}")

        if not wait_for_server(ready_url, process=server_process):
            raise ServerNotReadyError(
                "Server did not become ready in the allotted time."
            )

        return server_process
    except BaseException:
        if server_process:
            terminate_process_top_down(server_process)
        raise


def terminate_process_top_down(process: psutil.Process, timeout: int = 10):
    """Terminate a process tree gracefully, escalating to KILL when necessary."""
    logger.info(f"Terminating server process {process.pid} and its children.")
    lifecycle = getattr(process, "_trt_lifecycle", None)
    if isinstance(lifecycle, _TRTLifecycle):
        lifecycle.cleanup(timeout)
        return
    try:
        children = process.children(recursive=True)
        # Terminate children first
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass

        # Terminate parent
        process.terminate()

        # Wait for all to die
        _, alive = psutil.wait_procs([process] + children, timeout=timeout)

        # Force kill any remaining processes
        if alive:
            logger.warning(
                "Some processes did not terminate gracefully. Escalating to SIGKILL."
            )
            for p in alive:
                try:
                    p.kill()
                except psutil.NoSuchProcess:
                    pass
            psutil.wait_procs(alive, timeout=timeout)
    except psutil.NoSuchProcess:
        logger.debug(f"Process {process.pid} was already gone.")
    except Exception as e:
        logger.error(f"Error during process termination: {e}")


def get_server_env(
    framework: t.Optional[str],
    gpu_count: int,
    server_args: dict[str, t.Any],
) -> dict[str, str]:
    """Select GPUs for a managed server and validate TRT parallelism."""
    if gpu_count < 1:
        raise ValueError("--gpus must be at least 1")

    if framework == "trtllm":
        required = required_gpu_count(server_args)
        if required > gpu_count:
            raise ValueError(
                "TensorRT-LLM parallelism requires "
                f"{required} GPUs (TP x PP x CP), but --gpus={gpu_count}."
            )
        gpus_per_node = server_args.get("gpus_per_node")
        if gpus_per_node is not None and int(gpus_per_node) > gpu_count:
            raise ValueError(f"gpus_per_node={gpus_per_node}, but --gpus={gpu_count}.")

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    available = (
        [device.strip() for device in visible_devices.split(",") if device.strip()]
        if visible_devices
        else [str(index) for index in range(gpu_count)]
    )
    if len(available) < gpu_count:
        raise ValueError(
            f"--gpus={gpu_count}, but CUDA_VISIBLE_DEVICES exposes only "
            f"{len(available)} device(s)."
        )
    return {"CUDA_VISIBLE_DEVICES": ",".join(available[:gpu_count])}
