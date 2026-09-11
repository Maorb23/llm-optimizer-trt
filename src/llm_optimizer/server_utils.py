import os
import shlex
import subprocess
import time
import typing as t

import psutil
import requests

from llm_optimizer.exceptions import ServerNotReadyError
from llm_optimizer.logging import get_logger
from llm_optimizer.predefined.trtllm import required_gpu_count

logger = get_logger("server_utils")


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

    try:
        stdout_pipe = subprocess.DEVNULL if mute else None
        server_process = psutil.Popen(
            cmd, text=True, stdout=stdout_pipe, stderr=subprocess.STDOUT, env=envs
        )
        logger.debug(f"Started server with PID: {server_process.pid}")

        if not wait_for_server(ready_url, process=server_process):
            raise ServerNotReadyError(
                "Server did not become ready in the allotted time."
            )

        return server_process
    except Exception:
        if server_process:
            terminate_process_top_down(server_process)
        raise


def terminate_process_top_down(process: psutil.Process, timeout: int = 10):
    """Terminate a process tree gracefully, escalating to KILL when necessary."""
    logger.info(f"Terminating server process {process.pid} and its children.")
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
            raise ValueError(
                f"gpus_per_node={gpus_per_node}, but --gpus={gpu_count}."
            )

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
