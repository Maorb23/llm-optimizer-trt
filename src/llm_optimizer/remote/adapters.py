"""Nebius CLI and S3 adapters, adapted from hf_validator's job and object store code."""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from .contracts import OptimizationRequest


class NebiusCommandError(RuntimeError):
    """A Nebius CLI failure whose message is safe to log and persist."""


class NebiusAdapter:
    def __init__(
        self, project_id: str | None = None, run_command: Any = subprocess.run
    ):
        self.project_id = project_id
        self.run_command = run_command

    @staticmethod
    def _job_id(value: Any) -> str | None:
        if not isinstance(value, dict):
            return None
        metadata = value.get("metadata")
        job_id = (
            metadata.get("id") if isinstance(metadata, dict) else None
        ) or value.get("id")
        return (
            job_id
            if isinstance(job_id, str) and re.fullmatch(r"aijob-[a-z0-9]+", job_id)
            else None
        )

    def submit(self, payload: dict[str, Any]) -> str:
        request = OptimizationRequest.parse(payload["request"]).to_dict()
        framework = payload["framework"]
        if request["dry_run"]:
            return f"dry-run-{payload['run_id']}-{framework}"
        duration = divmod(request["timeout_minutes"], 60)
        debug_ssh_key = os.getenv("NEBIUS_DEBUG_SSH_PUBLIC_KEY")
        if debug_ssh_key:
            debug_ssh_key = str(Path(debug_ssh_key).expanduser())
            if not Path(debug_ssh_key).is_file():
                raise ValueError(
                    "NEBIUS_DEBUG_SSH_PUBLIC_KEY must name an existing file"
                )
        container_command = "bash" if debug_ssh_key else "python3"
        container_args = (
            "-lc 'python3 -m llm_optimizer.remote.worker || "
            "(status=$?; echo Worker exited with status $status; sleep 3600; exit $status)'"
            if debug_ssh_key
            else "-m llm_optimizer.remote.worker"
        )
        command = [
            "nebius",
            "ai",
            "job",
            "create",
            "--platform",
            request["gpu_type"],
            "--preset",
            request["preset"],
            "--name",
            self.name(payload["run_id"], framework),
            "--image",
            request["images"][framework],
            "--container-command",
            container_command,
            "--args",
            container_args,
            "--disk-size",
            "450Gi",
            "--timeout",
            f"{duration[0]}h{duration[1]}m0s",
            "--format",
            "json",
            "--no-progress",
        ]
        if self.project_id:
            command += ["--parent-id", self.project_id]
        if request["preemptible"]:
            command.append("--preemptible")
        if debug_ssh_key:
            command += ["--ssh-key", debug_ssh_key]
        # The scheduler option is not needed by workers. Existing images were
        # built before it existed, so keep their request payload compatible.
        worker_payload = {
            **payload,
            "request": {
                key: value for key, value in request.items() if key != "preemptible"
            },
        }
        encoded = base64.b64encode(json.dumps(worker_payload).encode()).decode()
        command += ["--env", f"LLM_OPTIMIZER_PAYLOAD_B64={encoded}"]
        for key in (
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "AWS_DEFAULT_REGION",
            "AWS_ENDPOINT_URL_S3",
            "HF_TOKEN",
            "WANDB_API_KEY",
            "WANDB_PROJECT",
            "WANDB_ENTITY",
            "WANDB_RUN_NAME",
            "WANDB_TAGS",
            "WANDB_MODE",
        ):
            if os.getenv(key):
                command += ["--env", f"{key}={os.environ[key]}"]
        try:
            result = self.run_command(
                command, check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError as exc:
            # CalledProcessError.__str__ includes the full command, including
            # credentials supplied via --env. Never expose or persist it.
            detail = (exc.stderr or exc.stdout or "").strip()
            secret_values = [
                os.getenv(key)
                for key in (
                    "AWS_ACCESS_KEY_ID",
                    "AWS_SECRET_ACCESS_KEY",
                    "AWS_SESSION_TOKEN",
                    "HF_TOKEN",
                    "WANDB_API_KEY",
                )
            ]
            if any(
                marker in detail
                for marker in (
                    "--env",
                    "AWS_ACCESS_KEY_ID",
                    "AWS_SECRET_ACCESS_KEY",
                    "AWS_SESSION_TOKEN",
                    "HF_TOKEN",
                    "WANDB_API_KEY",
                    "WANDB_PROJECT",
                    "WANDB_ENTITY",
                    "WANDB_RUN_NAME",
                    "WANDB_TAGS",
                    "WANDB_MODE",
                    "LLM_OPTIMIZER_PAYLOAD_B64",
                )
            ) or any(value and value in detail for value in secret_values):
                detail = "CLI output omitted because it contained environment values"
            else:
                detail = detail[:1200] or "no error details returned"
            raise NebiusCommandError(
                f"Nebius job creation failed (exit {exc.returncode}): {detail}"
            ) from None
        try:
            job_id = self._job_id(json.loads(result.stdout))
        except (ValueError, TypeError):
            job_id = None
        if job_id is None:
            raise RuntimeError(
                "Nebius did not return a valid aijob ID; reconcile before retrying"
            )
        return job_id

    @staticmethod
    def name(run_id: str, framework: str) -> str:
        return f"llm-opt-{run_id[:12]}-{framework}"

    def find_by_name(self, run_id: str, framework: str) -> str | None:
        if not self.project_id:
            return None
        command = [
            "nebius",
            "ai",
            "job",
            "get-by-name",
            "--name",
            self.name(run_id, framework),
            "--parent-id",
            self.project_id,
            "--format",
            "json",
        ]
        try:
            result = self.run_command(
                command, check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError:
            return None
        return self._job_id(json.loads(result.stdout))

    def status(self, job_id: str) -> str:
        if job_id.startswith("dry-run-"):
            return "dry_run"
        result = self.run_command(
            ["nebius", "ai", "job", "get", job_id, "--format", "json"],
            check=True,
            capture_output=True,
            text=True,
        )
        value = json.loads(result.stdout)
        state = value.get("state") or value.get("status") or "unknown"
        if isinstance(state, dict):
            state = state.get("state") or state.get("status") or "unknown"
        return str(state).lower()

    def cancel(self, job_id: str) -> None:
        if not job_id.startswith("dry-run-"):
            self.run_command(
                ["nebius", "ai", "job", "cancel", job_id],
                check=True,
                capture_output=True,
                text=True,
            )


class S3Artifacts:
    def __init__(
        self,
        bucket: str,
        prefix: str,
        endpoint_url: str | None = None,
        client: Any = None,
    ):
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        if client is None:
            import boto3

            endpoint_url = (
                endpoint_url
                or os.getenv("AWS_ENDPOINT_URL_S3")
                or os.getenv("AWS_ENDPOINT_URL")
            )
            client = boto3.client(
                "s3", **({"endpoint_url": endpoint_url} if endpoint_url else {})
            )
        self.client = client

    def key(self, run_id: str, framework: str, name: str) -> str:
        return "/".join(x for x in (self.prefix, run_id, framework, name) if x)

    def upload(
        self, path: Path, run_id: str, framework: str, name: str | None = None
    ) -> str:
        key = self.key(run_id, framework, name or path.name)
        self.client.upload_file(str(path), self.bucket, key)
        return f"s3://{self.bucket}/{key}"

    def download(self, path: Path, run_id: str, framework: str, name: str) -> bool:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.client.download_file(
                self.bucket, self.key(run_id, framework, name), str(path)
            )
            return True
        except Exception as exc:
            if _missing_key(exc):
                return False
            raise

    def manifest(self, run_id: str, framework: str) -> dict[str, Any] | None:
        try:
            response = self.client.get_object(
                Bucket=self.bucket, Key=self.key(run_id, framework, "manifest.json")
            )
            return json.loads(response["Body"].read())
        except Exception as exc:
            if _missing_key(exc):
                return None
            raise

    def execution_log_tail(self, run_id: str, framework: str) -> str | None:
        try:
            response = self.client.get_object(
                Bucket=self.bucket,
                Key=self.key(run_id, framework, "execution.log"),
                Range="bytes=-16384",
            )
            return response["Body"].read().decode("utf-8", "replace")
        except Exception as exc:
            if _missing_key(exc):
                return None
            raise

    def progress(self, run_id: str, framework: str) -> dict[str, Any] | None:
        try:
            response = self.client.get_object(
                Bucket=self.bucket, Key=self.key(run_id, framework, "progress.json")
            )
            return json.loads(response["Body"].read())
        except Exception as exc:
            if _missing_key(exc):
                return None
            raise

    def put_report(self, run_id: str, report: dict[str, Any]) -> str:
        key = "/".join(x for x in (self.prefix, run_id, "aggregate.json") if x)
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json.dumps(report, indent=2, default=str).encode(),
            ContentType="application/json",
        )
        return f"s3://{self.bucket}/{key}"


def _missing_key(exc: Exception) -> bool:
    response = getattr(exc, "response", {})
    code = response.get("Error", {}).get("Code") if isinstance(response, dict) else None
    return code in {"404", "NoSuchKey", "NotFound"}
