"""Single-framework GPU worker; invoked by the Nebius container image."""

from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .adapters import S3Artifacts
from .contracts import OptimizationRequest


def optimizer_command(payload: dict[str, Any], output: Path) -> list[str]:
    request = OptimizationRequest.parse(payload["request"])
    framework = payload["framework"]
    optuna = request.optuna_for(framework)
    if framework not in request.frameworks:
        raise ValueError("framework is not selected in request")
    args = [
        sys.executable,
        "-m",
        "llm_optimizer.cli",
        "--model",
        request.model,
        "--framework",
        framework,
        "--gpus",
        str(request.gpu_count),
        "--output-json",
        str(output),
        "--search-strategy",
        request.search_strategy,
    ]
    if request.constraints:
        args += ["--constraints", request.constraints]
    for value in request.server_args.get(framework, []):
        args += ["--server-args", value]
    if request.revision:
        revision_arg = (
            "huggingface_model_revision" if framework == "max" else "revision"
        )
        args += ["--server-args", f"{revision_arg}={request.revision}"]
    for value in request.client_args:
        args += ["--client-args", value]
    if request.search_strategy == "optuna":
        mapping = {
            "trials": "--optuna-trials",
            "timeout": "--optuna-timeout",
            "objective": "--optuna-objective",
            "seed": "--optuna-seed",
            "sampler": "--optuna-sampler",
            "pruner": "--optuna-pruner",
            "storage": "--optuna-storage",
            "study_name": "--optuna-study-name",
        }
        if "objectives" in optuna:
            args += ["--optuna-objective", ",".join(optuna["objectives"])]
        for key, flag in mapping.items():
            if key in optuna:
                value = optuna[key]
                if key == "study_name":
                    value = f"{value}-{payload.get('run_id', 'run')}-{framework}"
                args += [flag, str(value)]
        for key in ("sampler_kwargs", "pruner_kwargs"):
            if key in optuna:
                args += [
                    "--optuna-" + key.replace("_", "-"),
                    json.dumps(optuna[key]),
                ]
        for key in ("server_args", "client_args"):
            for value in optuna.get(key, []):
                args += ["--optuna-" + key.replace("_", "-"), value]
    return args


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _public_storage(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.password is None and not parsed.query:
        return url
    host = parsed.hostname or ""
    if parsed.port:
        host += f":{parsed.port}"
    if parsed.username:
        host = f"{parsed.username}:***@{host}"
    return urlunsplit((parsed.scheme, host, parsed.path, "", ""))


def _checkpoint(
    store: S3Artifacts, folder: Path, run_id: str, framework: str
) -> dict[str, str]:
    artifacts = {}
    records = folder / "results.jsonl"
    completed = (
        sum(1 for line in records.read_text(encoding="utf-8").splitlines() if line)
        if records.exists()
        else 0
    )
    progress = folder / "progress.json"
    progress.write_text(json.dumps({"completed_trials": completed}), encoding="utf-8")
    artifacts["progress.json"] = store.upload(progress, run_id, framework)
    database = folder / "results.optuna.db"
    if database.exists():
        snapshot = folder / "study.checkpoint.db"
        with (
            sqlite3.connect(str(database)) as source,
            sqlite3.connect(str(snapshot)) as target,
        ):
            source.backup(target)
        bundle = folder / "checkpoint.zip"
        with zipfile.ZipFile(bundle, "w") as archive:
            archive.write(snapshot, "results.optuna.db")
            if records.exists():
                archive.write(records, records.name)
        artifacts["checkpoint.zip"] = store.upload(bundle, run_id, framework)
    for name in (
        "results.jsonl",
        "results.json",
        "results.workload.json",
        "deployment.json",
        "execution.log",
    ):
        path = folder / name
        if path.exists():
            artifacts[name] = store.upload(path, run_id, framework)
    return artifacts


def run_worker(
    payload: dict[str, Any],
    *,
    store: S3Artifacts | None = None,
    work_dir: Path | None = None,
    runner: Any = subprocess.Popen,
) -> dict[str, Any]:
    request = OptimizationRequest.parse(payload["request"])
    run_id, framework = payload["run_id"], payload["framework"]
    if request.dry_run:
        raise ValueError("dry-run requests do not launch GPU workers")
    store = store or S3Artifacts(
        request.bucket, request.prefix, os.getenv("AWS_ENDPOINT_URL_S3")
    )
    folder = work_dir or Path(
        tempfile.mkdtemp(prefix=f"optimizer-{run_id}-{framework}-")
    )
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / "results.json"
    command = optimizer_command(payload, output)
    worker_env = os.environ.copy()
    worker_env["LLM_OPTIMIZER_RUN_ID"] = run_id
    public_command = command.copy()
    optuna = request.optuna_for(framework)
    public_optuna = optuna.copy()
    if "storage" in public_optuna:
        public_optuna["storage"] = _public_storage(str(public_optuna["storage"]))
        public_command[public_command.index("--optuna-storage") + 1] = public_optuna[
            "storage"
        ]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "framework": framework,
        "model": request.model,
        "revision": request.revision,
        "speculative_heads": request.speculative_heads,
        "gpu_type": request.gpu_type,
        "gpu_count": request.gpu_count,
        "preset": request.preset,
        "image": request.images[framework],
        "workload": request.client_args,
        "semantic_context": request.semantic_context,
        "objectives": optuna.get(
            "objectives", optuna.get("objective", "output_throughput")
        )
        if request.search_strategy == "optuna"
        else "throughput",
        "constraints": request.constraints,
        "optimization": {
            "strategy": request.search_strategy,
            "optuna": public_optuna,
            "server_args": request.server_args.get(framework, []),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "llm_optimizer": _version("llm-optimizer"),
            framework: _version({"trtllm": "tensorrt-llm"}.get(framework, framework)),
            "torch": _version("torch"),
            "transformers": _version("transformers"),
            "optuna": _version("optuna"),
            "huggingface_hub": _version("huggingface-hub"),
        },
        "status": "failed",
        "artifacts": {},
        "command": public_command,
    }
    log = folder / "execution.log"
    process = None
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        if len(gpu.stdout.strip().splitlines()) < request.gpu_count:
            raise RuntimeError("fewer GPUs visible than requested")
        manifest["runtime"]["visible_gpus"] = gpu.stdout.strip().splitlines()
        if manifest["runtime"][framework] is None:
            raise RuntimeError(f"{framework} package is absent from the container")
        from huggingface_hub import HfApi

        model_info = HfApi().model_info(
            request.model, revision=request.revision, token=os.getenv("HF_TOKEN")
        )
        if not model_info.sha:
            raise RuntimeError("Hugging Face did not return a model commit")
        manifest["resolved_revision"] = model_info.sha
        if not request.revision:
            revision_arg = (
                "huggingface_model_revision" if framework == "max" else "revision"
            )
            command += ["--server-args", f"{revision_arg}={model_info.sha}"]
            public_command += ["--server-args", f"{revision_arg}={model_info.sha}"]
        with log.open("a", encoding="utf-8") as stream:
            preflight = subprocess.run(
                command + ["--dry-run"], stdout=stream, stderr=subprocess.STDOUT
            )
        if preflight.returncode:
            raise RuntimeError("framework argument preflight failed; see execution.log")
        # Restore only this framework's checkpoint. One controller job owns the study.
        if request.search_strategy == "optuna" and not optuna.get("storage"):
            checkpoint = folder / "checkpoint.zip"
            if store.download(checkpoint, run_id, framework, checkpoint.name):
                with zipfile.ZipFile(checkpoint) as archive:
                    for name in ("results.optuna.db", "results.jsonl"):
                        if name in archive.namelist():
                            (folder / name).write_bytes(archive.read(name))
                command.append("--continue")
        with log.open("a", encoding="utf-8") as stream:
            process = runner(
                command, stdout=stream, stderr=subprocess.STDOUT, env=worker_env
            )
            while process.poll() is None:
                time.sleep(30)
                try:
                    manifest["artifacts"].update(
                        _checkpoint(store, folder, run_id, framework)
                    )
                except Exception as exc:
                    stream.write(f"Checkpoint upload failed: {exc}\n")
                    stream.flush()
            if process.returncode != 0:
                raise RuntimeError(f"optimizer exited with code {process.returncode}")
        result = json.loads(output.read_text(encoding="utf-8"))
        manifest["test_results"] = result.get("test_results", [])
        manifest["study_summary"] = result.get("optimization", {}).get(
            "study_summary", {}
        )
        manifest["best_configurations"] = result.get("best_configurations")
        manifest["total_tests"] = result.get("metadata", {}).get("total_tests")
        manifest["workload_sha256"] = result.get("metadata", {}).get("workload_sha256")
        (folder / "deployment.json").write_text(
            json.dumps(
                {
                    "model": request.model,
                    "revision": request.revision,
                    "framework": framework,
                    "image": request.images[framework],
                    "gpu_type": request.gpu_type,
                    "gpu_count": request.gpu_count,
                    "best_configurations": result.get("best_configurations"),
                    "instructions": "Use the selected result's cmd and server_args "
                    "in the framework image with this model revision.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        manifest["status"] = "completed"
    except Exception as exc:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        manifest["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        # Keep successful measurements even when a later trial/process fails.
        if not manifest.get("test_results"):
            records = folder / "results.jsonl"
            rows = []
            if records.exists():
                for line in records.read_text(encoding="utf-8").splitlines():
                    try:
                        row = json.loads(line)
                        if isinstance(row, dict):
                            rows.append(row)
                    except json.JSONDecodeError:
                        continue
            manifest["test_results"] = rows
        if not manifest.get("study_summary") and request.search_strategy == "optuna":
            try:
                from llm_optimizer.optimization import (
                    objective_names,
                    require_optuna,
                    summarize_study,
                )

                library = require_optuna()
                database = folder / "results.optuna.db"
                storage = optuna.get("storage") or f"sqlite:///{database.resolve()}"
                names = manifest["objectives"]
                objectives = (
                    names if isinstance(names, list) else objective_names(names)
                )
                if optuna.get("storage") or database.exists():
                    studies = library.get_all_study_summaries(storage)
                    name = optuna.get("study_name")
                    if name:
                        name = f"{name}-{run_id}-{framework}"
                    elif len(studies) == 1:
                        name = studies[0].study_name
                    if name:
                        study = library.load_study(study_name=name, storage=storage)
                        manifest["study_summary"] = summarize_study(study, objectives)
            except Exception:
                # Analysis cannot replace the original execution failure.
                pass
        try:
            manifest["artifacts"].update(_checkpoint(store, folder, run_id, framework))
            manifest["sha256"] = {
                name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                for name in manifest["artifacts"]
                if (folder / name).is_file()
            }
        except Exception as exc:
            manifest["status"] = "failed"
            manifest["error"] = f"artifact upload failed: {exc}"
        path = folder / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        manifest["manifest_uri"] = store.upload(path, run_id, framework)
        try:
            from llm_optimizer.wandb_tracking import WandbTracker

            study_name = str(optuna.get("study_name", f"llm-optimizer-{framework}"))
            tracker = WandbTracker.start(
                study_name=study_name,
                framework=framework,
                model=request.model,
                gpu=request.gpu_type,
                run_id=run_id,
                objective=str(manifest["objectives"]),
            )
            tracker.artifacts(
                [
                    folder / "results.json",
                    path,
                    folder / "results.jsonl",
                    folder / "execution.log",
                ],
                name=f"{run_id}-{framework}-final",
            )
            tracker.finish()
        except Exception:
            # Telemetry is strictly best effort, including unexpected wrapper bugs.
            logger = __import__("logging").getLogger(__name__)
            logger.warning("W&B final artifact upload failed; continuing")
    return manifest


if __name__ == "__main__":
    payload = json.loads(base64.b64decode(os.environ["LLM_OPTIMIZER_PAYLOAD_B64"]))
    result = run_worker(payload)
    sys.exit(0 if result["status"] == "completed" else 1)
