"""Opt-in v4 worker using profiles supplied by the deployment operator."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

from .adapters import S3Artifacts

FRAMEWORKS = {"vllm", "sglang", "trtllm"}


def validate_selection(request):
    if not isinstance(request.campaign_v4, str) or not re.fullmatch(
        r"[a-zA-Z0-9_-]{1,80}", request.campaign_v4
    ):
        raise ValueError("campaign_v4 must select a deployment profile by name")
    if not isinstance(request.revision, str) or not re.fullmatch(
        r"[a-f0-9]{40}", request.revision
    ):
        raise ValueError("v4 campaign requests require a pinned model revision")
    if not set(request.frameworks) <= FRAMEWORKS:
        raise ValueError("v4 campaigns support vLLM, SGLang and TensorRT-LLM")
    if (
        request.client_args
        or request.server_args
        or request.server_config_files
        or request.constraints
        or request.optuna
        or request.optuna_by_framework
        or request.workload_manifest
        or request.speculative_heads
    ):
        raise ValueError("Configure workload, tuning and drafts in the v4 profile")


def run_campaign_worker(request, run_id, framework, store=None, work_dir=None):
    from llm_optimizer.experiment.v4.headless import run_campaign

    if framework not in request.frameworks or not re.fullmatch(
        r"[a-zA-Z0-9_-]{1,100}", run_id
    ):
        raise ValueError("Invalid v4 run identity")
    store = store or S3Artifacts(request.bucket, request.prefix)
    folder = Path(work_dir or tempfile.mkdtemp(prefix="optimizer-v4-")).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "framework": framework,
        "model": request.model,
        "revision": request.revision,
        "image": request.images[framework],
        "campaign_v4": request.campaign_v4,
        "status": "failed",
        "artifacts": {},
    }
    try:
        profile_root = Path(os.environ["LLM_OPTIMIZER_CAMPAIGN_PROFILES"])
        profile_bytes = (profile_root / f"{request.campaign_v4}.json").read_bytes()
        profile = json.loads(profile_bytes)
        manifest["campaign_profile_sha256"] = hashlib.sha256(profile_bytes).hexdigest()
        runner, studies = run_campaign(
            profile[framework],
            framework=framework,
            model=request.model,
            revision=request.revision,
            gpu_count=request.gpu_count,
            run_root=folder / "campaign",
        )
        archive = Path(
            shutil.make_archive(
                str(folder / "campaign-v4"), "zip", root_dir=runner.run_dir
            )
        )
        manifest["artifacts"][archive.name] = store.upload(archive, run_id, framework)
        manifest["studies"] = sorted(studies)
        manifest["status"] = "completed"
    except Exception as error:
        manifest["error"] = f"{type(error).__name__}: {error}"
    path = folder / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_uri"] = store.upload(path, run_id, framework)
    return manifest
