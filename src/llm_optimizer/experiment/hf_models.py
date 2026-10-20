"""Pinned model setup shared by notebooks and Python experiments."""


def configure_hf_models(
    context,
    target_model_id,
    target_revision=None,
    *,
    framework=None,
    enable_dflash=False,
    speculative_algorithm=None,
    max_draft_len=3,
    draft_model_id=None,
    draft_revision=None,
    block_size=16,
    num_speculative_tokens=15,
    cache_dir=None,
):
    """Configure Phase B atomically, retaining resolved revisions for safe resume."""
    import copy
    import importlib.metadata
    import json
    import re
    import subprocess
    import sys
    from pathlib import Path

    import torch
    from huggingface_hub import HfApi, snapshot_download

    RUN_REAL_GPU = context["RUN_REAL_GPU"]
    REAL_FRAMEWORKS = context["REAL_FRAMEWORKS"]
    REAL_PREFLIGHT = context["REAL_PREFLIGHT"]
    MODEL_SNAPSHOTS = context["MODEL_SNAPSHOTS"]
    PHASE_MODELS = context["PHASE_MODELS"]
    RUN_DIR = context["RUN_DIR"]
    expanded_choices = context["expanded_choices"]
    display = context.get("display", lambda value: None)
    if not RUN_REAL_GPU:
        raise ValueError("Set RUN_REAL_GPU=True for model setup")
    if framework is None:
        if len(REAL_FRAMEWORKS) != 1:
            raise ValueError(
                "Set HF_FRAMEWORK explicitly when selecting multiple frameworks"
            )
        framework = REAL_FRAMEWORKS[0]
    if framework not in REAL_FRAMEWORKS or framework not in {
        "vllm",
        "sglang",
        "trtllm",
    }:
        raise ValueError(f"Unsupported or unselected framework: {framework}")
    if ("B", framework) not in REAL_PREFLIGHT:
        raise ValueError(f"Define the Phase B {framework} preflight before model setup")
    if not torch.cuda.is_available():
        raise RuntimeError("The selected notebook kernel has no CUDA GPU")
    if not isinstance(target_model_id, str) or not target_model_id.strip():
        raise ValueError("Provide a target Hugging Face model ID")
    draft_enabled = enable_dflash or speculative_algorithm is not None
    if enable_dflash and framework == "trtllm":
        raise ValueError(
            "Select an explicit TRT speculative_algorithm; DFlash/DSpark are distinct"
        )
    if framework == "trtllm" and draft_enabled:
        from llm_optimizer.predefined.trtllm import validate_draft_request

        validate_draft_request(speculative_algorithm, max_draft_len, draft_model_id)
    elif speculative_algorithm is not None:
        raise ValueError("speculative_algorithm is currently a TRT-LLM option")
    if enable_dflash and (
        not draft_model_id
        or isinstance(block_size, bool)
        or not isinstance(block_size, int)
        or block_size < 1
    ):
        raise ValueError("DFlash requires a draft model ID and a positive block size")

    HF_BASE_PROFILES = context.setdefault("HF_BASE_PROFILES", {})
    profile_key = ("B", framework)
    if profile_key not in HF_BASE_PROFILES:
        HF_BASE_PROFILES[profile_key] = copy.deepcopy(REAL_PREFLIGHT[profile_key])
    profile = copy.deepcopy(HF_BASE_PROFILES[profile_key])
    fixed_server = {
        key: value
        for key, value in profile.get("fixed_server", {}).items()
        if not key.startswith("speculative_")
    }
    for key in ("draft_revision", "draft_model_id", "speculative_decoding"):
        profile.pop(key, None)
    if framework == "trtllm" and draft_enabled:
        if fixed_server.get("backend", "pytorch") != "pytorch":
            raise ValueError("TRT draft setup requires the PyTorch backend")

    if enable_dflash:
        if framework == "vllm" and (
            isinstance(num_speculative_tokens, bool)
            or not isinstance(num_speculative_tokens, int)
            or num_speculative_tokens < 1
        ):
            raise ValueError("vLLM num_speculative_tokens must be a positive integer")
        module = (
            "sglang.launch_server"
            if framework == "sglang"
            else "vllm.entrypoints.openai.api_server"
        )
        help_result = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            capture_output=True,
            text=True,
            timeout=90,
        )
        help_text = help_result.stdout + help_result.stderr
        required = (
            [
                "DFLASH",
                "--speculative-algorithm",
                "--speculative-draft-model-path",
                "--speculative-num-draft-tokens",
            ]
            if framework == "sglang"
            else ["--speculative-config"]
        )
        if help_result.returncode or any(flag not in help_text for flag in required):
            raise RuntimeError(
                f"This notebook kernel's {framework} does not expose the required "
                "speculative decoding flags. Runtime output:\n" + help_text[-2000:]
            )
        if framework == "sglang":
            choices = profile["expanded_choices"]
            for key, value in (
                ("enable_mixed_chunk", False),
                ("disable_overlap_schedule", True),
            ):
                if value not in choices[key]:
                    raise ValueError(
                        f"DFlash requires {key}={value} in your preflight choices"
                    )
                choices[key] = [value]

    def download_pinned(model_id, revision):
        revision = revision or "main"
        if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            revision = HfApi().model_info(model_id, revision=revision).sha
        if not isinstance(revision, str) or not re.fullmatch(
            r"[0-9a-fA-F]{40}", revision
        ):
            raise RuntimeError(
                f"Could not resolve an immutable revision for {model_id}"
            )
        snapshot = Path(
            snapshot_download(
                repo_id=model_id,
                revision=revision,
                cache_dir=cache_dir,
            )
        )
        if not (snapshot / "config.json").is_file():
            raise RuntimeError(f"Downloaded snapshot has no config.json: {snapshot}")
        return str(snapshot), revision

    target_path, resolved_target = download_pinned(target_model_id, target_revision)
    draft_path, resolved_draft = None, None
    if draft_enabled:
        draft_path, resolved_draft = download_pinned(draft_model_id, draft_revision)
        if framework == "trtllm":
            from llm_optimizer.predefined.trtllm import draft_config

            fixed_server["speculative_config"] = draft_config(
                speculative_algorithm, max_draft_len, target_path, draft_path
            )
        elif framework == "sglang":
            fixed_server.update(
                speculative_algorithm="DFLASH",
                speculative_draft_model_path=draft_path,
                speculative_num_draft_tokens=block_size,
            )
        else:
            fixed_server["speculative_config"] = {
                "method": "dspark",
                "model": draft_path,
                "num_speculative_tokens": num_speculative_tokens,
            }
        profile.update(
            draft_model_id=draft_model_id,
            draft_revision=resolved_draft,
            speculative_decoding={
                "algorithm": speculative_algorithm
                if framework == "trtllm"
                else "DFLASH",
                **(
                    {"native_config": fixed_server["speculative_config"]}
                    if framework == "trtllm"
                    else {}
                ),
                "draft_model_id": draft_model_id,
                "draft_revision": resolved_draft,
                **(
                    {"max_draft_len": max_draft_len}
                    if framework == "trtllm"
                    else {"block_size": block_size}
                    if framework == "sglang"
                    else {"num_speculative_tokens": num_speculative_tokens}
                ),
            },
        )

    # Preserve the user's native/quality verification declarations.
    profile.update(
        gpu_type=torch.cuda.get_device_name(0),
        framework_version=importlib.metadata.version(
            "tensorrt_llm" if framework == "trtllm" else framework
        ),
        cuda_version=torch.version.cuda,
        driver_version=subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            timeout=10,
        )
        .strip()
        .splitlines()[0],
        model_revision=resolved_target,
        hf_target_model_id=target_model_id,
        fixed_server=fixed_server,
    )
    native_config_path = None
    expanded_choices(framework, profile["expanded_choices"])
    if framework == "trtllm" and draft_enabled:
        # Preview native YAML now; retain structured settings for per-trial merging.
        from llm_optimizer.predefined.trtllm import serialize_config

        preview = serialize_config(fixed_server, RUN_DIR)
        native_config_path = preview.get("config")
    # Commit the new configuration only after all checks/downloads succeeded.
    summary = {
        "framework": framework,
        "target_model_id": target_model_id,
        "target_revision": resolved_target,
        "target_path": target_path,
        "draft_model_id": draft_model_id if draft_enabled else None,
        "draft_revision": resolved_draft,
        "draft_path": draft_path,
        "speculative_decoding": profile.get("speculative_decoding"),
    }
    if native_config_path:
        summary["native_config_path"] = native_config_path
    (RUN_DIR / f"hf-{framework}-models.json").write_text(json.dumps(summary, indent=2))
    MODEL_SNAPSHOTS["B"] = target_path
    PHASE_MODELS["B"] = target_model_id
    REAL_PREFLIGHT[profile_key] = profile
    display(summary)
    print(
        f"Model setup complete for {framework}. "
        "Run the existing Phase B objective cells."
    )
    return summary


def fingerprint_payload(model, framework, concurrency, config, profile):
    """Retain old identities for non-speculative runs; bind drafts to revisions."""
    import json

    payload = [model, framework, concurrency, config]
    if profile.get("speculative_decoding"):
        from llm_optimizer.experiment.gp_policy import canonical_config

        payload.append(
            {
                key: profile.get(key)
                for key in ("model_revision", "draft_revision", "speculative_decoding")
            }
        )
        payload.append(canonical_config(profile.get("fixed_server", {})))
    return json.dumps(payload, sort_keys=True)
