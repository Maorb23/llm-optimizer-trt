"""Run a deployment-owned v4 campaign profile without a notebook kernel."""

from __future__ import annotations

from copy import deepcopy
from importlib import import_module

from .campaign import CampaignWorkloads, attach_campaign, evaluate_held_out
from .runtime import Generation
from .scorers import ScorerSettings

RUNNERS = {
    "vllm": ("vllm_policy_v3", "VLLMRunnerV3"),
    "sglang": ("sglang_policy_v3", "SGLangRunnerV3"),
    "trtllm": ("trtllm_policy_v3", "TRTLLMRunnerV3"),
}


def run_campaign(profile, *, framework, model, revision, gpu_count, run_root):
    """Use the notebooks' runner, model setup, workload binding and study APIs."""
    module_name, class_name = RUNNERS[framework]
    policy = import_module(f"..{module_name}", __package__)
    runner_class = getattr(policy, class_name)
    settings = deepcopy(profile["settings"])
    model_setup = deepcopy(profile["model_setup"])
    if (
        model_setup.pop("target_model_id") != model
        or model_setup.pop("target_revision") != revision
    ):
        raise ValueError(
            "Campaign profile and request must select the same pinned model"
        )
    runner_options = deepcopy(profile["runner_options"])
    workload = deepcopy(profile["workload"])
    if workload.get("source") != "hf":
        raise ValueError("Remote v4 profiles require a pinned Hugging Face dataset")
    workload["generation"] = Generation(**workload["generation"])
    workload["scorer"] = ScorerSettings(**workload.get("scorer", {}))
    workload["requests"] = {
        int(key): value for key, value in workload["requests"].items()
    }
    workload["buckets"] = {
        int(key): value for key, value in workload.get("buckets", {}).items()
    }
    if set(workload["requests"]) != set(settings["concurrencies"]):
        raise ValueError("The workload request map must match campaign concurrencies")
    if "requests_by_concurrency" in runner_options:
        raise ValueError("Use workload.requests to specify request counts")
    if not profile["objective_groups"]:
        raise ValueError("Configure at least one objective group")
    policy.verify_runtime()

    phases = ("A", "B")
    settings.update(
        frameworks=[framework],
        real_frameworks=[framework],
        phases=list(phases),
        run_real_gpu=False,
        quick_demo=False,
        gpu_count=gpu_count,
        phase_models=dict.fromkeys(phases, model),
        phase_requests=dict.fromkeys(phases, max(workload["requests"].values())),
    )
    for phase in phases:
        settings["real_preflight"][phase][framework].setdefault(
            "expanded_choices", runner_options["expanded_choices"][framework]
        )
    setup = runner_class(
        settings,
        run_root=run_root / "setup",
        requests_by_concurrency=workload["requests"],
        **runner_options,
    )
    setup.lab.RUN_REAL_GPU = True
    snapshot = setup.lab.configure_hf_models(
        model, target_revision=revision, framework=framework, **model_setup
    )
    runtime_profile = setup.lab.REAL_PREFLIGHT["B", framework]
    settings["model_snapshots"] = dict.fromkeys(phases, snapshot["target_path"])
    workloads = CampaignWorkloads(
        framework=framework,
        models=settings["phase_models"],
        revisions=dict.fromkeys(phases, revision),
        snapshots=settings["model_snapshots"],
        **workload,
    )
    # Phase A uses the same pinned target. Phase B includes the native draft setup.
    phase_a = settings["real_preflight"]["A"][framework]
    for field in (
        "framework_version",
        "cuda_version",
        "driver_version",
        "gpu_type",
        "model_revision",
    ):
        phase_a[field] = runtime_profile[field]
    settings["real_preflight"]["B"][framework] = runtime_profile
    for phase in phases:
        settings["real_preflight"][phase][framework]["dataset_v4"] = workloads.identity
    settings.setdefault("base_workload", {})["dataset_name"] = "frozen_chat_v4"
    settings["run_real_gpu"] = True
    runner = attach_campaign(
        runner_class(
            settings,
            run_root=run_root,
            requests_by_concurrency=workload["requests"],
            **runner_options,
        ),
        workloads,
    )
    studies = {}
    for group in profile["objective_groups"]:
        studies.update(runner.run_objective_group(**group))
    held_out = profile.get("held_out")
    if held_out is not None:
        study = studies[held_out["study"]]
        evaluate_held_out(runner, study, pareto_trial=held_out.get("pareto_trial"))
    if not studies:
        raise ValueError("Configure at least one study with completed measurements")
    return runner, studies
