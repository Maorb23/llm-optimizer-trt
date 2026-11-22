import json
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from llm_optimizer.experiment.v4 import headless
from llm_optimizer.remote.contracts import OptimizationRequest
from llm_optimizer.remote.worker import run_worker


def request(**options):
    return {
        "model": "org/model",
        "frameworks": ["vllm"],
        "gpu_type": "gpu-l40s-d",
        "images": {"vllm": "image"},
        "bucket": "test",
        "dry_run": False,
        **options,
    }


def test_v4_selection_is_opt_in_and_validated():
    assert "campaign_v4" not in OptimizationRequest.parse(request()).to_dict()
    with pytest.raises(ValueError, match="profile by name"):
        OptimizationRequest.parse(request(campaign_v4="../profile", revision="a" * 40))
    with pytest.raises(ValueError, match="pinned model"):
        OptimizationRequest.parse(request(campaign_v4="profile"))
    with pytest.raises(ValueError, match="Configure workload"):
        OptimizationRequest.parse(
            request(campaign_v4="profile", revision="a" * 40, client_args=["seed=21"])
        )


def test_worker_dispatches_v4_and_uploads_study(tmp_path, monkeypatch):
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    (profiles / "demo.json").write_text(json.dumps({"vllm": {"settings": {}}}))
    monkeypatch.setenv("LLM_OPTIMIZER_CAMPAIGN_PROFILES", str(profiles))

    def execute(profile, *, framework, model, revision, gpu_count, run_root):
        assert (profile, framework, model, revision, gpu_count) == (
            {"settings": {}},
            "vllm",
            "org/model",
            "a" * 40,
            1,
        )
        run_root.mkdir()
        (run_root / "study.json").write_text("{}")
        return SimpleNamespace(run_dir=run_root), {"study": None}

    monkeypatch.setattr(headless, "run_campaign", execute)

    class Store:
        def upload(self, path, run_id, framework):
            assert (run_id, framework) == ("run-1", "vllm")
            if path.suffix == ".zip":
                with ZipFile(path) as archive:
                    assert archive.namelist() == ["study.json"]
            return f"s3://test/{path.name}"

    result = run_worker(
        {
            "request": request(campaign_v4="demo", revision="a" * 40),
            "run_id": "run-1",
            "framework": "vllm",
        },
        store=Store(),
        work_dir=tmp_path / "job",
    )
    assert result["status"] == "completed"
    assert result["studies"] == ["study"]
    assert result["artifacts"] == {"campaign-v4.zip": "s3://test/campaign-v4.zip"}
