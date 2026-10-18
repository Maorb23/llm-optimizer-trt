import pytest

from llm_optimizer.remote.contracts import OptimizationRequest


def request(**changes):
    return {
        "model": "org/model",
        "frameworks": ["vllm"],
        "gpu_type": "gpu-h100-sxm",
        "images": {"vllm": "test:image"},
        "bucket": "results",
        **changes,
    }


def test_legacy_spot_environment_is_snapshotted(monkeypatch):
    monkeypatch.setenv("NEBIUS_SPOT_PRICING_POLICY_ID", "policy-example")
    parsed = OptimizationRequest.parse(request(preemptible=True)).to_dict()
    assert parsed["spot_pricing_policy_id"] == "policy-example"
    assert parsed["follows_spot_price"] is False
    monkeypatch.setenv("NEBIUS_SPOT_PRICING_POLICY_ID", "different-policy")
    assert OptimizationRequest.parse(parsed).to_dict() == parsed


def test_explicit_follow_spot_overrides_environment(monkeypatch):
    monkeypatch.setenv("NEBIUS_SPOT_PRICING_POLICY_ID", "policy-example")
    parsed = OptimizationRequest.parse(
        request(preemptible=True, follows_spot_price=True)
    )
    assert parsed.spot_pricing_policy_id is None
    assert parsed.follows_spot_price is True


@pytest.mark.parametrize(
    "changes",
    [
        {"preemptible": True, "follows_spot_price": False},
        {
            "preemptible": True,
            "spot_pricing_policy_id": "policy",
            "follows_spot_price": True,
        },
        {"preemptible": False, "spot_pricing_policy_id": "policy"},
        {"preemptible": False, "follows_spot_price": True},
        {"preemptible": True, "spot_pricing_policy_id": ""},
        {"preemptible": True, "spot_pricing_policy_id": 12},
        {"preemptible": True, "follows_spot_price": "true"},
    ],
)
def test_invalid_spot_policy_is_rejected(changes):
    with pytest.raises(ValueError):
        OptimizationRequest.parse(request(**changes))
