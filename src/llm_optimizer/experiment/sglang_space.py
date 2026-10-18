"""Fixed categorical encoding for SGLang's coupled cache/scheduler settings."""

CACHE_POLICY_PARAMETER = "cache_policy"
CACHE_POLICIES = (
    "cache_on__fcfs",
    "cache_on__lpm",
    "cache_on__shortest-prefill-first",
    "cache_off__fcfs",
    "cache_off__lpm",
)


def resolve_cache_policy(value):
    """Return native settings without exposing the synthetic knob to the server."""
    if value not in CACHE_POLICIES:
        raise ValueError(f"Unknown SGLang cache policy: {value!r}")
    cache, policy = value.split("__", 1)
    return {"disable_radix_cache": cache == "cache_off", "schedule_policy": policy}


def sampling_choices(choices):
    """Freeze one domain per study, respecting verified native-choice subsets."""
    result = dict(choices)
    cache_choices = result.pop("disable_radix_cache")
    policy_choices = result.pop("schedule_policy")
    valid = [
        value
        for value in CACHE_POLICIES
        if resolve_cache_policy(value)["disable_radix_cache"] in cache_choices
        and resolve_cache_policy(value)["schedule_policy"] in policy_choices
    ]
    if not valid:
        raise ValueError("SGLang preflight choices contain no valid cache/policy pair")
    result[CACHE_POLICY_PARAMETER] = valid
    return result
