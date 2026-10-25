"""Keep the serving base's installed versions while adding worker dependencies."""

import argparse
import importlib.metadata as metadata
import json
import re
from pathlib import Path

# The optimizer itself is replaced. The legacy NVML distribution is intentionally
# removed and its shared pynvml module repaired with nvidia-ml-py in the Dockerfile.
EXCLUDED = {"llm-optimizer", "nvidia-ml-py3"}


def installed():
    result = {}
    for dist in metadata.distributions():
        name = re.sub(r"[-_.]+", "-", dist.metadata["Name"]).lower()
        if name not in EXCLUDED:
            result[name] = dist.version
    return result


def snapshot(prefix):
    versions = installed()
    prefix.with_suffix(".json").write_text(json.dumps(versions, indent=2) + "\n")
    prefix.with_suffix(".txt").write_text(
        "".join(f"{name}=={version}\n" for name, version in sorted(versions.items()))
    )


def verify(prefix):
    original = json.loads(prefix.with_suffix(".json").read_text())
    current = installed()
    changed = {
        name: [version, current.get(name)]
        for name, version in original.items()
        if current.get(name) != version
    }
    if changed:
        raise RuntimeError(f"Serving base packages changed: {changed}")
    print(f"Preserved {len(original)} serving-base package versions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["snapshot", "verify"])
    parser.add_argument("prefix", type=Path)
    args = parser.parse_args()
    {"snapshot": snapshot, "verify": verify}[args.action](args.prefix)
