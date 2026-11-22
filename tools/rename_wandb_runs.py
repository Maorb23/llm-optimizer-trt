import argparse
import re

import wandb


PREFIXES = ("fp8", "trt140rc3", "trtl40rc3")
SUFFIX_MARKER = "__model-"


def model_suffix(model):
    base_model = str(model).split("@", 1)[0].strip()
    suffix = re.sub(r"[^A-Za-z0-9._-]+", "-", base_model).strip("-")
    if not suffix:
        raise ValueError(f"Invalid target_model: {model!r}")
    return suffix


def get_target_model(config):
    for key in ("target_model", "TARGET_MODEL", "TAGET_MODEL"):
        value = config.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def renamed(name, model):
    original = name.split(SUFFIX_MARKER, 1)[0]
    return f"{original}{SUFFIX_MARKER}{model_suffix(model)}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project",
        default=(
            "maorblumberg-tel-aviv-university/"
            "inference-optimization"
        ),
        help="W&B entity/project",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes; otherwise only preview",
    )
    args = parser.parse_args()

    api = wandb.Api(timeout=60)
    planned = []
    skipped = 0

    for run in api.runs(args.project):
        old_name = run.name or ""
        if not old_name.lower().startswith(PREFIXES):
            continue

        model = get_target_model(run.config)
        if not model:
            print(f"SKIP {run.id}: missing target_model ({old_name})")
            skipped += 1
            continue

        new_name = renamed(old_name, model)
        if new_name == old_name:
            continue

        planned.append((run, old_name, new_name))
        print(f"\nRun ID: {run.id}")
        print(f"Model:  {model}")
        print(f"Before: {old_name}")
        print(f"After:  {new_name}")

    print(f"\nPlanned: {len(planned)}; skipped: {skipped}")

    if not args.apply:
        print("Preview only. Add --apply to rename.")
        return

    failures = 0
    for run, old_name, new_name in planned:
        try:
            run.name = new_name
            run.update()
            print(f"RENAMED {run.id}: {new_name}")
        except Exception as error:
            failures += 1
            print(f"FAILED {run.id}: {error}")

    print(f"\nRenamed: {len(planned) - failures}; failed: {failures}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
