import argparse
import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path


SUFFIX_MARKER = "__model-"


def base_model(model):
    return str(model).split("@", 1)[0].strip()


def model_suffix(model):
    suffix = re.sub(
        r"[^A-Za-z0-9._-]+", "-", base_model(model)
    ).strip("-")
    if not suffix:
        raise ValueError(f"Invalid target_model: {model!r}")
    return suffix


def config_models(folder):
    path = folder / "config.json"
    if not path.is_file():
        return set()

    config = json.loads(path.read_text())
    models = set()

    direct = config.get("target_model")
    if isinstance(direct, str) and direct.strip():
        models.add(base_model(direct))

    phase_models = config.get("settings", {}).get("phase_models", {})
    for model in phase_models.values():
        if isinstance(model, str) and model.strip():
            models.add(base_model(model))

    return models


def database_models(folder):
    db = folder / "studies.sqlite3"
    if not db.is_file():
        return set()

    models = set()
    uri = db.resolve().as_uri() + "?mode=ro"

    with closing(sqlite3.connect(uri, uri=True)) as connection:
        rows = connection.execute(
            """
            SELECT value_json
            FROM study_user_attributes
            WHERE key = 'contract'
            """
        )
        for (value_json,) in rows:
            contract = json.loads(value_json)
            model = contract.get("target_model")
            if isinstance(model, str) and model.strip():
                models.add(base_model(model))

    return models


def find_run_directories(root):
    directories = {
        path.parent
        for path in root.rglob("studies.sqlite3")
        if path.is_file()
    }

    for path in root.rglob("config.json"):
        if path.parent in directories:
            continue
        try:
            config = json.loads(path.read_text())
            if config.get("settings", {}).get("phase_models"):
                directories.add(path.parent)
        except (OSError, ValueError, AttributeError):
            continue

    return sorted(
        directories,
        key=lambda path: (-len(path.parts), str(path)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=Path,
        help="Downloaded runs directory or snapshot directory",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes; otherwise only preview",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Directory does not exist: {root}")

    planned = []
    skipped = 0

    for folder in find_run_directories(root):
        try:
            models = database_models(folder)
            if not models:
                models = config_models(folder)

            if len(models) != 1:
                print(
                    f"SKIP {folder}: expected one base model, "
                    f"found {sorted(models)}"
                )
                skipped += 1
                continue

            model = next(iter(models))
            original = folder.name.split(SUFFIX_MARKER, 1)[0]
            destination = folder.with_name(
                f"{original}{SUFFIX_MARKER}{model_suffix(model)}"
            )

            if destination == folder:
                continue

            if destination.exists():
                print(f"SKIP {folder}: destination exists: {destination}")
                skipped += 1
                continue

            planned.append((folder, destination))
            print(f"\nModel:  {model}")
            print(f"Before: {folder}")
            print(f"After:  {destination}")

        except (
            OSError,
            ValueError,
            AttributeError,
            sqlite3.DatabaseError,
        ) as error:
            skipped += 1
            print(f"SKIP {folder}: {error}")

    print(f"\nPlanned: {len(planned)}; skipped: {skipped}")

    if not args.apply:
        print("Preview only. Add --apply to rename.")
        return

    failures = 0
    for source, destination in planned:
        try:
            if destination.exists():
                raise FileExistsError(destination)
            source.rename(destination)
            print(f"RENAMED {destination}")
        except OSError as error:
            failures += 1
            print(f"FAILED {source}: {error}")

    print(f"\nRenamed: {len(planned) - failures}; failed: {failures}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
