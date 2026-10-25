"""Backport the three parser changes from SGLang PR #43714.

Run with the serving environment's Python. This does not add Gemma DSpark support.
"""

import argparse
import ast
import importlib.metadata
import importlib.util
from pathlib import Path


def patched_source(source):
    replacements = (
        ('    "gemma4",\n', '    "gemma4",\n    "gemma4_text",\n'),
        (
            "    text_config = config.text_config\n",
            '    text_config = getattr(config, "text_config", None) or config\n',
        ),
        (
            "        if isinstance(model, str) and text_config is not None:\n",
            "        if (\n"
            "            isinstance(model, str)\n"
            "            and text_config is not None\n"
            "            and text_config is not config\n"
            "        ):\n",
        ),
    )
    for before, after in replacements:
        if after in source and before not in source:
            continue
        # The model-type replacement contains its original prefix.
        if before == '    "gemma4",\n' and after in source:
            continue
        if source.count(before) != 1:
            raise RuntimeError(f"Unexpected parser source; refusing patch: {before!r}")
        source = source.replace(before, after, 1)
    ast.parse(source)
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, help="Explicit source file for offline tests"
    )
    args = parser.parse_args()
    path = args.source
    if path is None:
        version = importlib.metadata.version("sglang")
        if version != "0.5.22":
            raise RuntimeError(f"Expected SGLang 0.5.22, found {version}")
        spec = importlib.util.find_spec("sglang")
        path = Path(next(iter(spec.submodule_search_locations))) / (
            "srt/utils/hf_transformers/config.py"
        )
    original = path.read_text()
    updated = patched_source(original)
    if updated == original:
        print(f"Already patched: {path}")
        return
    backup = path.with_suffix(path.suffix + ".before-gemma4-text")
    if backup.exists():
        raise RuntimeError(f"Backup already exists; refusing overwrite: {backup}")
    backup.write_text(original)
    path.write_text(updated)
    print(f"Patched: {path}\nBackup: {backup}")


if __name__ == "__main__":
    main()
