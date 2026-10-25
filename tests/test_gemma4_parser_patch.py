"""Backport application checks; hardware/config checks live in tools/."""

import ast
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "tools" / "patch_sglang_gemma4_text.py"
SPEC = importlib.util.spec_from_file_location("gemma4_parser_patch", SCRIPT)
PATCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCHER)

SOURCE = """\
_GEMMA4_MODEL_TYPES = (
    "gemma4",
)
def normalize(config):
    text_config = config.text_config
class Parser:
    def parse(self, model, config, text_config):
        if isinstance(model, str) and text_config is not None:
            for key, val in vars(text_config).items():
                if not hasattr(config, key):
                    setattr(config, key, val)
"""


def test_patch_skips_self_copy_and_preserves_nested_copy():
    namespace = {}
    exec(compile(PATCHER.patched_source(SOURCE), "patched", "exec"), namespace)

    class Guarded:
        def __init__(self):
            self.num_key_value_heads = 8

        def __getattribute__(self, key):
            if key == "num_key_value_heads":
                raise RuntimeError("ambiguous per-layer access")
            return object.__getattribute__(self, key)

    guarded = Guarded()
    namespace["Parser"]().parse("draft", guarded, guarded)

    class Config:
        pass

    outer, inner = Config(), Config()
    inner.hidden_size = 3840
    namespace["Parser"]().parse("target", outer, inner)
    assert outer.hidden_size == 3840


def test_patch_is_idempotent_and_valid_python():
    updated = PATCHER.patched_source(SOURCE)
    ast.parse(updated)
    assert PATCHER.patched_source(updated) == updated
    namespace = {}
    exec(updated, namespace)
    assert "gemma4_text" in namespace["_GEMMA4_MODEL_TYPES"]


def test_unknown_source_rejected():
    with pytest.raises(RuntimeError, match="refusing patch"):
        PATCHER.patched_source("# incompatible source")
