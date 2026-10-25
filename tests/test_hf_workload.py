"""CPU-only tests of the shared notebook/controller dataset workload contract."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from llm_optimizer.hf_workload import (
    dataset_id,
    first_human,
    prepare_hf_workload,
    validate_manifest,
)

REV = "a" * 40


class Tokenizer:
    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


def materialize(**overrides):
    rows = [
        {
            "conversations": [
                {"from": "system", "value": "rules"},
                {"from": "human", "value": str(i)},
                {"from": "gpt", "value": "answer"},
            ]
        }
        for i in range(10)
    ]
    args = {
        "dataset": {"repo_id": "org/data"},
        "model": "org/model",
        "model_revision": REV,
        "input_tokens": 8,
        "output_tokens": 4,
        "num_prompts": 3,
        "seed": 12,
        "api": SimpleNamespace(dataset_info=lambda *a, **kw: SimpleNamespace(sha=REV)),
        "loader": lambda *a, **kw: rows,
        "tokenizer": Tokenizer(),
    }
    args.update(overrides)
    return prepare_hf_workload(**args)


def test_reproducible_fixed_workload():
    a, b = materialize(), materialize()
    assert a == b
    assert a["sha256"] != materialize(seed=13)["sha256"]
    assert len(set(a["row_indices"])) == 3
    assert all(len(r["prompt"]) == 8 and r["output_len"] == 4 for r in a["requests"])
    assert a["dataset"]["revision"] == REV
    assert a["tokenizer_revision"] == REV
    a["requests"][0]["prompt"] = "tampered"
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_manifest(a)


@pytest.mark.parametrize(
    "row",
    [
        {},
        {"conversations": []},
        {"conversations": [{"from": "user", "value": "hello"}]},
        {"conversations": [{"from": "human", "value": " "}]},
        {"conversations": [{"from": "gpt", "value": "answer"}]},
    ],
)
def test_invalid_schema(row):
    with pytest.raises(ValueError):
        materialize(loader=lambda *a, **kw: [row])


def test_first_human_not_first_turn():
    assert (
        first_human(
            {
                "conversations": [
                    {"from": "system", "value": "rules"},
                    {"from": "human", "value": "prompt"},
                    {"from": "human", "value": "later"},
                ]
            }
        )
        == "prompt"
    )


def test_insufficient_rows_never_repeat_records():
    with pytest.raises(ValueError, match="Insufficient"):
        materialize(num_prompts=11)


def test_missing_access_redacts_error_no_fallback():
    loader = Mock(side_effect=RuntimeError("secret-hf-token"))
    with pytest.raises(ValueError, match="loading or access failed") as error:
        materialize(loader=loader)
    assert "secret-hf-token" not in str(error.value)
    assert loader.call_count == 1


def test_revision_pinned_before_load():
    loader = Mock(return_value=[{"conversations": [{"from": "human", "value": "x"}]}])
    materialize(loader=loader, num_prompts=1)
    assert loader.call_args.kwargs["revision"] == REV


@pytest.mark.parametrize(
    "value",
    [
        "https://huggingface.co/datasets/org/data?token=secret",
        "https://user:secret@huggingface.co/datasets/org/data",
        "/tmp/data",
    ],
)
def test_credentials_and_local_paths_rejected(value):
    with pytest.raises(ValueError):
        dataset_id(value)
