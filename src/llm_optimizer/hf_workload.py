"""Shared CPU-only HF ShareGPT workload preparation for notebooks and workers."""

from __future__ import annotations

import hashlib
import json
import random
import re
from urllib.parse import urlsplit


def dataset_id(value):
    if value.startswith("https://"):
        url = urlsplit(value)
        if url.netloc != "huggingface.co" or url.query or url.fragment:
            raise ValueError("Expected a credential-free Hugging Face dataset URL")
        parts = url.path.strip("/").split("/")
        if len(parts) != 3 or parts[0] != "datasets":
            raise ValueError(
                "Use the dataset repository URL; supply revision separately"
            )
        value = "/".join(parts[1:])
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", value):
        raise ValueError("Expected a Hugging Face dataset repo ID")
    return value


def fingerprint(manifest):
    body = {k: v for k, v in manifest.items() if k != "sha256"}
    return hashlib.sha256(
        json.dumps(
            body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()


def validate_manifest(manifest):
    if not isinstance(manifest, dict) or manifest.get("sha256") != fingerprint(
        manifest
    ):
        raise ValueError("Custom workload hash mismatch")
    for revision in (
        manifest.get("tokenizer_revision"),
        manifest.get("dataset", {}).get("revision"),
    ):
        if not isinstance(revision, str) or not re.fullmatch(
            r"[0-9a-fA-F]{40,64}", revision
        ):
            raise ValueError(
                "Custom workload requires pinned dataset and tokenizer commits"
            )
    dataset_id(manifest["dataset"]["repo_id"])
    if len(manifest.get("row_indices", [])) != len(manifest.get("requests", [])):
        raise ValueError("Custom workload row identity mismatch")
    requests = manifest.get("requests")
    if not isinstance(requests, list) or not requests:
        raise ValueError("Custom workload has no requests")
    for row in requests:
        if (
            not isinstance(row, dict)
            or set(row) != {"prompt", "prompt_len", "output_len"}
            or not isinstance(row["prompt"], str)
            or not row["prompt"].strip()
            or not isinstance(row["prompt_len"], int)
            or row["prompt_len"] < 1
            or not isinstance(row["output_len"], int)
            or row["output_len"] < 1
        ):
            raise ValueError("Invalid custom workload request")
    return manifest


class ShareGPTSchemaError(ValueError):
    """A schema failure with no source text or credentials in its message."""


def first_human(row):
    turns = row.get("conversations") if isinstance(row, dict) else None
    if not isinstance(turns, list) or not turns:
        raise ShareGPTSchemaError("Expected nonempty conversations [{from, value}]")
    prompt = None
    for turn in turns:
        if (
            not isinstance(turn, dict)
            or turn.get("from") not in {"human", "gpt", "system"}
            or not isinstance(turn.get("value"), str)
            or not turn["value"].strip()
        ):
            raise ShareGPTSchemaError("Invalid ShareGPT role or empty text")
        if turn["from"] == "human" and prompt is None:
            prompt = turn["value"]
    if prompt is None:
        raise ShareGPTSchemaError("ShareGPT conversation has no human prompt")
    return prompt


def prepare_hf_workload(
    dataset,
    model,
    model_revision,
    input_tokens,
    output_tokens,
    num_prompts=100,
    seed=1,
    *,
    api=None,
    loader=None,
    tokenizer=None,
):
    """Pin, validate and sample without replacement; repeat/truncate first-human tokens.

    Dependencies are optional until a custom dataset is requested. Auth comes only
    from the process HF environment/cache, never from serialized configuration.
    """
    if min(input_tokens, output_tokens, num_prompts) < 1:
        raise ValueError("Token lengths and prompt count must be positive")
    repo = dataset_id(dataset["repo_id"])
    try:
        if api is None:
            from huggingface_hub import HfApi

            api = HfApi()
        if not re.fullmatch(r"[0-9a-fA-F]{40,64}", model_revision or ""):
            model_revision = api.model_info(model, revision=model_revision).sha
        if not re.fullmatch(r"[0-9a-fA-F]{40,64}", model_revision or ""):
            raise ValueError("Tokenizer revision did not resolve to a commit")
        revision = api.dataset_info(repo, revision=dataset.get("revision")).sha
        if not re.fullmatch(r"[0-9a-fA-F]{40,64}", revision or ""):
            raise ValueError("Dataset revision did not resolve to a commit")
        if loader is None:
            from datasets import load_dataset

            loader = load_dataset
        rows = loader(
            repo,
            name=dataset.get("config"),
            split=dataset.get("split", "train"),
            revision=revision,
        )
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                model, revision=model_revision, trust_remote_code=False
            )
    except Exception:
        raise ValueError(
            "Custom HF dataset/tokenizer loading or access failed; "
            "check access and dependencies"
        ) from None
    valid = []
    try:
        for index, row in enumerate(rows):
            prompt = first_human(row)
            tokens = tokenizer.encode(prompt)
            if tokens:
                valid.append((index, tokens))
    except ShareGPTSchemaError:
        raise
    except Exception:
        raise ValueError("Custom HF dataset iteration or tokenization failed") from None
    if len(valid) < num_prompts:
        raise ValueError(
            f"Insufficient valid dataset rows: need {num_prompts}, found {len(valid)}"
        )
    selected = random.Random(seed).sample(valid, num_prompts)
    requests = []
    for _, tokens in selected:
        fixed = (tokens * ((input_tokens + len(tokens) - 1) // len(tokens)))[
            :input_tokens
        ]
        requests.append(
            {
                "prompt": tokenizer.decode(fixed),
                "prompt_len": input_tokens,
                "output_len": output_tokens,
            }
        )
    manifest = {
        "schema_version": 1,
        "dataset": {
            "repo_id": repo,
            "revision": revision,
            "config": dataset.get("config"),
            "split": dataset.get("split", "train"),
        },
        "row_indices": [i for i, _ in selected],
        "tokenizer": model,
        "tokenizer_revision": model_revision,
        "seed": seed,
        "length_policy": "first_human_repeat_or_truncate",
        "requests": requests,
    }
    manifest["sha256"] = fingerprint(manifest)
    return validate_manifest(manifest)
