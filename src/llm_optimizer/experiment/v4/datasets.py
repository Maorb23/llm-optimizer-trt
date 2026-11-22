"""Versioned, reference-free workloads with immutable canonical serialization.

Field conventions follow hf_validator/web/infiro_platform/workloads.py, adapter
version 1. Unlike that throughput adapter, all conversation history is retained.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

VERSION = "datasets-v4.1"
TASKS = {"conversation", "gpqa_diamond", "humaneval", "humaneval_plus", "extraction"}


def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def pinned_revision(value):
    if not re.fullmatch(r"(?:[0-9a-f]{40}|sha256:[0-9a-f]{64})", value):
        raise ValueError(
            "Use an immutable 40-character HF commit or sha256:<local content hash>"
        )
    return value


@dataclass(frozen=True)
class Example:
    example_id: str
    task: str
    payload_json: str
    reference_json: str

    @property
    def payload(self):
        return json.loads(self.payload_json)

    @property
    def reference(self):
        return json.loads(self.reference_json)


def adapt(row, source_format, *, example_id=None, seed=0):
    """Hold out only the final assistant turn. Prior assistant turns are context."""
    task = source_format if source_format in TASKS else "conversation"
    identifier = str(
        example_id
        if example_id is not None
        else row.get("id", row.get("task_id", digest(row)))
    )
    reference = {}
    if source_format in {"messages", "sharegpt"}:
        source = row["messages" if source_format == "messages" else "conversations"]
        roles = {
            "human": "user",
            "gpt": "assistant",
            "system": "system",
            "user": "user",
            "assistant": "assistant",
        }
        messages = []
        for turn in source:
            role = turn.get("role") if source_format == "messages" else turn.get("from")
            content = (
                turn.get("content")
                if source_format == "messages"
                else turn.get("value")
            )
            if role not in roles or not isinstance(content, str) or not content.strip():
                raise ValueError(
                    "Each turn requires a supported role and nonempty text"
                )
            messages.append({"role": roles[role], "content": content})
        if messages and messages[-1]["role"] == "assistant":
            reference["target"] = messages.pop()["content"]
        if not messages or messages[-1]["role"] != "user":
            raise ValueError(
                "Conversation must end with a user prompt after holding out target"
            )
    elif source_format == "alpaca":
        instruction, extra = row["instruction"], row.get("input", "")
        if (
            not isinstance(instruction, str)
            or not instruction.strip()
            or not isinstance(extra, str)
        ):
            raise ValueError("Alpaca instruction and input must be text")
        messages = [
            {
                "role": "user",
                "content": instruction + ("\n\n" + extra if extra.strip() else ""),
            }
        ]
        if "output" in row:
            reference["target"] = row["output"]
    elif task == "gpqa_diamond":
        # Diamond must be selected by the source config; never silently substitute main.
        choices = [(row["Correct Answer"], True)] + [
            (row[f"Incorrect Answer {i}"], False) for i in range(1, 4)
        ]
        random.Random(f"{seed}:{identifier}").shuffle(choices)
        reference["answer"] = "ABCD"[
            next(i for i, (_, correct) in enumerate(choices) if correct)
        ]
        text = (
            row["Question"]
            + "\n"
            + "\n".join(
                f"{label}. {answer}" for label, (answer, _) in zip("ABCD", choices)
            )
        )
        messages = [
            {
                "role": "user",
                "content": text + "\nEnd with Final answer: <A, B, C, or D>.",
            }
        ]
    elif task in {"humaneval", "humaneval_plus"}:
        prompt = row["prompt"]
        reference = {"prompt": prompt, "entry_point": row["entry_point"]}
        for key in ("test", "canonical_solution", "base_input", "plus_input", "atol"):
            if key in row:
                reference[key] = row[key]
        if task == "humaneval" and "test" not in reference:
            raise ValueError("HumanEval requires executable tests")
        if task == "humaneval_plus" and not all(
            k in reference
            for k in ("canonical_solution", "base_input", "plus_input", "atol")
        ):
            raise ValueError("HumanEval+ requires full EvalPlus inputs and oracle")
        messages = [
            {
                "role": "user",
                "content": "Complete this Python function. Return only Python code.\n"
                + prompt,
            }
        ]
    elif task == "extraction":
        schema = row["schema"]
        reference = {"expected": row["expected"], "schema": schema}
        # Only an explicit input projection is allowed; never serialize the whole row.
        messages = [
            {
                "role": "user",
                "content": row["instruction"]
                + "\n"
                + row["text"]
                + "\nReturn JSON matching this schema:\n"
                + canonical(schema),
            }
        ]
    else:
        raise ValueError(f"Unsupported format: {source_format}")
    if not any(m["role"] == "user" for m in messages):
        raise ValueError("Missing user message")
    return Example(
        identifier, task, canonical({"messages": messages}), canonical(reference)
    )


@dataclass(frozen=True)
class Sampling:
    mode: str = "natural"
    count: int = 32
    seed: int = 0
    # Each bucket is (minimum inclusive, maximum exclusive, number of examples).
    buckets: tuple = ()

    def __post_init__(self):
        if self.mode not in {"natural", "token_bucket"} or self.count < 1:
            raise ValueError("Invalid sampling mode/count")
        object.__setattr__(self, "buckets", tuple(tuple(b) for b in self.buckets))
        if self.mode == "token_bucket":
            if not self.buckets or sum(b[2] for b in self.buckets) != self.count:
                raise ValueError("Bucket quotas must sum to count")
            previous = 0
            for low, high, quota in sorted(self.buckets):
                if low < previous or high <= low or quota < 1:
                    raise ValueError("Buckets must be positive, disjoint intervals")
                previous = high
        elif self.buckets:
            raise ValueError("Natural sampling does not use buckets")


def sample(
    examples: Iterable[Example], settings: Sampling, token_count: Callable | None = None
):
    examples = list(examples)
    if len({e.example_id for e in examples}) != len(examples):
        raise ValueError("Duplicate example IDs")
    pool = unique_payloads(examples)
    random.Random(settings.seed).shuffle(pool)
    if settings.mode == "natural":
        selected = pool[: settings.count]
    else:
        if token_count is None:
            raise ValueError(
                "Token buckets require a pinned tokenizer chat-template counter"
            )
        lengths = {e.example_id: token_count(e.payload["messages"]) for e in pool}
        selected = []
        for low, high, quota in settings.buckets:
            candidates = [e for e in pool if low <= lengths[e.example_id] < high]
            if len(candidates) < quota:
                raise ValueError(f"Insufficient examples in bucket [{low}, {high})")
            selected.extend(candidates[:quota])
    if len(selected) != settings.count:
        raise ValueError(
            "Insufficient unique examples; task text is never repeated or padded"
        )
    return tuple(selected)


def unique_payloads(examples):
    """Keep the first stable ID for duplicate inference tasks."""
    seen = set()
    result = []
    for example in sorted(examples, key=lambda e: e.example_id):
        key = digest(example.payload)
        if key not in seen:
            seen.add(key)
            result.append(example)
    return result


@dataclass(frozen=True)
class FrozenManifest:
    document_json: str
    references_json: str

    @property
    def document(self):
        return json.loads(self.document_json)

    @property
    def references(self):
        return json.loads(self.references_json)

    @property
    def signature(self):
        return digest(self.document)

    def verify(self):
        from .runtime import Generation

        doc, refs = self.document, self.references
        if doc["version"] != VERSION or digest(refs) != doc["references_sha256"]:
            raise ValueError("Manifest version or reference hash mismatch")
        ids = [e["example_id"] for e in doc["examples"]]
        if len(set(ids)) != len(ids) or set(ids) != set(refs):
            raise ValueError("Manifest IDs and references differ")
        if len({e["payload_sha256"] for e in doc["examples"]}) != len(ids):
            raise ValueError("Repeated inference tasks are not allowed")
        sampling = Sampling(**doc["sampling"])
        Generation(**doc["generation"])
        if not ids or len(ids) != sampling.count or doc["task"] not in TASKS:
            raise ValueError("Manifest task/count mismatch")
        supported_formats = {"sharegpt", "messages", "alpaca"} | (
            TASKS - {"conversation"}
        )
        if doc["source_format"] not in supported_formats:
            raise ValueError("Unknown manifest source format")
        expected_task = (
            doc["source_format"] if doc["source_format"] in TASKS else "conversation"
        )
        if doc["task"] != expected_task:
            raise ValueError("Source format and task differ")
        if doc["task"] == "gpqa_diamond" and doc["dataset_config"] != "gpqa_diamond":
            raise ValueError("GPQA Diamond requires its explicit dataset config")
        for key in ("revision", "model_revision", "tokenizer_revision"):
            pinned_revision(doc[key])
        for item in doc["examples"]:
            if digest(item["payload"]) != item["payload_sha256"]:
                raise ValueError("Payload hash mismatch")
            if set(item["payload"]) != {"messages"}:
                raise ValueError("Inference payload may contain only messages")
            messages = item["payload"]["messages"]
            if not messages or messages[-1]["role"] != "user":
                raise ValueError("Inference conversation must end with a user turn")
            for message in messages:
                if (
                    set(message) != {"role", "content"}
                    or message["role"] not in {"system", "user", "assistant"}
                    or not isinstance(message["content"], str)
                    or not message["content"].strip()
                ):
                    raise ValueError("Invalid inference message")
        return self

    def save(self, directory):
        self.verify()
        root = Path(directory) / self.signature
        root.mkdir(parents=True, exist_ok=True)
        for name, data in (
            ("manifest.json", self.document_json),
            ("references.json", self.references_json),
        ):
            path = root / name
            if path.exists():
                if path.read_text() != data:
                    raise ValueError("Refusing to overwrite a frozen artifact")
            else:
                with path.open("x") as stream:
                    stream.write(data)
                path.chmod(0o444)
        return root

    @classmethod
    def load(cls, directory):
        root = Path(directory)
        result = cls(
            (root / "manifest.json").read_text(), (root / "references.json").read_text()
        ).verify()
        if root.name != result.signature:
            raise ValueError(
                "Manifest content no longer matches its frozen directory hash"
            )
        return result


def freeze(
    examples,
    *,
    dataset_id,
    revision,
    source_format,
    split,
    generation,
    sampling,
    model,
    model_revision,
    tokenizer_revision,
    evaluation,
    dataset_config=None,
    role="train",
    tokenizer_id=None,
    tokenizer_settings=None,
):
    from dataclasses import asdict

    examples = tuple(examples)
    if not examples or role not in {"train", "validation"}:
        raise ValueError("Nonempty workload and train/validation role required")
    if len({e.example_id for e in examples}) != len(examples):
        raise ValueError("Duplicate example IDs")
    if len({e.task for e in examples}) != 1:
        raise ValueError("A manifest must contain exactly one task")
    if source_format == "gpqa_diamond" and dataset_config != "gpqa_diamond":
        raise ValueError("GPQA Diamond requires dataset_config='gpqa_diamond'")
    refs = {e.example_id: e.reference for e in examples}
    doc = {
        "version": VERSION,
        "dataset_id": dataset_id,
        "revision": pinned_revision(revision),
        "source_format": source_format,
        "dataset_config": dataset_config,
        "split": split,
        "role": role,
        "task": examples[0].task,
        "model": model,
        "model_revision": pinned_revision(model_revision),
        "tokenizer_id": tokenizer_id or model,
        "tokenizer_revision": pinned_revision(tokenizer_revision),
        "tokenizer_settings": tokenizer_settings or {},
        "generation": asdict(generation),
        "sampling": asdict(sampling),
        "evaluation": evaluation,
        "references_sha256": digest(refs),
        "examples": [
            {
                "example_id": e.example_id,
                "payload": e.payload,
                "payload_sha256": digest(e.payload),
            }
            for e in examples
        ],
    }
    return FrozenManifest(canonical(doc), canonical(refs)).verify()


def load_hf(dataset_id, *, revision, source_format, split, config=None, seed=0):
    """Lazy optional dependency. Explicit revision, config and split, no fallback."""
    from datasets import load_dataset

    pinned_revision(revision)
    if not re.fullmatch("[0-9a-f]{40}", revision):
        raise ValueError("HF loading requires a commit SHA")
    if source_format == "gpqa_diamond" and config != "gpqa_diamond":
        raise ValueError("Select the GPQA Diamond config explicitly")
    rows = load_dataset(dataset_id, name=config, split=split, revision=revision)
    return tuple(adapt(row, source_format, seed=seed) for row in rows)


def held_out(examples, *, validation_count, seed=0):
    pool = sorted(examples, key=lambda e: e.example_id)
    if len({e.example_id for e in pool}) != len(pool):
        raise ValueError("Duplicate IDs")
    pool = unique_payloads(pool)
    if not 0 < validation_count < len(pool):
        raise ValueError("Held-out partition must leave both partitions nonempty")
    random.Random(seed).shuffle(pool)
    return tuple(pool[validation_count:]), tuple(pool[:validation_count])
