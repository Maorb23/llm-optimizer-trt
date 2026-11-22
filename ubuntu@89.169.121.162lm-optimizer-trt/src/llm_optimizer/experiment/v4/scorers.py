"""Task quality evaluation, called only after runtime measurement has completed."""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from .datasets import canonical

SCORER_VERSION = "scorers-v4.1"


@dataclass(frozen=True)
class ScorerSettings:
    image: str | None = None
    timeout_s: float = 30
    code_mode: str = "auto"
    schema_draft: str = "2020-12"

    def __post_init__(self):
        if self.timeout_s <= 0 or self.code_mode not in {
            "auto",
            "completion",
            "solution",
        }:
            raise ValueError("Invalid scorer timeout/code mode")
        if self.schema_draft != "2020-12":
            raise ValueError("Only JSON Schema Draft 2020-12 is supported")
        if self.image is not None and not re.fullmatch(
            r"(?:[^\s]+@sha256:|sha256:)[0-9a-f]{64}", self.image
        ):
            raise ValueError(
                "Executable evaluation requires a digest-pinned container image"
            )

    @property
    def identity(self):
        import importlib.metadata

        try:
            schema_version = importlib.metadata.version("jsonschema")
        except importlib.metadata.PackageNotFoundError:
            schema_version = "unavailable"
        return {
            "version": SCORER_VERSION,
            "jsonschema_version": schema_version,
            **asdict(self),
        }


def gpqa_score(text, reference):
    # Require an unambiguous final answer, never search the chain of thought.
    text = text.rstrip()
    matches = list(
        re.finditer(
            r"(?im)^[ \t]*Final answer[ \t]*:[ \t]*\(?([ABCD])\)?[. \t]*$", text
        )
    )
    if matches and (len(matches) != 1 or matches[0].end() != len(text)):
        return {"accuracy": 0.0, "predicted": None}
    matches = [m.group(1).upper() for m in matches]
    if not matches and re.fullmatch(r"\s*\(?[ABCD]\)?[.\s]*", text):
        matches = re.findall("[ABCD]", text)
    predicted = matches[0] if len(matches) == 1 else None
    return {"accuracy": float(predicted == reference["answer"]), "predicted": predicted}


def _reject_constant(value):
    raise ValueError(f"Invalid JSON constant: {value}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


def _leaves(value, prefix=()):
    if isinstance(value, dict) and value:
        return {
            p: v
            for k, child in value.items()
            for p, v in _leaves(child, prefix + (k,)).items()
        }
    if isinstance(value, list) and value:
        return {
            p: v
            for k, child in enumerate(value)
            for p, v in _leaves(child, prefix + (k,)).items()
        }
    return {prefix: value}


def extraction_score(text, reference):
    from jsonschema import Draft202012Validator, FormatChecker

    # Schema errors are scorer failures; validate even when generated JSON is invalid.
    Draft202012Validator.check_schema(reference["schema"])
    validator = Draft202012Validator(
        reference["schema"], format_checker=FormatChecker()
    )
    expected = _leaves(reference["expected"])
    try:
        value = json.loads(
            text, parse_constant=_reject_constant, object_pairs_hook=_unique_object
        )
    except (ValueError, TypeError):
        return {"json_validity": 0.0, "schema_compliance": 0.0, "field_accuracy": 0.0}
    actual = _leaves(value)
    # Missing and extra fields both count as errors; JSON types are significant.
    paths = set(expected) | set(actual)
    correct = sum(
        p in expected and p in actual and canonical(expected[p]) == canonical(actual[p])
        for p in paths
    )
    return {
        "json_validity": 1.0,
        "schema_compliance": float(validator.is_valid(value)),
        "field_accuracy": correct / len(paths) if paths else 1.0,
    }


def code_solution(text, reference, mode):
    fenced = re.fullmatch(r"\s*```(?:python)?\s*\n(.*?)\n```\s*", text, flags=re.S)
    if fenced:
        text = fenced.group(1)
    if mode == "solution" or (
        mode == "auto"
        and re.search(rf"(?m)^def\s+{re.escape(reference['entry_point'])}\s*\(", text)
    ):
        return text
    return reference["prompt"] + text


# This driver executes ONLY in a CPU container. Candidate code executes in a
# subprocess; exit(0), a missing result and a timeout cannot masquerade as pass.
_CONTAINER_DRIVER = r"""
import contextlib, json, multiprocessing, os, sys
from pathlib import Path


def human_worker(code, ref, queue):
    try:
        ns = {}
        with open(os.devnull, 'w') as null:
            os.dup2(null.fileno(), 1)
            os.dup2(null.fileno(), 2)
            exec(code, ns)
            exec(ref['test'], ns)
            ns['check'](ns[ref['entry_point']])
        queue.put(True)
    except BaseException:
        queue.put(False)


def main():
    data = json.loads(Path('/work/input.json').read_text())
    code, ref = data['code'], data['reference']
    if data['task'] == 'humaneval_plus':
        from importlib.metadata import version
        if version('evalplus') != '0.3.1':
            raise RuntimeError('Expected evalplus==0.3.1')
        from evalplus.eval import untrusted_check
        from evalplus.gen.util import trusted_exec
        passed = True
        for key in ('base_input', 'plus_input'):
            inputs = ref[key]
            expected, timing = trusted_exec(ref['prompt'] + ref['canonical_solution'],
                inputs, ref['entry_point'], record_time=True)
            status, details = untrusted_check('humaneval', code, inputs,
                ref['entry_point'], expected, ref['atol'], timing, fast_check=True)
            passed = passed and status == 'pass'
        return 0 if passed else 10
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=human_worker, args=(code, ref, queue))
    process.start()
    process.join(data['timeout_s'])
    if process.is_alive():
        process.kill()
        process.join()
        return 10
    try:
        return 0 if queue.get(timeout=.5) is True else 10
    except Exception:
        return 10


if __name__ == '__main__':
    try:
        with open(os.devnull, 'w') as null:
            with contextlib.redirect_stdout(null), contextlib.redirect_stderr(null):
                status = main()
    except BaseException:
        status = 20
    sys.exit(status)
"""


def executable_score(task, text, reference, settings):
    if settings.image is None:
        raise RuntimeError(
            "Set a digest-pinned CPU scorer image; code is never executed on the host"
        )
    if task == "humaneval":
        # Compile the trusted harness without executing it on the host.
        import ast

        tree = ast.parse(reference["test"])
        if not any(
            isinstance(node, ast.FunctionDef) and node.name == "check"
            for node in tree.body
        ):
            raise ValueError("HumanEval reference is missing its check function")
    name = "llm-v4-score-" + uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix="llm-v4-score-") as directory:
        root = Path(directory)
        (root / "driver.py").write_text(_CONTAINER_DRIVER)
        (root / "input.json").write_text(
            canonical(
                {
                    "task": task,
                    "reference": reference,
                    "code": code_solution(text, reference, settings.code_mode),
                    "timeout_s": max(0.1, settings.timeout_s - 2),
                }
            )
        )
        command = [
            "docker",
            "run",
            "--rm",
            "--pull=never",
            "--name",
            name,
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--pids-limit=64",
            "--memory=1g",
            "--memory-swap=1g",
            "--cpus=1",
            "--user=65534:65534",
            "--ulimit",
            "nofile=128:128",
            "--ulimit",
            "fsize=1048576:1048576",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=128m,mode=1777",
            "--mount",
            f"type=bind,src={root},dst=/work,readonly",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            "HOME=/tmp",
            "--env",
            "OMP_NUM_THREADS=1",
            "--env",
            "OPENBLAS_NUM_THREADS=1",
            settings.image,
            "python",
            "-B",
            "/work/driver.py",
        ]
        # Permit the unprivileged container to read its inputs.
        root.chmod(0o755)
        for path in root.iterdir():
            path.chmod(0o444)
        timed_out = False
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=settings.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            # Killing the docker client alone does not kill its container.
            subprocess.run(
                ["docker", "rm", "-f", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
            )
        if timed_out:
            # A hard outer timeout cannot distinguish infrastructure/candidate hangs.
            raise RuntimeError("Executable scorer exceeded its container timeout")
        if result.returncode not in {0, 10}:
            raise RuntimeError(
                f"Executable scorer infrastructure failed: exit {result.returncode}"
            )
        return {"pass@1": float(result.returncode == 0)}


def score(task, completion, reference, settings):
    if completion.generation_error:
        return {
            "example_id": completion.example_id,
            "metrics": {},
            "generation_error": completion.generation_error,
            "scorer_error": None,
            "truncated": completion.truncated,
        }
    try:
        if task == "gpqa_diamond":
            metrics = gpqa_score(completion.text, reference)
        elif task in {"humaneval", "humaneval_plus"}:
            metrics = executable_score(task, completion.text, reference, settings)
        elif task == "extraction":
            metrics = extraction_score(completion.text, reference)
        elif task == "conversation":
            metrics = {}  # Reference matching is not a chat quality metric.
        else:
            raise ValueError(f"Unknown task: {task}")
        return {
            "example_id": completion.example_id,
            "metrics": metrics,
            "generation_error": None,
            "scorer_error": None,
            "truncated": completion.truncated,
        }
    except Exception as exc:
        return {
            "example_id": completion.example_id,
            "metrics": {},
            "generation_error": None,
            "scorer_error": f"{type(exc).__name__}: {exc}",
            "truncated": completion.truncated,
        }


def aggregate_scores(rows):
    """One sample per example; failures contribute zero to the full denominator."""
    if not rows or len({r["example_id"] for r in rows}) != len(rows):
        raise ValueError("Scoring requires one completion per unique example")
    numeric = {
        k for r in rows for k, v in r["metrics"].items() if isinstance(v, (float, int))
    }
    return {
        "quality": {
            k: sum(r["metrics"].get(k, 0.0) for r in rows) / len(rows) for k in numeric
        },
        "scored": sum(
            not r["scorer_error"] and not r["generation_error"] for r in rows
        ),
        "scorer_failures": sum(r["scorer_error"] is not None for r in rows),
        "generation_failures": sum(r["generation_error"] is not None for r in rows),
        "truncated": sum(r["truncated"] for r in rows),
        "total": len(rows),
    }
