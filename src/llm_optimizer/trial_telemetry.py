"""Optional execution evidence, separate from benchmark metrics and engine policy."""

import json
import re
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path

from llm_optimizer.utils import InfinityToNullEncoder

_active = ContextVar("trial_telemetry", default=None)


def current_telemetry():
    return _active.get()


class TrialTelemetry:
    """One durable attempt. Unknown phases stay absent, including on partial runs."""

    def __init__(self, root, **identity):
        self.started = time.perf_counter()
        self.directory = Path(root) / uuid.uuid4().hex
        self.directory.mkdir(parents=True)
        self.path = self.directory / "execution.json"
        self.log_path = self.directory / "server.log"
        self.log_path.touch()
        self.running = {}
        self.data = {
            "schema_version": 1,
            "attempt_id": self.directory.name,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "timings_s": {},
            "phase_status": {},
            "startup_observations": [],
            "server_log": str(self.log_path.resolve()),
            "attempt_artifact": str(self.path.resolve()),
            **identity,
        }
        self.save()

    @contextmanager
    def activate(self):
        token = _active.set(self)
        try:
            yield self
        finally:
            _active.reset(token)

    def start(self, name):
        self.running[name] = time.perf_counter()
        self.data["phase_status"][name] = "running"

    def stop(self, name, complete=True):
        started = self.running.pop(name, None)
        if started is not None:
            timings = self.data["timings_s"]
            timings[name] = timings.get(name, 0.0) + time.perf_counter() - started
            self.data["phase_status"][name] = "complete" if complete else "partial"

    @contextmanager
    def phase(self, name):
        self.start(name)
        try:
            yield
        except BaseException:
            self.stop(name, complete=False)
            raise
        else:
            self.stop(name)

    @contextmanager
    def rest(self):
        try:
            with self.phase("rest"):
                yield
        except BaseException:
            self.data["rest_status"] = "interrupted"
            raise
        else:
            self.data["rest_status"] = "complete"
        finally:
            self.save()

    def save(self):
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.data, indent=2, cls=InfinityToNullEncoder))
        temporary.replace(self.path)

    def finish(self, error=None):
        for name in list(self.running):
            self.stop(name, complete=False)
        self.data["incomplete_phases"] = [
            name
            for name, status in self.data["phase_status"].items()
            if status == "partial"
        ]
        if error is None:
            status = "complete"
        elif isinstance(error, (KeyboardInterrupt, SystemExit)):
            status = "interrupted"
        elif any(base.__name__ == "TrialPruned" for base in type(error).__mro__):
            status = "pruned"
        else:
            status = "failed"
        self.data["status"] = (
            "cleanup_failed" if "cleanup_error" in self.data else status
        )
        if error is not None:
            self.data["error"] = f"{type(error).__name__}: {error}"
        timings = self.data["timings_s"]
        timings.setdefault("runner_wall", time.perf_counter() - self.started)
        timings["orchestration"] = max(
            0.0,
            timings["runner_wall"]
            - sum(
                timings.get(name, 0.0)
                for name in (
                    "launch_to_ready",
                    "client_preparation",
                    "warmup",
                    "benchmark",
                    "cleanup",
                )
            ),
        )
        self.data["finished_at"] = datetime.now(timezone.utc).isoformat()
        self.data["startup_observations"] = parse_startup_log(self.log_path)
        self.save()


# These are explicit durations emitted by engines, not inferred phase proportions.
# Keep repeated/rank-specific observations separate; never sum them as wall time.
_DURATION = re.compile(
    r"\b(?:took|takes|in|time[:=]?)\s+.*?"
    r"(\d+(?:\.\d+)?)\s*(ms|seconds?|secs?|s)\b",
    re.I,
)


def parse_startup_log(path):
    observations = []
    with Path(path).open(errors="replace") as stream:
        for number, line in enumerate(stream, 1):
            lowered = line.lower()
            if "graph" in lowered and ("captur" in lowered):
                phase = "graph_capture"
            elif "autotun" in lowered:
                phase = "autotuning"
            elif "compil" in lowered:
                phase = "compilation"
            elif "load" in lowered and ("weight" in lowered or "model" in lowered):
                phase = "weight_loading"
            else:
                continue
            match = _DURATION.search(line)
            if match:
                observations.append(
                    {
                        "phase": phase,
                        "seconds": float(match[1])
                        / (1000 if match[2].lower() == "ms" else 1),
                        "line_number": number,
                        "source": line.rstrip(),
                        "parser": "explicit-log-duration-v1",
                    }
                )
    return observations
