"""Finite-space Optuna helpers used by the benchmark command.

This module deliberately does not import Optuna at module import time.  Grid users do
not need the optional dependency installed.
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import typing as t
from dataclasses import dataclass
from urllib.parse import urlsplit

from llm_optimizer.args import ArgSet, BaseArg

THROUGHPUT_OBJECTIVES = {
    "request_throughput",
    "input_throughput",
    "output_throughput",
    "total_throughput",
}
LATENCY_OBJECTIVES = {
    f"{stat}_{metric}_ms"
    for stat in ("mean", "median", "p95", "p99")
    for metric in ("ttft", "itl", "tpot", "e2e_latency")
}
SUPPORTED_OBJECTIVES = THROUGHPUT_OBJECTIVES | LATENCY_OBJECTIVES


def require_optuna() -> t.Any:
    """Import Optuna with an actionable optional-dependency error."""
    try:
        import optuna  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on test environment
        raise RuntimeError(
            "Optuna search requires the optional dependency. "
            "Install it with: pip install -e '.[optuna]'"
        ) from exc
    return optuna


def objective_direction(metric: str) -> str:
    """Return the Optuna direction for a supported benchmark metric."""
    if metric in THROUGHPUT_OBJECTIVES:
        return "maximize"
    if metric in LATENCY_OBJECTIVES:
        return "minimize"
    choices = ", ".join(sorted(SUPPORTED_OBJECTIVES))
    raise ValueError(
        f"Unsupported Optuna objective '{metric}'. Choose one of: {choices}"
    )


def extract_objective(metrics: dict[str, t.Any], metric: str) -> float:
    """Extract a finite numeric objective value, rejecting bools and missing data."""
    objective_direction(metric)  # also validates the name
    value = metrics.get(metric)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Objective metric '{metric}' is missing or non-numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"Objective metric '{metric}' must be finite")
    return numeric


def constraint_metric_name(constraint: t.Any) -> str:
    """Map an SLO constraint to the benchmark result field name."""
    aliases = {
        "ttft": f"{constraint.stat_type}_ttft_ms",
        "itl": f"{constraint.stat_type}_itl_ms",
        "tpot": f"{constraint.stat_type}_tpot_ms",
        "e2e_latency": f"{constraint.stat_type}_e2e_latency_ms",
    }
    return aliases.get(
        constraint.metric, f"{constraint.stat_type}_{constraint.metric}"
    )


def constraint_violations(
    metrics: dict[str, t.Any], constraints: t.Sequence[t.Any]
) -> list[float]:
    """Return non-positive values for satisfied constraints, Optuna style."""
    violations: list[float] = []
    for constraint in constraints:
        value = metrics.get(constraint_metric_name(constraint))
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            violations.append(float("inf"))
            continue
        threshold = float(constraint.value)
        # Benchmark latency fields are milliseconds. SLO values retain their unit.
        if getattr(constraint, "unit", "ms") == "s":
            threshold *= 1000
        numeric = float(value)
        if not math.isfinite(numeric):
            violations.append(float("inf"))
            continue
        operator = constraint.operator
        if operator == "<":
            violation = numeric - threshold
            violations.append(
                math.ulp(threshold or 1.0) if violation == 0 else violation
            )
        elif operator == "<=":
            violations.append(numeric - threshold)
        elif operator == ">":
            violation = threshold - numeric
            violations.append(
                math.ulp(threshold or 1.0) if violation == 0 else violation
            )
        elif operator == ">=":
            violations.append(threshold - numeric)
        elif operator == "==":
            violations.append(abs(numeric - threshold))
        elif operator == "!=":
            violations.append(0.0 if numeric != threshold else 1.0)
        else:
            raise ValueError(f"Unsupported SLO operator '{operator}'")
    return violations


def satisfies_constraints(
    result_or_metrics: dict[str, t.Any], constraints: t.Sequence[t.Any]
) -> bool:
    """Evaluate every supported SLO operator against one benchmark result."""
    if not constraints:
        return True
    metrics = result_or_metrics.get("results", result_or_metrics)
    return all(value <= 0 for value in constraint_violations(metrics, constraints))


@dataclass(frozen=True)
class OptunaSearchSpace:
    """A finite product of ArgSets sampled through categorical indices."""

    arg_sets: tuple[ArgSet, ...]

    @property
    def cardinality(self) -> int:
        return math.prod(len(arg_set.values) for arg_set in self.arg_sets)

    @staticmethod
    def parameter_name(arg_set: ArgSet) -> str:
        scope = arg_set.scope.name.lower()
        name = (
            "*".join(arg_set.name)
            if isinstance(arg_set.name, tuple)
            else arg_set.name
        )
        return f"{scope}:{name}"

    def suggest(self, trial: t.Any) -> list[BaseArg]:
        selected: list[BaseArg] = []
        for arg_set in self.arg_sets:
            choices = arg_set.get_all_possible_arg_values()
            if len(choices) == 1:
                selected.append(choices[0])
                continue
            index = trial.suggest_categorical(
                self.parameter_name(arg_set), list(range(len(choices)))
            )
            selected.append(choices[index])
        return selected

    def canonical(self) -> list[dict[str, t.Any]]:
        return [
            {
                "scope": arg_set.scope.name.lower(),
                "name": list(arg_set.name)
                if isinstance(arg_set.name, tuple)
                else arg_set.name,
                "values": arg_set.values,
            }
            for arg_set in self.arg_sets
        ]


def canonical_signature(payload: dict[str, t.Any]) -> tuple[str, str]:
    """Return a stable SHA-256 signature and its canonical JSON representation."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest(), canonical


def default_storage_url(output_json: pathlib.Path) -> str:
    database = output_json.with_suffix(".optuna.db").resolve()
    return f"sqlite:///{database.as_posix()}"


def storage_kind(storage_url: str) -> str:
    """Return only the database scheme; never expose credentials."""
    scheme = urlsplit(storage_url).scheme.split("+")[0]
    return scheme or "unknown"
