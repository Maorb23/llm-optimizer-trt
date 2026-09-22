"""Finite-space Optuna helpers used by the benchmark command.

This module deliberately does not import Optuna at module import time.  Grid users do
not need the optional dependency installed.
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import re
import typing as t
from dataclasses import dataclass
from urllib.parse import urlsplit

from llm_optimizer.args import Arg, ArgConfig, ArgScope, ArgSet, BaseArg

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
    if metrics.get("benchmark_valid") is False or metrics.get("failure_rate", 0) != 0:
        raise ValueError("Benchmark is failed or incomplete; objective is invalid")
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
class IntRange:
    """An inclusive, stepped Optuna integer distribution."""

    scope: ArgScope
    name: str
    low: int
    high: int
    step: int = 1

    @property
    def cardinality(self) -> int:
        return (self.high - self.low) // self.step + 1


_INT_RANGE_RE = re.compile(
    r"^((?:--)?[A-Za-z][A-Za-z0-9_-]*)\s*=\s*int\(\s*"
    r"(-?\d+)\s*,\s*(-?\d+)\s*"
    r"(?:,\s*(?:step\s*=\s*)?(\d+)\s*)?\)$"
)


def parse_int_ranges(
    specifications: t.Sequence[str],
    scope: ArgScope,
    configs: t.Mapping[str, ArgConfig[t.Any]],
) -> tuple[IntRange, ...]:
    """Parse Optuna-only ``name=int(low,high[,step])`` definitions."""
    ranges: list[IntRange] = []
    seen: set[str] = set()
    for specification in specifications:
        for part in specification.split(";"):
            part = part.strip()
            if not part:
                continue
            match = _INT_RANGE_RE.fullmatch(part)
            if match is None:
                raise ValueError(
                    f"Invalid Optuna integer range '{part}'; expected "
                    "name=int(low,high[,step])"
                )
            raw_name = match.group(1)
            name = (
                raw_name[2:] if raw_name.startswith("--") else raw_name
            ).replace("-", "_")
            if name not in configs or configs[name].arg_type is not int:
                raise ValueError(
                    f"Optuna integer range '{name}' needs a known integer argument"
                )
            if name in seen:
                raise ValueError(f"Duplicate Optuna integer range '{name}'")
            low, high = int(match.group(2)), int(match.group(3))
            step = int(match.group(4) or 1)
            if step < 1 or high < low or (high - low) % step:
                raise ValueError(
                    f"Optuna integer range '{name}' needs low <= high, a positive "
                    "step, and a high bound reachable by that step"
                )
            ranges.append(IntRange(scope, name, low, high, step))
            seen.add(name)
    return tuple(ranges)


@dataclass(frozen=True)
class OptunaSearchSpace:
    """A product of categorical ArgSets and Optuna integer distributions."""

    arg_sets: tuple[ArgSet, ...]
    int_ranges: tuple[IntRange, ...] = ()

    def __post_init__(self) -> None:
        names: set[tuple[ArgScope, str]] = set()
        for arg_set in self.arg_sets:
            for name in (
                arg_set.name if isinstance(arg_set.name, tuple) else (arg_set.name,)
            ):
                key = (arg_set.scope, name)
                if key in names:
                    raise ValueError(
                        f"Duplicate argument '{arg_set.scope.name.lower()}:{name}'"
                    )
                names.add(key)
        for int_range in self.int_ranges:
            key = (int_range.scope, int_range.name)
            if key in names:
                scoped_name = f"{int_range.scope.name.lower()}:{int_range.name}"
                raise ValueError(
                    f"Optuna integer range '{scoped_name}' "
                    "also appears in fixed or categorical arguments"
                )
            names.add(key)

    @property
    def cardinality(self) -> int:
        return math.prod(len(arg_set.values) for arg_set in self.arg_sets) * math.prod(
            int_range.cardinality for int_range in self.int_ranges
        )

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
        for int_range in self.int_ranges:
            value = trial.suggest_int(
                f"{int_range.scope.name.lower()}:{int_range.name}",
                int_range.low,
                int_range.high,
                step=int_range.step,
            )
            selected.append(
                Arg[int](scope=int_range.scope, name=int_range.name, value=value)
            )
        return selected

    def canonical(self) -> list[dict[str, t.Any]]:
        categorical = [
            {
                "scope": arg_set.scope.name.lower(),
                "name": list(arg_set.name)
                if isinstance(arg_set.name, tuple)
                else arg_set.name,
                "values": arg_set.values,
            }
            for arg_set in self.arg_sets
        ]
        if not self.int_ranges:
            return categorical
        return categorical + [
            {
                "scope": int_range.scope.name.lower(),
                "name": int_range.name,
                "distribution": "int",
                "low": int_range.low,
                "high": int_range.high,
                "step": int_range.step,
            }
            for int_range in self.int_ranges
        ]

    def grid_search_space(self) -> dict[str, list[t.Any]]:
        """Return the concrete Optuna GridSampler domain for this space."""
        values: dict[str, list[t.Any]] = {}
        for arg_set in self.arg_sets:
            values[self.parameter_name(arg_set)] = list(range(len(arg_set.values)))
        for int_range in self.int_ranges:
            values[f"{int_range.scope.name.lower()}:{int_range.name}"] = list(
                range(int_range.low, int_range.high + 1, int_range.step)
            )
        return values


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
