"""Small shared description of a managed inference framework."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional

from llm_optimizer.args import ConfigsDict


@dataclass(frozen=True)
class FrameworkSpec:
    """Lifecycle details needed by the benchmark runner.

    Keeping deployment details here leaves room for a future deployment mode
    (for example, Dynamo) without coupling it to the inference engine's native
    arguments.
    """

    server_configs: ConfigsDict
    parameter_mapping: Mapping[str, Optional[str]]
    server_command: str
    default_port: int
    ready_endpoint: str = "/health"
    benchmark_backend: str = "vllm"
    cli_option_separator: str = "-"
