import typing as t

import llm_optimizer.predefined.max as max
import llm_optimizer.predefined.sglang as sglang
import llm_optimizer.predefined.trtllm as trtllm
import llm_optimizer.predefined.vllm as vllm
from llm_optimizer.args import BaseArg, get_all_cmd_args, get_all_kv_pairs
from llm_optimizer.predefined.client import CLIENT_CONFIGS as CLIENT_CONFIGS
from llm_optimizer.predefined.framework import FrameworkSpec

FRAMEWORKS = {
    "sglang": FrameworkSpec(
        server_configs=sglang.SERVER_CONFIGS,
        parameter_mapping=sglang.PARAMETER_MAPPING,
        server_command=(
            "python3 -m sglang.launch_server --model-path {model} "
            "--host {host} --port {port}"
        ),
        default_port=30000,
    ),
    "vllm": FrameworkSpec(
        server_configs=vllm.SERVER_CONFIGS,
        parameter_mapping=vllm.PARAMETER_MAPPING,
        server_command="vllm serve {model} --host {host} --port {port}",
        default_port=8000,
    ),
    "max": FrameworkSpec(
        server_configs=max.SERVER_CONFIGS,
        parameter_mapping=max.PARAMETER_MAPPING,
        server_command="max serve --model-path {model} --port {port}",
        default_port=8000,
    ),
    "trtllm": FrameworkSpec(
        server_configs=trtllm.SERVER_CONFIGS,
        parameter_mapping=trtllm.PARAMETER_MAPPING,
        server_command="trtllm-serve {model} --host {host} --port {port}",
        default_port=8000,
        cli_option_separator="_",
    ),
}

SERVER_CONFIGS = {name: spec.server_configs for name, spec in FRAMEWORKS.items()}

PARAMETER_MAPPINGS = {name: spec.parameter_mapping for name, spec in FRAMEWORKS.items()}

SEVER_CMD_TMPL = {name: spec.server_command for name, spec in FRAMEWORKS.items()}


def get_framework_spec(framework: str) -> FrameworkSpec:
    return FRAMEWORKS[framework]


def get_server_cmd_args(
    framework: t.Optional[str], args: list[BaseArg]
) -> list[str]:
    """Render native CLI options, including TRT-LLM's underscore spelling."""
    if not framework or FRAMEWORKS[framework].cli_option_separator == "-":
        return get_all_cmd_args(args)

    rendered = []
    for name, value in get_all_kv_pairs(args):
        option = "--" + name.replace("-", "_")
        if isinstance(value, bool):
            if value:
                rendered.append(option)
            elif framework == "trtllm" and name == "telemetry":
                rendered.append("--no-telemetry")
        else:
            rendered.append(f"{option}={value}")
    return rendered
