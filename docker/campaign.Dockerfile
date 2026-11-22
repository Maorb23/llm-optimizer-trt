ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG FRAMEWORK
ARG FRAMEWORK_VERSION

WORKDIR /opt/llm-optimizer
COPY docker/runtime_constraints.py /opt/worker-checks/runtime_constraints.py
COPY pyproject.toml README.md ./
COPY src ./src

ENV PYTHONPATH=/opt/llm-optimizer/src:${PYTHONPATH} \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH=/usr/local/tensorrt/lib:${LD_LIBRARY_PATH}

# Match the selected serving runtime and preserve its Python package versions.
# SGLang's base ships protobuf 7, which W&B 0.19 does not support.
RUN python3 -c 'import sys; from importlib.metadata import version; assert sys.version_info >= (3, 11), "v4 requires Python 3.11+"; package, expected = sys.argv[1:]; actual = version({"vllm": "vllm", "sglang": "sglang", "trtllm": "tensorrt-llm"}[package]); assert actual == expected, (package, expected, actual)' "$FRAMEWORK" "$FRAMEWORK_VERSION" \
    && python3 /opt/worker-checks/runtime_constraints.py snapshot /opt/worker-checks/base \
    && case "$FRAMEWORK" in \
         sglang) wandb_requirement='wandb==0.29.0' ;; \
         *) wandb_requirement='wandb>=0.19,<0.20' ;; \
       esac \
    && python3 -m pip install --no-cache-dir -c /opt/worker-checks/base.txt \
         '.[remote,dataset,notebook,campaign]' 'optuna-integration[wandb]>=4,<5' "$wandb_requirement" \
    && if python3 -m pip show nvidia-ml-py3 >/dev/null 2>&1; then \
         python3 -m pip uninstall -y nvidia-ml-py3 \
         && python3 -m pip install --no-cache-dir --no-deps --force-reinstall -c /opt/worker-checks/base.txt 'nvidia-ml-py>=13.610.43'; \
       fi \
    && python3 /opt/worker-checks/runtime_constraints.py verify /opt/worker-checks/base \
    && python3 -c 'import datasets, jsonschema, nbformat; import pynvml; import optuna_integration.wandb; import llm_optimizer.remote.worker; import llm_optimizer.experiment.v4.campaign'

ENTRYPOINT []
