# NVIDIA's release image already contains TensorRT-LLM and its CUDA stack.
ARG TRTLLM_IMAGE=nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc26
FROM ${TRTLLM_IMAGE}

WORKDIR /opt/llm-optimizer
COPY pyproject.toml README.md ./
COPY src ./src
COPY docker/runtime_constraints.py /opt/worker-checks/runtime_constraints.py

# TensorRT libraries live outside the default loader path in this base image.
ENV LD_LIBRARY_PATH=/usr/local/tensorrt/lib:${LD_LIBRARY_PATH}

# Importing tensorrt_llm requires libcuda.so.1, which Docker builds do not
# receive. Check its installed distribution here; import it on a GPU later.
# Pin every existing distribution before adding worker dependencies. Resolution
# must fail rather than upgrade/downgrade NVIDIA's compatible runtime stack.
RUN python3 /opt/worker-checks/runtime_constraints.py snapshot /opt/worker-checks/base \
    && python3 -m pip install --no-cache-dir -c /opt/worker-checks/base.txt ".[remote,wandb]" \
    && python3 -m pip uninstall -y nvidia-ml-py3 \
    && python3 -m pip install --no-cache-dir --no-deps --force-reinstall -c /opt/worker-checks/base.txt nvidia-ml-py \
    && python3 /opt/worker-checks/runtime_constraints.py verify /opt/worker-checks/base \
    && python3 -c "import importlib.metadata as m; import wandb; import optuna_integration.wandb; import llm_optimizer.remote.worker; assert m.version('tensorrt-llm') == '1.3.0rc26'" \
    && command -v trtllm-serve

ENV PYTHONUNBUFFERED=1
ENTRYPOINT []
