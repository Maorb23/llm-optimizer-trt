# NVIDIA's release image already contains TensorRT-LLM and its CUDA stack.
ARG TRTLLM_IMAGE=nvcr.io/nvidia/tensorrt-llm/release:1.2.0
FROM ${TRTLLM_IMAGE}

WORKDIR /opt/llm-optimizer
COPY pyproject.toml README.md ./
COPY src ./src

# TensorRT libraries live outside the default loader path in this base image.
ENV LD_LIBRARY_PATH=/usr/local/tensorrt/lib:${LD_LIBRARY_PATH}

# Importing tensorrt_llm requires libcuda.so.1, which Docker builds do not
# receive. Check its installed distribution here; import it on a GPU later.
RUN python3 -m pip install --no-cache-dir ".[remote]" \
    && python3 -m pip uninstall -y nvidia-ml-py3 \
    && python3 -m pip install --no-cache-dir --force-reinstall nvidia-ml-py \
    && python3 -c "import importlib.metadata as m; import llm_optimizer.remote.worker; m.version('tensorrt-llm')" \
    && command -v trtllm-serve

ENV PYTHONUNBUFFERED=1
ENTRYPOINT []
