# Build with a pinned VLLM_IMAGE tag for reproducible worker environments.
ARG VLLM_IMAGE=vllm/vllm-openai:latest
FROM ${VLLM_IMAGE}

WORKDIR /opt/llm-optimizer

# The remote worker starts this package itself; its request payload is supplied
# by Nebius at job creation rather than baked into the image.
COPY pyproject.toml README.md ./
COPY src ./src

# vLLM's image exposes Python as `python3`; add `python` because the Nebius
# worker command uses that conventional executable name.
RUN python3 -m pip install --no-cache-dir ".[remote]" \
    && ln -s "$(command -v python3)" /usr/local/bin/python

ENV PYTHONUNBUFFERED=1
ENTRYPOINT []
