# Use a versioned SGLang release tag; the upstream latest tag changes over time.
ARG SGLANG_IMAGE=lmsysorg/sglang:v0.5.18
FROM ${SGLANG_IMAGE}

WORKDIR /opt/llm-optimizer
COPY pyproject.toml README.md ./
COPY src ./src

# Install into the image's Python environment so the worker and SGLang server
# use the same dependencies and GPU runtime.
RUN python3 -m pip install --no-cache-dir ".[remote]" \
    && python3 -c "import sglang; import llm_optimizer.remote.worker"

ENV PYTHONUNBUFFERED=1
ENTRYPOINT []
