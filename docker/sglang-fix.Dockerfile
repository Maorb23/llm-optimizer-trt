FROM maorb23/llm-optimizer-sglang:v1
WORKDIR /opt/llm-optimizer
COPY pyproject.toml README.md ./
COPY src ./src
RUN python3 -m pip install --no-cache-dir --no-deps --force-reinstall .
