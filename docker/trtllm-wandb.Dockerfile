FROM maorb23/llm-optimizer-trtllm:workload-fix-1
WORKDIR /opt/llm-optimizer
COPY pyproject.toml README.md ./
COPY src ./src
RUN python3 -m pip install --no-cache-dir "wandb>=0.19,<0.20" "optuna-integration[wandb]>=4,<5" \
    && python3 -m pip install --no-cache-dir --no-deps --force-reinstall . \
    && python3 -c "import wandb; import optuna_integration.wandb; import llm_optimizer.remote.worker"
ENTRYPOINT []
