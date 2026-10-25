# Optional dependency repair for an approved vLLM/SGLang base. Follow this with
# Infiro's platform-worker.Dockerfile so its PYTHONPATH overlay is refreshed.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY docker/runtime_constraints.py /opt/worker-checks/runtime_constraints.py
RUN python3 /opt/worker-checks/runtime_constraints.py snapshot /opt/worker-checks/base \
    && python3 -m pip install --no-cache-dir -c /opt/worker-checks/base.txt \
       boto3 'optuna>=4,<5' 'wandb>=0.19,<0.20' 'optuna-integration[wandb]>=4,<5' \
    && python3 /opt/worker-checks/runtime_constraints.py verify /opt/worker-checks/base \
    && python3 -c "import boto3; import optuna; import wandb; import optuna_integration.wandb"
ENTRYPOINT []
