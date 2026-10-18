# Nebius remote optimization

This opt-in interface runs one existing `llm-optimizer` search per framework on a
single Nebius GPU node. The local `llm-optimizer` CLI is unchanged. Install the
controller dependencies with `pip install -e '.[remote]'`. Each framework image
must contain this same package, its framework runtime, benchmark dependencies,
`boto3`, `optuna` for Optuna runs, and a working NVIDIA driver/CUDA stack.

## Configuration

Create `request.json`:

```json
{
  "model": "Qwen/Qwen3.5-4B",
  "revision": "exact-hugging-face-commit",
  "frameworks": ["vllm", "sglang"],
  "gpu_type": "gpu-h100-sxm",
  "gpu_count": 1,
  "preset": "1gpu-16vcpu-96gb",
  "images": {
    "vllm": "registry.example/vllm-optimizer:cuda-tag",
    "sglang": "registry.example/sglang-optimizer:cuda-tag"
  },
  "bucket": "optimizer-results",
  "prefix": "llm-optimizer",
  "client_args": ["dataset_name=random;random_input_len=512;random_output_len=128"],
  "server_args": {"vllm": ["max_num_seqs=16,32"], "sglang": ["max_running_requests=16,32"]},
  "search_strategy": "optuna",
  "optuna": {
    "trials": 20,
    "objective": "output_throughput",
    "sampler": "TPESampler",
    "sampler_kwargs": {},
    "pruner": "NopPruner",
    "pruner_kwargs": {},
    "seed": 7
  },
  "timeout_minutes": 180,
  "preemptible": false,
  "dry_run": true
}
```

Set `frameworks` to `["vllm"]` for one job. Use the supported framework names
from `llm-optimizer --help`. Search arguments have the same syntax as the local
CLI. Optional `speculative_heads` is metadata; provide the runtime's actual
speculative flags in each framework's `server_args` as well. Revisions are passed
through the framework's supported revision argument. A revision should be a
specific commit for reproducibility.
Use `optuna_by_framework` to override any Optuna setting for one framework,
for example `{"sglang":{"server_args":["max_running_requests=int(1,4)"]}}`.

Run the persistent controller on a non-GPU host with a durable local filesystem:

```bash
cp .env.example .env
# Edit .env with your Nebius project, S3 credentials, and database path.
set -a
. ./.env
set +a
mkdir -p .local
python -m llm_optimizer.remote --database "$LLM_OPTIMIZER_CONTROLLER_DB" serve --interval 15
```

The Nebius CLI must be installed with an active profile on that host. The
[README setup](../README.md#run-a-nebius-gpu-optimization-job) shows service-account
profile creation. `.env` is ignored by Git and must be sourced in each controller
shell; the optimizer does not load it automatically. `NB_PROJECT_ID`
is used as `--parent-id`. The controller forwards AWS credentials, endpoint,
region, and `HF_TOKEN` to workers. Use a service account with the required job
permissions. Do not put credentials in `request.json`. The SQLite database must
survive controller restarts; run **one controller service per database**. S3 is
for artifacts and checkpoints, not the job queue.

For a diagnostic run, set `NEBIUS_DEBUG_SSH_PUBLIC_KEY` to an existing public
key file before starting the controller, for example
`export NEBIUS_DEBUG_SSH_PUBLIC_KEY="$HOME/.ssh/id_ed25519.pub"`. The controller
then adds the key to the Nebius job and runs the worker through a `bash -lc`
wrapper. If the worker fails, the container prints its exit status and stays
alive for one hour so you can use `nebius ai job ssh JOB_ID -i
"$HOME/.ssh/id_ed25519" -s bash`. The job's configured timeout still applies;
stop the debug job once you have collected the failure. Unset the variable for
normal jobs. Use a new idempotency key for every new attempt.

In another shell, submit and query:

```bash
python -m llm_optimizer.remote --database "$LLM_OPTIMIZER_CONTROLLER_DB" submit request.json --idempotency-key my-run-001
python -m llm_optimizer.remote --database "$LLM_OPTIMIZER_CONTROLLER_DB" status RUN_ID
python -m llm_optimizer.remote --database "$LLM_OPTIMIZER_CONTROLLER_DB" results RUN_ID
python -m llm_optimizer.remote --database "$LLM_OPTIMIZER_CONTROLLER_DB" cancel RUN_ID --framework vllm
```

Omit `--framework` to cancel the parent run. The commands print JSON and are the
four operations for the future Orchestrator integration. `submit` returns a
`run_id`; `status` includes each framework's state, Nebius job ID, and latest
checkpointed trial count; `results`
includes partial manifests and aggregate best configurations. The controller
only marks results comparable when model, revision, GPU, workload, objective,
and constraints match. It does not rank incomparable results.

By default `dry_run` is true. A dry run creates state and job plans but **does
not call Nebius or upload S3 artifacts**. Change it to `false` only when ready
to incur GPU costs. The worker verifies visible GPUs and the runtime package
before invoking the existing optimizer.
Set `preemptible` to `true` for a preemptible Nebius job. Nebius can stop it
at any time; completed Optuna trials are checkpointed to S3.

Artifacts are under
`s3://<bucket>/<prefix>/<run_id>/<framework>/`. Retrieve them with
`aws s3 sync s3://<bucket>/<prefix>/<run_id>/ ./downloaded-run/ --endpoint-url "$AWS_ENDPOINT_URL_S3"`.
Each framework prefix has `manifest.json`, `results.json`, `results.jsonl`,
`results.workload.json`, `execution.log`, and, for local SQLite Optuna studies,
`checkpoint.zip` containing the study DB and JSONL records. The manifest has
artifact URIs, SHA-256 digests, runtime versions, the search configuration,
best parameters, and failure details. The original result JSON contains every
benchmark measurement and trial outcome. Its `cmd` entries and framework
parameters provide launch instructions for the best configuration.
The parent controller writes `aggregate.json` under the run prefix once all
framework jobs reach terminal states.

An interrupted job's completed Optuna trials can be resumed from the checkpoint
by starting the same worker payload after reconciling the original Nebius job.
Jobs have deterministic names. The controller searches Nebius by name after an
uncertain submission and resumes tracking a found job. It never automatically
retries a submission that remains `submission_unknown`, preventing duplicate
GPU jobs and simultaneous study writers. Operators must inspect Nebius before
any manual retry. `serve` prints submission failures to its terminal or
redirected log; the controller stores a redacted error without CLI arguments
or credentials.
External Optuna RDB storage uses the existing `optuna.storage` option; provide a
durable URL reachable by the worker. The controller does not create that database
or coordinate multiple writers to a study. Explicit study names are suffixed
with the run ID and framework. For managed SQLite storage, omit `optuna.storage`;
custom SQLite URLs are rejected because they cannot use the managed checkpoint.
Checkpoint uploads occur about every
30 seconds, so work since the last checkpoint can be lost on abrupt termination.

Framework jobs are independent. The controller does not share observations or
reallocate budgets between them; no measured benefit currently justifies that
complexity. Aggregate results retain successful jobs when another framework
fails. Nebius CLI behavior and container compatibility require a live smoke test
in an approved account; automated tests use mocks and do not launch GPUs.

## Optional W&B tracking

Install `.[wandb]` for local Optuna runs, or use the `wandb-1` overlay images.
Tracking is enabled only when `WANDB_PROJECT` is non-empty. The controller safely
forwards `WANDB_API_KEY`, `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_RUN_NAME`,
`WANDB_TAGS` (comma-separated), and `WANDB_MODE`. Never put the API key in a
request JSON or command-line argument.

Each framework job uses a deterministic W&B run ID derived from the durable
controller run ID and framework, so a restored S3/SQLite checkpoint resumes the
same W&B run. Trial state, objective, parameters, numeric benchmark metrics,
framework, model, GPU type, controller run ID, and study name are recorded. The
final result JSON, manifest, JSONL trial log, and execution log are uploaded as
a W&B artifact. W&B initialization, logging, upload, and shutdown errors are
warnings only and never change optimization or checkpoint status.

## Live Optuna pruning

Use a scalar `objective` (or a one-element `objectives` list) with a relative
pruner such as `MedianPruner`, `PercentilePruner`, `SuccessiveHalvingPruner`,
or `HyperbandPruner`. Multiobjective and constrained studies require `NopPruner`.
GPU `ThresholdPruner` is rejected pending independent workload calibration.

```json
{"optuna": {"trials": 30, "objective": "p95_e2e_latency_ms",
 "sampler": "TPESampler", "pruner": "MedianPruner",
 "pruner_kwargs": {"n_startup_trials": 10, "n_min_trials": 3},
 "min_completed_requests": 64}}
```

One continuous post-warmup benchmark reports cumulative metrics at 25%, 50%,
and 100% of the request budget. The minimum completed-request setting delays
checkpoints; it does not guarantee tail-statistic precision. Optuna steps are
completed-request counts. No pruning occurs at the final checkpoint. Request
failures produce `FAIL`; statistical early stops produce `PRUNED`. Pending
client tasks are cancelled and the managed server is cleaned up before the next
trial. Pruned configurations count toward the attempted configuration budget,
including after resume, and are not written as completed benchmark candidates.
Study summaries expose trial states, allocation GPU seconds (wall-time proxy),
and an explicit unvalidated status. Independently validate shortlisted configs
at the full request budget before deployment. Rebuild worker images with this
package version before submitting jobs.
