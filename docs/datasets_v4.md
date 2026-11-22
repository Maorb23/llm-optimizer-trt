# Phase 1 dataset-aware notebooks

Install the notebook and dataset extras from the repository root.

```sh
python -m pip install -e '.[dataset,notebook,dev,wandb]' jsonschema nbformat
```

Open the notebook for your engine. Running all cells as shipped prepares an
offline extraction fixture, constructs the engine menus and checks CPU
preflight. GPU setup, studies, validation and W&B are disabled.

- [vLLM](phase1_vllm__datasets_v4.ipynb)
- [TensorRT-LLM](phase1_trtllm_datasets_v4.ipynb)
- [SGLang](phase1_sglang_datasets_v4.ipynb)

## Qwen3.8-27B and GPQA Diamond

Use the following controls in the **vLLM** or **SGLang** notebook as a starting
campaign. Repository API metadata and the model's pinned chat template were
checked on 2026-11-20. The model is public and its template defaults to thinking
on. GPQA requires accepting its access terms on Hugging Face and authenticating
with a read token for the approved account. An unauthenticated request for the
pinned CSV returned HTTP 401. Keep prompt, reference, and response artifacts
private under the dataset terms. [Pinned Qwen model and template](https://huggingface.co/Qwen/Qwen3.8-27B/tree/37561816e66b5b9153b1b64c10968c18c1e6f4a0)
[GPQA dataset and access terms](https://huggingface.co/datasets/Idavidrein/gpqa/tree/83022cefff930aea54f654c0b282e74b9eeda5c6)

```python
MODEL = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "37561816e66b5b9153b1b64c10968c18c1e6f4a0"
DATA_SOURCE = "hf"
SOURCE_FORMAT = "gpqa_diamond"
DATASET_ID = "Idavidrein/gpqa"
DATASET_REVISION = "83022cefff930aea54f654c0b282e74b9eeda5c6"
DATASET_CONFIG = "gpqa_diamond"
DATASET_SPLIT = "train"
CONCURRENCIES = [1, 8, 16]
REQUESTS_BY_CONCURRENCY = {1: 32, 8: 62, 16: 125}
PRUNING_MIN_COMPLETED_REQUESTS = {1: 12, 8: 24, 16: 50}
DATASET_REPEAT_REQUESTS = 125
HELD_OUT_COUNT = 32
SAMPLING_SEED = 2026
TOKEN_BUCKETS_BY_CONCURRENCY = {}
GENERATION = Generation(
    output_tokens=2048, reasoning_tokens=None, thinking="off", eos="stop",
    temperature=0.0, top_p=1.0, seed=2026,
)
THINKING_TEMPLATE_VERIFIED = True  # pinned template handles enable_thinking
SCORE_QUALITY = True
QUALITY_THRESHOLDS = ()  # optionally (("accuracy", 0.60),)
WANDB_ENABLED = False
```

Set these in the initial controls before evaluating the dependent cells. Use
`hf auth login` or set `HF_TOKEN` in the VM environment. Use BF16 weights and
cache for the initial compatibility check by setting both entries in the
engine's `PRECISION_CHOICES` to `["bf16"]`. The existing phase B verification
cell must reflect the same precision choice and your actual verification.
Enable `RUN_REAL_GPU` and `RUN_MODEL_SETUP` to resolve the pinned checkpoint.
Run the preserved native audit and smoke cells for the installed engine and
confirm that the managed chat endpoint honors the template override and returns
streaming completion usage. Then run preparation and enable `RUN_STUDIES` for
the study cells you intend to execute. This example pins the Hub artifacts and
request semantics. GPU fitting, engine compatibility, and task accuracy were
unverified. The TRT wrapper supports `thinking="default"` only. Qwen's default
then enables thinking, so it is a distinct configuration.

## Workloads and evaluation

| Source format | Input and quality |
|---|---|
| `sharegpt`, `messages`, `alpaca` | Text conversations. All prior turns are preserved and a final assistant target is held separately. Quality is unavailable. |
| `gpqa_diamond` | Deterministically shuffled choices. The scorer reads the final answer letter. |
| `humaneval`, `humaneval_plus` | One generated solution per task, tested in an isolated CPU container for pass@1. HumanEval+ rows require the canonical solution and both frozen input suites. |
| `extraction` | Instruction, input text and public schema. Scores report JSON validity, Draft 2020-12 compliance, and typed field accuracy. |
| `legacy_random` via `DATA_SOURCE` | Original ShareGPT first-user fixed-length performance sampler. Defaults to 512 input and 128 output tokens with EOS ignored, matching v3. No task quality. |

For task data use `DATA_SOURCE="hf"` with a full Hub revision, or
`DATA_SOURCE="local"` with `DATASET_ID` pointing to a JSON or JSONL file.
Use the dedicated `LEGACY_*` controls for the historical workload. It uses
the pinned ShareGPT source by default and hashes the bytes of an optional
local source. It retains the original v3 request materialization and benchmark
backend.

Natural sampling selects unique, unmodified payloads. For a token-bucket arm
set `TOKEN_BUCKETS_BY_CONCURRENCY[c]` to `(lower, upper, count)` triples whose
counts sum to its request budget. Bounds are lower-inclusive and upper-exclusive.
Token counts use the phase snapshot's tokenizer and chat template. Held-out and
repeat subsets use natural sampling. Small task benchmarks require reducing
request budgets and pruning floors to fit their unique examples. Preparation
fails if a requested count or bucket cannot be filled.

The frozen manifests record dataset and model/tokenizer revisions, IDs,
payload and reference hashes, generation, sampling and scorer settings.
The measured client sends only chat messages and generation settings.
`output_tokens` caps **all** generated tokens, including reasoning. The separate
`reasoning_tokens` control is rejected by these wrappers. Explicit thinking
uses `chat_template_kwargs.enable_thinking` on verified vLLM/SGLang templates.
TRT task chat rejects explicit thinking and EOS-ignore overrides.
`SCORE_QUALITY=False` retains performance and completion accounting with task
scores disabled.

For executable evaluation, build a CPU scorer image and use its immutable ID.

```sh
docker build -f docs/Dockerfile.scorer_v4 -t llm-v4-scorer .
docker image inspect llm-v4-scorer --format '{{.Id}}'
```

Set `SCORER = ScorerSettings(image="sha256:…")`. The scorer uses a read-only
root and input mount, no network, an unprivileged user, and bounded CPU,
memory, PIDs and runtime. The image must be available before GPU studies.
HumanEval+ uses [EvalPlus 0.3.1](https://github.com/evalplus/evalplus/blob/v0.3.1/evalplus/eval/__init__.py).

## Study results

The v3 inference metrics, SLO constraints, checkpoints, and objectives still
drive each arm. The task scorer sees only the measured generations, after
telemetry and server cleanup. Live pruning sees performance alone. When
quality thresholds are enabled, final quality failures or generation/scorer
errors leave an otherwise admissible trial pruned, outside scalar and Pareto
winner selection. Those trials may have performance checkpoint values.
With thresholds empty, scoring does not change performance feasibility.
Failed requests count as zero in full-workload task scores. Truncation is
recorded separately and the emitted final answer is scored as received.

The campaign hash includes the complete evaluation identity and the inherited
v3 experiment settings. Restore the same controls and run root to resume.
The v3 study JSON, performance JSONL, attempt telemetry, SQLite and plots
remain available. v4 adds frozen workloads and per-attempt quality JSONL
linked by attempt ID. W&B is optional and logs study identity, quality,
performance and telemetry. Held-out evaluation has its own tracking run.

Enable `RUN_VALIDATION` for the original shortlist and performance repeats.
These repeats use tuning data with the original repeat seed progression.
For post-selection testing set `RUN_HELD_OUT=True` and `HELD_OUT_STUDY_NAME` to
the chosen arm. A Pareto arm also requires `HELD_OUT_PARETO_TRIAL` from its
eligible front. Held-out data is disjoint by ID and inference payload from
all tuning and repeat data. The selected configuration runs at its study
concurrency. Results go to `held-out-results.jsonl` outside optimization.

## v3 to v4 parity

| v3 feature | v4 location |
|---|---|
| Initial model, VM, campaign, sampler, W&B, budget controls | Initial configuration, with GPU/W&B opt-ins and new dataset controls |
| Base and extended menus, optional dimensions, conditional parameters, fixed settings and baseline rendering | Original search and precision cells, engine policy and managed serializer |
| Pinned HF model and speculative setup, TRT ModelOpt export, precision and compatibility, CLI audit and native smoke | Original setup/audit cells, with opt-in setup and smoke switches |
| A/B e2e and TTFT at c1, A/B throughput and e2e at loaded concurrency, B median-pruned throughput | Original independent cells and plots |
| Pareto latency/throughput, SLO constraints, pruning calibration and comparison | Original optional experiment cells with original objective/sampler/pruner wiring |
| Seeds, attempt budgets, GPU lock, SQLite identity/resume, retry and failure evidence, cleanup, telemetry, W&B | Same runner, lab and lifecycle, with scoped chat request binding for natural/task data |
| Saved responses, plots, shortlist and repeat statistics | Original response and validation cells, with quality JSONL and independent held-out evaluation added |

The v3 batch CLI already omitted notebook-specific settings. For unattended
notebook execution use a notebook executor with the same configured cells and
run root. Abrupt kernel termination still requires checking orphaned servers
before resuming. [Validation record](datasets_v4_validation.md)

## Build the v3 and v4 campaign images

Build one Linux AMD64 image per framework from the same source snapshot.
The recent collection recorded vLLM 0.33.0, SGLang 0.5.23, and TensorRT-LLM
1.4.0rc3. Choose the CUDA image digest from the environment you want to
reproduce. The build checks the installed framework version and constrains
pip to the versions already present in that base. Use a runtime image with
Python 3.11 or newer. Keep Optuna 4.9.0 consistent with the notebook extra.

To copy the build context to the build machine, run from the checkout with
`VM_IP` and `VM_PORT` set:

```bash
tar --exclude='__pycache__' --exclude='*.pyc' -czf - \
  pyproject.toml README.md .dockerignore src \
  docker/campaign.Dockerfile docker/runtime_constraints.py |
  ssh -p "$VM_PORT" root@"$VM_IP" \
    'mkdir -p /workspace/llm-optimizer-campaign && tar -xzf - -C /workspace/llm-optimizer-campaign'
```

On the build machine, set `VLLM_BASE`, `SGLANG_BASE`, and `TRTLLM_BASE` to
the selected `repository@sha256:…` references. Set `IMAGE_PREFIX` to your
registry path plus repository prefix, for example `maorb23/llm-optimizer`.

```bash
cd /workspace/llm-optimizer-campaign
set -euo pipefail
for framework in vllm sglang trtllm; do
  case "$framework" in
    vllm)   base="$VLLM_BASE";   version=0.33.0 ;;
    sglang) base="$SGLANG_BASE"; version=0.5.23 ;;
    trtllm) base="$TRTLLM_BASE"; version=1.4.0rc3 ;;
  esac
  [[ "$base" =~ @sha256:[a-f0-9]{64}$ ]]
  image="${IMAGE_PREFIX}-${framework}:campaign-v4-${version}"
  docker build --platform linux/amd64 -f docker/campaign.Dockerfile \
    --build-arg BASE_IMAGE="$base" --build-arg FRAMEWORK="$framework" \
    --build-arg FRAMEWORK_VERSION="$version" -t "$image" .
done
```

Mount the configured notebook checkout, Hugging Face cache, and persistent run
directory when running campaigns. Run the v3 notebook for the historical
workload or the v4 notebook for dataset selection, scoring, and held-out
evaluation. Model and dataset revisions, sampling, generation, and tuning
choices live in the campaign settings. Nebius can use the same Python
campaign logic through the headless worker profile described below.
HumanEval and HumanEval+ scoring also require a Docker CLI and a
reachable daemon with the digest-pinned scorer available. The GPU image
contains the Python campaign dependencies.

For DSpark, set `HF_SETUP["speculative_algorithm"]` to `"DSPARK"` on vLLM
and SGLang, or `"DSpark"` on TensorRT-LLM. Provide a compatible draft ID and
revision and set the engine-specific draft length. On vLLM,
`num_speculative_tokens` controls the draft width. On SGLang, `block_size`
matches the checkpoint and the wrapper adds one token to the verification
window. TensorRT-LLM uses `max_draft_len` and validates the installed native
config and draft architecture.

Before pushing each tag, use the selected GPU to run the engine's native
audit and smoke cells with the intended target and draft, followed by a small
v4 study on the intended dataset. SGLang 0.5.23 includes DSpark for supported
models, and its upstream DSpark model implementation lacks
`Gemma4DSparkModel`. Treat Gemma 4 DSpark as unverified on that image.
Use each engine's policy and native audit to choose its valid parameter menu.
[vLLM speculative config](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/config/speculative.py)
[SGLang DSpark implementation](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/models/dspark.py)
[TensorRT-LLM native config](https://github.com/NVIDIA/TensorRT-LLM/blob/v1.4.0rc3/tensorrt_llm/llmapi/llm_args.py)

### Headless campaign profiles for worker jobs

The optimizer worker now accepts an optional `campaign_v4` request field
containing a profile name. Its existing legacy request path is the default.
The headless path reads `<name>.json` from the deployment-owned directory
`LLM_OPTIMIZER_CAMPAIGN_PROFILES`. Deploy only operator-reviewed profiles.
Use private artifact storage because the campaign archive contains frozen
workloads and can contain responses.

The profile is a JSON object keyed by framework. Each selected framework
entry contains:

- `settings`: the same settings supplied to the notebook runner, including
  Phase A and B preflight evidence and the intended study budgets.
- `runner_options`: the notebook runner's `base_spaces`,
  `expanded_choices`, `extra_spaces`, and optional policy settings such as
  `precision_choices`, `telemetry_options`, and
  `pruning_min_completed_requests`.
- `model_setup`: the notebook's Hugging Face model setup arguments, with
  `target_model_id` and full `target_revision`. The worker supplies the
  framework. Optional draft settings use the same Python setup helper.
- `workload`: the notebook's `CampaignWorkloads` arguments, with
  `source` set to `hf`. The worker supplies the framework, resolved models,
  revisions, and snapshots. Express `generation` and optional `scorer`
  as JSON objects, and provide `requests` by concurrency.
- `objective_groups`: a nonempty list of the arguments to
  `run_objective_group` as JSON objects. Optional `held_out` selects a
  completed `study` and optionally a `pareto_trial`.

The worker validates that the request model matches the pinned profile. It
writes a manifest and uploads `campaign-v4.zip` under the existing artifact
prefix. Configure the profile directory in the job, or bake it into a derived
image using a build context containing only the profile JSON files:

```bash
docker build -f docker/campaign-profile.Dockerfile \
  --build-arg CAMPAIGN_IMAGE="$CAMPAIGN_IMAGE" \
  -t "$WORKER_IMAGE" "$PROFILES_DIR"
```

The hf_validator payload transport preserves the optional field when a
prepared request contains it. Its current preparation UI still produces
legacy workload requests. It needs a profile-selection and approval path,
plus v4 result rendering, before users can launch these campaigns from that UI.
