# Infiro worker image rebuild

Inspected against both working trees on 2026-10-25. These are operator commands;
none of the Docker builds, pushes, configuration switches, restarts, or GPU runs
below have been executed. Docker Desktop WSL integration is unavailable here.

## Which images to build

| Variant | Dockerfile to use | Build context | Runtime action |
| --- | --- | --- | --- |
| vLLM | `hf_validator/docker/platform-worker.Dockerfile` | A directory containing the current `src/llm_optimizer` as `llm_optimizer/` | Package overlay on the configured vLLM digest |
| SGLang | `hf_validator/docker/platform-worker.Dockerfile` | The same package directory | Package overlay on the configured SGLang digest |
| TRT-LLM | `llm-optimizer/docker/trtllm.Dockerfile` | Optimizer repository root, or the source snapshot below with the same layout | New worker image from NVIDIA `1.3.0rc26`; no compilation of TensorRT itself |

Infiro's configured images are:

```json
{
  "vllm": "cr.eu-north1.nebius.cloud/e00p24e8mwqab11v1e/vllm@sha256:6fee13c1255f849e3e67a732cd4987081e507a65ba4caeb52044c0765da6b9b7",
  "sglang": "cr.eu-north1.nebius.cloud/e00p24e8mwqab11v1e/sglang@sha256:f6b1fc8f9d4b6d1422200f0c0affc0d4951df624b4c1ff650f3704902b747001",
  "trtllm": "cr.eu-north1.nebius.cloud/e00p24e8mwqab11v1e/trtllm@sha256:127113a484eb46b7e978b0aa00c2d71942fb8abef0bbb7bebb0d539e4ddcbb85"
}
```

These are repository configuration, not registry/content verification. All three
map to `:workload1` aliases. `INFIRO_SERVING_IMAGES` and `INFIRO_IMAGE_ALIASES`
override the files and must be accounted for in the actual deployment environment.

The base Dockerfiles `vllm.Dockerfile` (upstream `latest`) and `sglang.Dockerfile`
(upstream `v0.5.18`) would select a different runtime lineage. Do not use them for
this package-only refresh. `sglang-fix.Dockerfile` uses a legacy `:v1` base; the
three `*-wandb.Dockerfile` files use `:workload-fix-1`. None identifies the current
Infiro digest. Building those unchanged would replace the selected runtime lineage.

TRT's Dockerfile now defaults to `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc26`,
includes remote and W&B extras, and snapshots every preinstalled distribution.
Pip uses those versions as constraints; a second check rejects changed or removed
base distributions. Conflicts stop the build for review. The intentional exceptions
are the optimizer and legacy `nvidia-ml-py3`; the Dockerfile retains the existing
NVML repair using `nvidia-ml-py`, preserving its base version when present. It also
retains `/usr/local/tensorrt/lib` in `LD_LIBRARY_PATH`. No TRT import is attempted
during the build because it needs the GPU driver library.

## Freeze the source and choose new tags

Use a host with working Docker Engine/Desktop integration, Buildx, enough disk for
three large GPU images, and existing registry authentication. Authenticate through
your normal credential helper; no credentials belong in these commands or images.
Run all snippets in the same Bash session. Adjust only the checkout paths if copied
to a different host. Ensure the image environment overrides match the service's
effective configuration; the shell cannot discover another process's overrides.

```bash
set -euo pipefail
export OPT=/mnt/c/Users/maorb/work/bento_extend/llm-optimizer
export INFIRO=/mnt/c/Users/maorb/work/hf_validator
export RELEASE_DIR="$(mktemp -d /tmp/infiro-images.XXXXXX)"
export BUILD_SRC="$RELEASE_DIR/source"
export OVERLAY_CONTEXT="$RELEASE_DIR/package"
# 12 characters: the longest repository + ':' + this tag is exactly 64.
export RELEASE="s$(date -u +%y%m%d)-$(python3 -c 'import secrets; print(secrets.token_hex(2))')"

python3 - <<'PY'
import json, os, pathlib, shlex, shutil
opt, infiro, out = map(pathlib.Path, (os.environ['OPT'], os.environ['INFIRO'], os.environ['RELEASE_DIR']))
src = pathlib.Path(os.environ['BUILD_SRC'])
src.mkdir()
for name in ('pyproject.toml', 'README.md', '.dockerignore'):
    shutil.copy2(opt / name, src / name)
for name in ('src', 'docker'):
    shutil.copytree(opt / name, src / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
overlay = pathlib.Path(os.environ['OVERLAY_CONTEXT'])
overlay.mkdir()
shutil.copytree(src / 'src/llm_optimizer', overlay / 'llm_optimizer')
shutil.copy2(infiro / 'docker/platform-worker.Dockerfile', out / 'platform-worker.Dockerfile')
images = json.loads(os.environ.get('INFIRO_SERVING_IMAGES') or (infiro / 'config/platform-images.json').read_text())
aliases = json.loads(os.environ.get('INFIRO_IMAGE_ALIASES') or (infiro / 'config/platform-image-aliases.json').read_text())
(out / 'effective-images.before.json').write_text(json.dumps(images, indent=2) + '\n')
(out / 'effective-aliases.before.json').write_text(json.dumps(aliases, indent=2) + '\n')
for name in ('platform-images.json', 'platform-image-aliases.json'):
    shutil.copy2(infiro / 'config' / name, out / (name + '.before'))
lines = []
for f in ('vllm', 'sglang', 'trtllm'):
    base = images[f]
    assert '@sha256:' in base, (f, 'requires an immutable configured base')
    tag = base.split('@')[0] + ':' + os.environ['RELEASE']
    assert len(tag) <= 64, tag
    lines += [f'export {f.upper()}_BASE={shlex.quote(base)}',
              f'export {f.upper()}_TAG={shlex.quote(tag)}']
(out / 'images.env').write_text('\n'.join(lines) + '\n')
print('Source, prior configuration, and new image names saved in', out)
PY
. "$RELEASE_DIR/images.env"
docker version
docker buildx version
```

Keep this release directory on durable storage through rollback. The snapshot
includes current uncommitted and untracked package files, including `hf_workload.py`
and the draft-model fixes. A Git commit ID alone would not identify this checkout.
The source snapshot is also the expected content for the CPU checks below.

Verify that each new tag is absent in the registry using authenticated manifest
inspection or the registry UI. Only an explicit manifest-not-found/404 establishes
absence; authentication and network failures do not. Choose another release token
if any tag exists. Keep the old tags and digest references intact.

## Build locally; no push

First check the dependencies inherited by the two approved serving bases. These
commands may pull image layers but run without GPU devices or container networking:

```bash
for base in "$VLLM_BASE" "$SGLANG_BASE"; do
  docker run --rm --network none --entrypoint python3 "$base" -c \
    'import boto3, optuna, wandb, optuna_integration.wandb; import importlib.metadata as m; print({p:m.version(p) for p in ("boto3","optuna","wandb","optuna-integration")})'
done
export VLLM_BUILD_BASE="$VLLM_BASE"
export SGLANG_BUILD_BASE="$SGLANG_BASE"
```

If a base lacks these dependencies, use the optional constrained dependency layer
for that backend, then select it as the overlay base. For example, **only if needed**:

```bash
docker buildx build --platform linux/amd64 --load --provenance=false --sbom=false \
  -f "$BUILD_SRC/docker/worker-deps.Dockerfile" \
  --build-arg BASE_IMAGE="$VLLM_BASE" -t "infiro-vllm-deps:$RELEASE" "$BUILD_SRC"
export VLLM_BUILD_BASE="infiro-vllm-deps:$RELEASE"
# Equivalent optional SGLang repair:
docker buildx build --platform linux/amd64 --load --provenance=false --sbom=false \
  -f "$BUILD_SRC/docker/worker-deps.Dockerfile" \
  --build-arg BASE_IMAGE="$SGLANG_BASE" -t "infiro-sglang-deps:$RELEASE" "$BUILD_SRC"
export SGLANG_BUILD_BASE="infiro-sglang-deps:$RELEASE"
```

This layer constrains existing runtime versions and fails if incompatible; it does
not authorize replacing the runtime. Always apply the package overlay afterward:
installing into site-packages alone could leave the old `/opt/infiro-worker`
package winning import resolution. Inspect failures instead of loosening pins.

```bash
digest_of() {
  docker buildx imagetools inspect "$1" | awk '$1 == "Digest:" {print $2}'
}
export TRT_UPSTREAM=nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc26
export TRT_UPSTREAM_DIGEST="$(digest_of "$TRT_UPSTREAM")"
[[ "$TRT_UPSTREAM_DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]]
export TRT_PINNED="nvcr.io/nvidia/tensorrt-llm/release@$TRT_UPSTREAM_DIGEST"
printf '%s\n' "$TRT_PINNED" > "$RELEASE_DIR/trt-upstream.txt"

docker buildx build --platform linux/amd64 --load --provenance=false --sbom=false \
  -f "$RELEASE_DIR/platform-worker.Dockerfile" \
  --build-arg BASE_IMAGE="$VLLM_BUILD_BASE" -t "$VLLM_TAG" "$OVERLAY_CONTEXT"

docker buildx build --platform linux/amd64 --load --provenance=false --sbom=false \
  -f "$RELEASE_DIR/platform-worker.Dockerfile" \
  --build-arg BASE_IMAGE="$SGLANG_BUILD_BASE" -t "$SGLANG_TAG" "$OVERLAY_CONTEXT"

docker buildx build --platform linux/amd64 --load --provenance=false --sbom=false \
  -f "$BUILD_SRC/docker/trtllm.Dockerfile" \
  --build-arg TRTLLM_IMAGE="$TRT_PINNED" -t "$TRTLLM_TAG" "$BUILD_SRC"
```

The two overlay Dockerfiles only copy the package and prepend `PYTHONPATH`.
TRT's new image installs into `/opt/llm-optimizer`/site-packages, with no inherited
Infiro package overlay. Do not build `trtllm-wandb.Dockerfile` afterward: its
hardcoded `workload-fix-1` base would select the old runtime again.

## CPU-only verification

Run against each newly built image, with no GPU flag, no network, and no secrets:

```bash
for framework in vllm sglang trtllm; do
  variable="${framework^^}_TAG"
  docker run --rm --network none --entrypoint python3 \
    --mount "type=bind,src=$BUILD_SRC/src/llm_optimizer,dst=/expected,readonly" \
    --mount "type=bind,src=$BUILD_SRC/docker/check_worker_cpu.py,dst=/check.py,readonly" \
    "${!variable}" /check.py "$framework" --expected-source /expected
done

docker run --rm --network none --entrypoint python3 "$TRTLLM_TAG" \
  /opt/worker-checks/runtime_constraints.py verify /opt/worker-checks/base
docker run --rm --network none --entrypoint sh "$TRTLLM_TAG" -c 'command -v trtllm-serve'
```

The check prints package versions and the imported optimizer path; compares every
Python/JSON package file with the frozen source; imports W&B and its Optuna
integration without initializing a run; tests synthetic empty-choice, usage,
finish and DONE events; verifies a frozen workload and rejects tampering; rejects
99/100 scoring; checks failure-summary/telemetry diagnostics and sanitized
`attempts.zip` creation without a result row. S3 is replaced with a local fake.
TRT's distribution version must be exactly `1.3.0rc26`. Serving runtime versions
are checked through metadata, without importing `tensorrt_llm` or loading models.
These checks must pass in all three containers before pushing.

Optionally record `python3 -m pip check` output for each image. TRT intentionally
removes `nvidia-ml-py3` in favor of `nvidia-ml-py`; the optimizer's current metadata
still declares the former, so that specific unmet-distribution report is expected.
It does not establish a broken `pynvml` import. Investigate any other new conflicts
against the untouched upstream/base image before rollout. The constraints snapshot
also records Python packages, not every shared library or native ABI.

Host regression checks, using the existing CPU environment in this workspace:

```bash
cd "$OPT"
PYTHONPATH=src /home/maorb/.venvs/bento_extend/bin/python -m pytest -q \
  tests/test_runtime_constraints.py tests/test_worker_cpu_check.py \
  tests/test_completion_stream.py tests/test_failure_diagnostics.py \
  tests/test_hf_workload.py tests/test_wandb_tracking.py \
  tests/test_remote.py::test_checkpoint_uploads_failed_attempt_evidence_without_metric_rows
```

GPU-only verification is separate: native framework import/CLI startup, target and
draft loading, CUDA kernels, generation correctness, actual request accounting,
throughput/latency, and serving shutdown. W&B authentication/connectivity and real
S3 uploads also need later authorized network validation. Passing CPU checks cannot
establish any of those. No GPU command is run by this guide's build/check steps.

## Push separately, then record and verify registry digests

Only after reviewing CPU results and reconfirming the new tags are absent:

```bash
docker push "$VLLM_TAG"
docker push "$SGLANG_TAG"
docker push "$TRTLLM_TAG"
```

These are new short release tags, used directly as Infiro's aliases. No second
registry retag/copy is necessary. Use registry manifest digests, not local image
IDs or config digests. Buildx documents the top-level `Digest:` field and raw
manifest inspection in its [inspect reference](https://docs.docker.com/reference/cli/docker/buildx/imagetools/inspect/).

```bash
export VLLM_DIGEST="$(digest_of "$VLLM_TAG")"
export SGLANG_DIGEST="$(digest_of "$SGLANG_TAG")"
export TRTLLM_DIGEST="$(digest_of "$TRTLLM_TAG")"

python3 - <<'PY'
import json, os, pathlib, re, subprocess
out = pathlib.Path(os.environ['RELEASE_DIR'])
images = {}
aliases = json.loads((out / 'effective-aliases.before.json').read_text())
for framework in ('vllm', 'sglang', 'trtllm'):
    tag = os.environ[framework.upper() + '_TAG']
    digest = os.environ[framework.upper() + '_DIGEST']
    assert re.fullmatch(r'sha256:[0-9a-f]{64}', digest), digest
    assert len(tag) <= 64
    ref = tag.rsplit(':', 1)[0] + '@' + digest
    def resolved(name):
        output = subprocess.check_output(['docker', 'buildx', 'imagetools', 'inspect', name], text=True)
        matches = re.findall(r'^Digest:\s+(sha256:[0-9a-f]{64})\s*$', output, re.M)
        assert len(matches) == 1, 'Unexpected inspect output'
        return matches[0]
    assert resolved(tag) == digest == resolved(ref), (framework, 'alias mismatch')
    images[framework] = ref
    aliases[ref] = tag
    print(framework, ref, 'alias:', tag, 'length:', len(tag))
(out / 'platform-images.next.json').write_text(json.dumps(images, indent=2) + '\n')
(out / 'platform-image-aliases.next.json').write_text(json.dumps(aliases, indent=2) + '\n')
PY
```

Retain these mappings and push output with the source snapshot. Do not mutate the
new aliases after publication. Infiro independently fetches the selected alias
manifest and hashes its raw bytes immediately before submitting a job. Its
registry access uses the user's jobs connection, which may differ from the Docker
operator's credentials. A mismatch/denied pull blocks submission. Verification
does not remove the race between submission-time validation and provider pull;
operator-controlled immutable tags are still required.

## Switch Infiro configuration and restart

Review both generated `*.next.json` files. Their exact changes are:

1. Replace each framework's value in `config/platform-images.json` with the new
   full `repository@sha256:...` generated above.
2. Add each new digest-to-short-tag pair to `config/platform-image-aliases.json`.
   All old entries are retained. Never replace the old `workload1` mappings.

For deployments using the checked-in files, these are the explicit switch commands
(not part of the build or push commands):

```bash
cp "$RELEASE_DIR/platform-image-aliases.next.json" "$INFIRO/config/platform-image-aliases.json"
cp "$RELEASE_DIR/platform-images.next.json" "$INFIRO/config/platform-images.json"
```

If environment overrides are used, update both in the process manager's existing
configuration instead; file changes will otherwise be ignored. The exact values
are the JSON contents of the corresponding `*.next.json` files. For a shell-launched
deployment, export only these non-secret settings before starting both processes:

```bash
export INFIRO_SERVING_IMAGES="$(cat "$RELEASE_DIR/platform-images.next.json")"
export INFIRO_IMAGE_ALIASES="$(cat "$RELEASE_DIR/platform-image-aliases.next.json")"
```

Pause preparations while switching files and restarting, so no subprocess captures
a mixed mapping. Restart both Django/Gunicorn and `manage.py platform_worker`,
including preparation-only workers, with the same configuration/environment and
current optimizer source. Queue subprocesses inherit startup settings. The repo
does not define deployment-specific web/queue unit names; use the actual supervisor
or stop the foreground processes gracefully and relaunch with the existing options.
The documented command forms, from the existing activated environment, are:

```bash
cd "$INFIRO"
export PYTHONPATH="$INFIRO/src:$OPT/src${PYTHONPATH:+:$PYTHONPATH}"
# In the web process's terminal/supervisor, retain existing environment and bind options:
(cd web && gunicorn infiro_site.wsgi:application --bind 127.0.0.1:8000 --workers 2 --access-logfile - --error-logfile -)
# In the queue process's terminal/supervisor, after the previous worker exits:
python web/manage.py platform_worker --prepare-only
```

Preparation-only mode cannot submit/recover GPU studies and uses the same lock as
the normal worker. Do not run both against one database. Restore normal mode only
when authorized operational processing is intended; a normal worker can process
already approved queued work. No frontend rebuild or database migration is needed
for an image configuration change.

If the separate managed optimizer controller is actually used, install this source
into its existing environment (or select this source on its PYTHONPATH) and restart
the documented `hf-model-watch-optimizer.service` in the coordinated rollout:

```bash
systemctl --user restart hf-model-watch-optimizer.service
```

That controller reconciles its existing queue; only restart when resuming that
processing is authorized. Do not start a second `serve`/`tick` or point it at another
database. Infiro's Django SDK adapter does not require a second standalone controller.

Confirm the new web and queue processes load the generated mappings; do not dump
their full environments or settings. Preparation pins W&B references separately;
retain the existing `INFIRO_WANDB` and credential-file configuration.

## New small validation preparation

Use a **new recommendation/selection** for a known compatible small target model,
even if selecting the same pinned model as before. `create_prepared` invalidates
older reviews on the same recommendation; using a fresh recommendation preserves
existing approved requests and their ability to be processed. Give the new study
a distinct name including the release token. Do not edit saved request JSON,
approval hashes, study databases, or prior image references.

In Configure choose 2 trials per framework, 16 prompts, 128 input tokens, 32 output
tokens, seed 7, concurrency 1, one GPU, and the existing compatible GPU/preset. Use
`speculative_decoding=off` for the initial parser/runtime baseline; draft validation
should be a separate new preparation with an explicitly compatible target/draft.
Select all three frameworks only if the model is supported on all three; otherwise
prepare separate fresh selections per backend. Keep the actual configured GPU
hourly price and allocation limits; the UI's timeout minimum is 60 minutes, so do
not invent a shorter unsupported timeout to satisfy the spending guard.
This is a smoke validation, not a statistically meaningful performance study.

Random mode is a small frozen-workload baseline. For a custom dataset, validate a
new small sample with exactly those token/request/concurrency values before
preparation. Do not alter an existing approved dataset/sample. A dry-run review
checks configuration but does not verify live registry/secret/W&B connectivity.
A live-mode preparation performs those reference checks and may remain unapproved;
preparing it does not authorize submission.

Before any approval, inspect the new `request.json`, `intent.json`, and
`workload.json` downloads. Verify new digests, trial budget 2 in the common and
effective per-framework Optuna settings, pinned model revisions, identical frozen
workload SHA-256 across backends, token/request counts, spending limits, tracking
references, and the new request/approval hash. The read-only W&B reference check is:

```bash
# Use the new preparation's UUID and the same environment as the web/queue processes.
python web/manage.py wandb_check --prepared NEW_PREPARATION_UUID
```

Leave it unapproved. A later separately approved GPU validation should confirm
16/16 main requests (warmup separate), valid objectives only for complete runs,
W&B runs, workload hashes, and retained failure diagnostics/attempts.zip where
failures occur. Do not create production failures merely to test the archive.
CPU regression tests cover that failure path deterministically.

## Rollback

Stop new preparations. Restore `platform-images.json.before` and
`platform-image-aliases.json.before` from the release directory, or restore the
effective image/alias environment overrides from `effective-*.before.json` in the
same process manager. If retaining preparations made during the new release,
keep the merged alias file: it already contains old mappings and permits those
new preparations to resolve their reviewed digests. Restart the same web and
queue processes with the restored image defaults. Restore any separately changed
controller package/environment and restart only that controller if applicable.

The old images and aliases remain published. A default-image rollback affects
future preparations only; it does not change or cancel running jobs, existing
approvals, or new-release preparations. Leave unwanted new preparations unapproved
or cancel them through the normal explicit controls. Do not restore an older
database over newer studies. No automatic resubmission, tag overwrite, or migration
is part of rollback.

Dependency behavior follows pip's [constraints documentation](https://pip.pypa.io/en/stable/user_guide/#constraints-files):
constraints limit versions but do not themselves install packages. Actual base
compatibility remains subject to the build resolver and later GPU verification.
