# Gemma 4 12B DSpark on SGLang 0.5.22

Investigated 2026-10-22. **The parser can be repaired, but stock SGLang
0.5.22 does not implement the draft architecture `Gemma4DSparkModel`.**
Changing the two model IDs in the Qwen cell is insufficient. No verified
end-to-end compatible version combination was found.

## Verified from official source

- [0.5.22 dependencies](https://github.com/sgl-project/sglang/blob/v0.5.22/python/pyproject.toml)
  explicitly pin Transformers 5.19.0 and Torch 2.14.1. Downgrading either is
  not an evidence-based repair for this release.
- [Draft config](https://huggingface.co/deepseek-ai/dspark_gemma4_12b_block7/blob/main/config.json):
  `model_type=gemma4_text`, `architectures=[Gemma4DSparkModel]`, five full
  attention layers, block size 7. Effective full-attention head dimension is
  512 and KV head count is 1; raw base values are 256 and 8. Full attention
  uses partial rotary factor 0.25 and proportional RoPE. Keep those semantics.
- [Parser](https://github.com/sgl-project/sglang/blob/v0.5.22/python/sglang/srt/utils/hf_transformers/config.py)
  copies a text config onto itself, calling `hasattr` on guarded attributes.
  Its Gemma normalization also omits `gemma4_text` and assumes `text_config`.
- [PR #43714](https://github.com/sgl-project/sglang/pull/43714) addresses exactly
  this failure with three small parser edits. It was open and unmerged when
  inspected. Its author reports CPU regression results, not GPU inference.
  [Issue #43434](https://github.com/sgl-project/sglang/issues/43434) and
  [earlier PR #32231](https://github.com/sgl-project/sglang/pull/32231) describe
  the related text-only full/SWA shape issue.
- [DSpark models](https://github.com/sgl-project/sglang/blob/v0.5.22/python/sglang/srt/models/dspark.py)
  register `DSparkDraftModel`, `Qwen3DSparkModel`, and `LingDSparkModel`.
  There is no Gemma draft implementation in this release's model tree.
  [Generic DFlash backbone](https://github.com/sgl-project/sglang/blob/v0.5.22/python/sglang/srt/models/dflash.py)
  uses Qwen-style RMSNorm and SiLU. Relabeling the checkpoint as the generic
  DSpark model does not implement Gemma's norms, activation, K=V attention,
  and partial RoPE. Transformers fallback is not a Gemma DSpark implementation.
- [DSpark config](https://github.com/sgl-project/sglang/blob/v0.5.22/python/sglang/srt/speculative/dspark_components/dspark_config.py)
  derives gamma as `speculative_num_draft_tokens - 1`: the cell's 8-token
  verify window for a 7-token draft is consistent with this code.

The earlier HF 401 traceback used `google/gemma-4-12B-i`, missing the final
`t`. That fails before config parsing and is a separate issue. Check credentials
only if the correctly spelled repository also fails.

## Apply and check the parser repair

Run from this repository with the serving environment activated (the notebook
must use the same interpreter). Retain the installed CUDA build of Torch and
SGLang's accompanying kernel packages. These commands repair the existing
0.5.22 environment; they do not reinstall or downgrade dependencies.

```bash
python -m pip check
python -c 'from importlib.metadata import version; print({p:version(p) for p in ("sglang","transformers","torch")})'
python tools/patch_sglang_gemma4_text.py
```

The script backs up the installed parser as `config.py.before-gemma4-text`,
checks the source anchors, and applies only the three PR #43714 changes.
Restart the notebook kernel after patching. Do not set
`allow_global_per_layer_attribute_access`. The existing SGLang helper reads
each layer first, rejects unsupported overrides or varying shapes within one
attention type, and only then maps full/SWA shapes into SGLang's fields.

Download only config files initially, pinning both revisions:

```python
from huggingface_hub import HfApi, snapshot_download

TARGET_MODEL_ID = "google/gemma-4-12B-it"
DRAFT_MODEL_ID = "deepseek-ai/dspark_gemma4_12b_block7"
TARGET_REVISION = HfApi().model_info(TARGET_MODEL_ID).sha
DRAFT_REVISION = HfApi().model_info(DRAFT_MODEL_ID).sha
draft_config_path = snapshot_download(
    DRAFT_MODEL_ID, revision=DRAFT_REVISION, allow_patterns=["config.json"]
)
print(draft_config_path)
```

Use the printed path below:

```bash
python tools/check_sglang_gemma4_dspark.py --draft /absolute/pinned/draft/snapshot
```

This exercises real Transformers config round trips and SGLang parsing,
checks mixed full/SWA shapes, checks the actual draft's 512/1 shapes and
partial rotary factor, then requires native `Gemma4DSparkModel` registration.
On stock 0.5.22 it is expected to stop with **BLOCKED** after the parser repair.
This is a useful diagnostic, not successful inference.

## Changes to the notebook cell

Set the two IDs above and retain the resolved full commit SHAs as
`TARGET_REVISION` and `DRAFT_REVISION`. Before the full draft download or
`configure(...)`, run the checker using the notebook interpreter:

```python
import subprocess
import sys

subprocess.run(
    [sys.executable, "tools/check_sglang_gemma4_dspark.py",
     "--draft", draft_config_path],
    check=True,
)
```

Do not continue to `configure` or a study if that fails. The provided helper
commits target setup before the cell adds DSpark settings; an exception can
otherwise leave a previous or target-only profile available to later cells.
The helper's `enable_dflash=False` is appropriate for the manual DSPARK setup.
Recognizing `SpeculativeAlgorithm.from_string("DSPARK")` alone proves only
that the algorithm exists, not that this draft architecture is supported.

## Minimal GPU launch diagnostic (not verified here)

Only after the architecture gate passes with a reviewed Gemma DSpark
implementation, download the full snapshots at the same pinned revisions.
The following uses local target/draft paths and a small single-request budget.
It is a candidate diagnostic command, not a verified working recipe for stock
0.5.22. Use `--help` to check backend options if testing a different build.

```bash
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path /absolute/pinned/target/snapshot \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path /absolute/pinned/draft/snapshot \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 8 \
  --dtype bfloat16 \
  --tp-size 1 \
  --context-length 2048 \
  --max-running-requests 1 \
  --mem-fraction-static 0.75 \
  --host 127.0.0.1 --port 30000
```

After the server reports ready, in a second terminal:

```bash
curl --fail http://127.0.0.1:30000/health
curl --fail http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"Count from 1 to 20: 1, 2, 3,","sampling_params":{"temperature":0,"max_new_tokens":32,"ignore_eos":true}}'
```

Require a successful generation and evidence in the server configuration/logs
that DSPARK loaded the intended draft. Compare with target-only greedy
generation before doing performance work. One completion establishes a smoke
test, not broad correctness or an acceleration claim.

## Validation performed here

No Torch, Transformers, SGLang, NVIDIA driver tool, or H100 is available in this
workspace. No model inference, package-combination compatibility, full installed
parser integration, or GPU launch is claimed.

Five offline source-level tests passed using official v0.5.22 parser functions
and the actual v5.19.0 heterogeneity mixin, with only its logging import stubbed:
exception reproduction; per-layer preservation for draft and mixed full/SWA
configs; rejection of varying shapes within one type; rejection of unsupported
per-layer attributes; patch idempotence. These use a minimal config shell,
not a full `Gemma4TextConfig`. The installed-environment checker above supplies
the separate integration test. Repository patch-application tests and Ruff
checks passed (3 repository tests). Existing notebook/backend tests could not
collect because this workspace lacks `huggingface_hub` and `optuna`; neither
was installed or invoked for a search. No Optuna searches or study changes
were made.
