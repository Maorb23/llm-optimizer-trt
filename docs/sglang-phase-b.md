# Phase B SGLang search on H100

The extracted experiment runner and `docs/optuna_inference_optimization_lab_prune.ipynb`
use the original separate Optuna categoricals: `disable_radix_cache` and
`schedule_policy`. Their choices are passed through unchanged, so all six cache/
scheduler combinations are available, including cache-off with
`shortest-prefill-first`. Unsupported combinations may fail at server startup.

`sglang_space.py` is retained but unused by those execution paths. The older
`docs/optuna_inference_optimization_lab.ipynb` and
`optuna_inference_optimization_lab_updated.ipynb` still explicitly use its five-value
`cache_policy` encoding.

Expanded prefill graph choices are `disabled` and `breakable`. Decode choices remain
`disabled`, `full`, and `tc_piecewise`. `breakable` is passed as
`--cuda-graph-backend-prefill=breakable`. Explicit verified preflight subsets remain
authoritative: an existing profile restricted to `disabled` must be deliberately
updated to include `breakable` after checking the pinned SGLang build on H100.

## GP failures

The shared GP failure policy fingerprints the resolved configuration, including
fixed server settings. Before launching the benchmark it rejects configurations
that failed previously. Failed trials remain `FAIL` with no objective value.
Rejected proposals also use `FAIL`, carry `gp_rejected=True`, and do not consume
the benchmark execution budget or invoke benchmark callbacks. GP requests another
proposal, with a maximum of ten consecutive rejections. Reaching that limit stops
cleanly with `gp_stop_reason` and a diagnostic in the existing GP audit JSON. There
is no replacement sampler or artificial penalty value.

Resume restores failures from the audit JSON and stored failed trials. The stored
`effective_config` snapshot takes precedence over the older `config` attribute,
so restoration does not depend on generated YAML files still being present.
Infrastructure failures marked `gp_non_config_failure` are not blocked.

## Resume and migration

Study names, storage locations, metrics, W&B integration, and benchmark execution
semantics are unchanged. Contracts record native choices and the sampling menu.
Compatible separate-parameter studies can resume. Studies using the combined
`cache_policy` parameter have a different contract and require a fresh study when
switching back to separate parameters; contract validation rejects mixing them.

Start a fresh SGLang Phase B study using the existing run-label mechanism, or a
fresh experiment output directory; keep old SQLite databases and artifacts for
analysis. Do not edit historical trials/distributions or remove contract fields to
bypass validation. Restart notebook kernels and rerun setup after updating the
installed package. GP blocklists remain scoped to their original study/identity;
new studies do not automatically import another study's failures.

## Optional draft-model decoding

The pruning notebook now has an optional **Hugging Face target and DFlash setup**
cell immediately after its adapter/preflight definitions. It automatically uses
the engine selected in `REAL_FRAMEWORKS`, including vLLM; it does not require
switching to SGLang. Set
`HF_MODEL_SETUP_ENABLED=True` to download/cache the selected models; set
`ENABLE_DFLASH=False` for target-only serving. A target revision of `None` resolves
the Hub's main branch to a concrete commit before download. The draft defaults to
the pinned Qwen3.5-4B DFlash revision. Paths and resolved revisions are displayed
and saved to `hf-<framework>-models.json` in the experiment directory.

The notebook passes the profile's fixed server settings through `benchmark_for()`
for both search and independent validation. Its contracts and GP fingerprints
include the draft settings/revision. SGLang DFlash setup narrows mixed-chunk and overlap
choices and checks the installed SGLang CLI before downloading. vLLM instead receives
its native `speculative_config` JSON and keeps its existing search choices, using
`VLLM_NUM_SPECULATIVE_TOKENS` independently from SGLang's block size. The setup never installs
packages or claims to have verified GPU model compatibility. Native/quality
verification declarations remain unchanged. Use a fresh experiment when changing
models or enabling a draft; first smoke-test the actual SGLang/H100 combination.

The native SGLang registry exposes `speculative_algorithm`,
`speculative_draft_model_path`, `speculative_num_steps`, `speculative_eagle_topk`,
and `speculative_num_draft_tokens`. The separate experiment CLI configuration still
has no typed speculative layer. Extending that CLI requires these scoped changes:

1. Add an optional `speculative_decoding` object to experiment defaults and config
   validation, separate from expanded attention/cache choices. Require an explicit
   existing local draft path plus explicit algorithm/settings; never download or
   infer a draft checkpoint. An absent object must leave current runs unchanged.
2. Pass those native flags as fixed settings only to the SGLang Phase B adapter.
   Preserve the existing target model snapshot; validate draft/target compatibility
   and the pinned runtime's support in preflight before starting any study.
3. Include the complete speculative layer and draft revision in the experiment
   contract, measured environment, and GP effective-config identity. Apply the
   same layer during independent validation and reject incompatible resumes.
4. Add CPU tests for flag serialization, disabled defaults, target preservation,
   missing local draft paths, and resume identity, then run an H100 compatibility
   smoke test with the actual draft checkpoint.

The CLI extension remains documented rather than implicitly enabled.

## Changed notebook cells

In each of `docs/optuna_inference_optimization_lab.ipynb`,
`docs/optuna_inference_optimization_lab_prune.ipynb`, and
`optuna_inference_optimization_lab_updated.ipynb`, search-space code is in cells at
zero-based indices **6** (`optuna-objectives-006`) and **10** (`optuna-objectives-010`).
Cell 6 defines expanded choices and parameter sampling; cell 10 defines the sampler
menu used for contracts and Cartesian traversal. Existing saved outputs are retained
as historical output and must not be read as results of the updated source.

The pruning notebook additionally wires `fixed_server` in its adapter cell
`optuna-objectives-012` and adds `hf-sglang-model-setup-notes` and
`hf-sglang-model-setup`. Those stable cell IDs identify the optional setup even
when inserting cells changes the displayed numeric indices.
