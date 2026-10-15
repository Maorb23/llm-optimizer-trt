# Phase B SGLang search on H100

Expanded SGLang studies sample one `cache_policy` categorical with the fixed domain:

| Optuna value | disable_radix_cache | schedule_policy |
| --- | --- | --- |
| cache_on__fcfs | false | fcfs |
| cache_on__lpm | false | lpm |
| cache_on__shortest-prefill-first | false | shortest-prefill-first |
| cache_off__fcfs | true | fcfs |
| cache_off__lpm | true | lpm |

The invalid cache-off/shortest-prefill-first pair has no categorical value.
Native preflight settings still specify `disable_radix_cache` and `schedule_policy`
subsets. Their valid combinations determine the categorical domain once per study;
choices never depend on another sampled parameter. An invalid-only preflight subset
is rejected before trials start. This encoding supports grouped multivariate TPE,
GP, and the existing lazy Cartesian grid. Resolved configs retain the two native
settings and never pass `cache_policy` to SGLang.

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
semantics are unchanged. The expanded study contract records the native choice
subsets and the new resolved sampling menu. Resuming with the same new contract is
supported. The old two-parameter SGLang search space is incompatible with the new
categorical and must not be appended to an existing study. Contract validation
rejects that resume, even if a verified profile excludes `breakable`.

Start a fresh SGLang Phase B study using the existing run-label mechanism, or a
fresh experiment output directory; keep old SQLite databases and artifacts for
analysis. Do not edit historical trials/distributions or remove contract fields to
bypass validation. Restart notebook kernels and rerun setup after updating the
installed package. GP blocklists remain scoped to their original study/identity;
new studies do not automatically import another study's failures.

## Optional draft-model decoding

This change leaves speculative decoding disabled and adds no draft checkpoint.
The native SGLang registry already exposes `speculative_algorithm`,
`speculative_draft_model_path`, `speculative_num_steps`, `speculative_eagle_topk`,
and `speculative_num_draft_tokens`. The experiment configuration currently has no
separate typed speculative layer. Supporting one requires these scoped changes:

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

This is documented rather than enabled implicitly: a compatible draft checkpoint
and pinned SGLang build have not been provided or verified here.

## Changed notebook cells

In each of `docs/optuna_inference_optimization_lab.ipynb`,
`docs/optuna_inference_optimization_lab_prune.ipynb`, and
`optuna_inference_optimization_lab_updated.ipynb`, only code cells at zero-based
indices **6** (`optuna-objectives-006`) and **10** (`optuna-objectives-010`) change.
Cell 6 updates expanded choices and native resolution; cell 10 updates the sampler
menu used for contracts and Cartesian traversal. Existing saved outputs are retained
as historical output and must not be read as results of the updated source.
