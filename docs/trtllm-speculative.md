# TRT-LLM draft setup

Use `configure_hf_models(..., framework="trtllm", speculative_algorithm="Eagle3",
draft_model_id=..., max_draft_len=3)` in the pruning notebook or through `Lab`.
Both use the same Python helper and YAML serializer. Omit `speculative_algorithm`
to restore the original target-only profile. Existing vLLM/SGLang setup is retained.

The helper resolves target and draft refs to immutable HF commits, downloads both
snapshots, checks checkpoint fields, and validates the installed native schema.
`hf-trtllm-models.json` records both revisions and the preview YAML path. Reuse the
recorded `target_revision` and `draft_revision` on resume. Per-trial YAML combines
`speculative_config` with sampled graph, scheduler, attention, and KV-cache fields.
Fixed YAML paths are rejected explicitly rather than overwritten. Search spaces,
precision, workload, and other performance settings are unchanged.

The native [1.3.0rc26 schema](https://github.com/NVIDIA/TensorRT-LLM/blob/v1.3.0rc26/tensorrt_llm/llmapi/llm_args.py)
defines separate `Eagle3`, `DFlash`, and `DSpark` configurations with
`max_draft_len` and `speculative_model`. Other native algorithms include MTP,
NGram, SA, PARD, and `Draft_Target`; these are outside this snapshot helper's
three supported draft modes. The exact source spelling is `Draft_Target`, even
though the feature guide lists `DraftTarget`. The helper requires 1.3.0rc26 for
speculative setup; ordinary target-only setup has no new version restriction.

The [pinned EAGLE3 loader](https://github.com/NVIDIA/TensorRT-LLM/blob/v1.3.0rc26/tensorrt_llm/_torch/models/modeling_speculative.py)
uses Llama-style draft layers (including DeepSeek MLA variants), with a separate
Mistral Large 3 path. This helper uses the default Llama-style path. The documented
example pairs `meta-llama/Llama-3.1-8B-Instruct` with
`yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`. EAGLE v1/v2 checkpoints are incompatible.
A missing `draft_vocab_size` is valid in this release and defaults to `vocab_size`.

The [pinned DFlash loader](https://github.com/NVIDIA/TensorRT-LLM/blob/v1.3.0rc26/tensorrt_llm/_torch/models/modeling_dflash.py)
resolves draft architectures through TRT's model registry, with a `model_type`
fallback for custom labels and a separate Laguna path. It rejects checkpoints
that declare DSpark heads. The helper likewise distinguishes DSpark head metadata,
checks captured target layers, and requires DSpark's checkpoint block size to equal
`max_draft_len`. It does not infer compatibility from repository names. Unsupported
registry architectures fail before profile activation; specialized Laguna/Mistral
paths and embedded DSpark checkpoints are not configured by this helper.

Metadata and fingerprints include the draft revision and structured native
settings. Regular study resume, previous-phase pruning checks, and held-out
validation reject changes. CLI results retain effective YAML contents, including
speculation. Model IDs and matching dimensions alone cannot prove training-pair
compatibility: select a draft trained for the exact target.

Tests use mocked HF snapshots, TRT schemas/registry, and CUDA metadata. They verify
translation, identity, rollback, and disabling without loading weights. They do not
validate GPU loading, generation parity, acceptance rate, memory use, or speed.
Run native serving and quality preflight on the pinned GPU environment before
asserting those properties. The helper preserves existing verification declarations.
