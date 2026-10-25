# Custom HF datasets for optimization

Custom datasets use ShareGPT records: `{"conversations": [{"from": "human", "value": "A prompt"}, {"from": "gpt", "value": "An answer"}]}`.
Roles must be `human`, `gpt`, or `system`; every turn must contain nonempty text,
and every record must contain a human turn. Schema errors fail preparation.
The first human turn supplies the prompt. Its token IDs are repeated or truncated
to `input_tokens`; every request asks for `output_tokens`. This measures fixed
length serving performance, not natural conversation lengths or answer quality.

Pass these choices through `hf-watch optimization prepare --run-id RUN --request-file choices.json`:

```json
{
  "dataset": {
    "repo_id": "https://huggingface.co/datasets/ORG/DATASET",
    "revision": "main",
    "config": "default",
    "split": "train"
  },
  "num_prompts": 100,
  "seed": 7,
  "input_tokens": 512,
  "output_tokens": 128
}
```

Repo IDs such as `ORG/DATASET` also work. Omit revision/config to use the repository
default; omit split for `train`. Existing GPU, framework and budget preparation
choices still apply. NanoClaw's `prepare_optimization` accepts the same `dataset`
object. Preparation and explicit approval remain required before submission.
Omitting `dataset` preserves the existing random/ShareGPT defaults.

Preparation resolves the dataset to a commit and loads that commit with the HF
Datasets library. It validates records, samples without replacement using a local
seeded generator, and freezes decoded fixed-length requests. The manifest records
split/config, source row indices, tokenizer commit, seed, and SHA-256. Insufficient
rows, unavailable dependencies, access failures and malformed datasets are errors;
custom workloads never fall back to ShareGPT. Source row indices refer to the
selected split expression at the pinned commit. Keep that expression for auditing.

`request.json` contains the frozen workload. Approved submission reuses it without
reloading HF. The controller stages the content-addressed JSON through its existing
S3 artifact adapter, and workers verify it before running. Frameworks and trials
reuse the same requests. Study compatibility includes dataset identity and workload
hash; a different dataset, sample, tokenizer, or token length cannot reuse a study.
Retain prepared files and workload artifacts. Treat prompt text and artifacts as
potentially private data and use an appropriately restricted artifact bucket.

Use HF environment/cache authentication (`HF_TOKEN` or an existing HF login) on
the preparation host, and existing AWS environment credentials for staging and
workers. Never put tokens, signed URLs, or credential fields in choices, notebook
cells, or request JSON. Repository URLs with credentials/query strings are rejected.
Loader failures suppress upstream exception text to avoid credential exposure.
HF authentication is not needed on workers to reload the dataset; workers may
still need it to load a gated model.

The shared notebook API is `llm_optimizer.hf_workload.prepare_hf_workload`.
See `optimization_hf_dataset.ipynb` for a CPU preparation example. A
materialized manifest can also be used locally via
`--client-args workload_path=/absolute/path/workload.json`. Native llm-optimizer
CLI preparation accepts `--client-args hf_dataset=ORG/DATASET`, optional
`hf_dataset_revision`, `hf_dataset_config`, `hf_dataset_split`, and the existing
`num_prompts`, `seed`, `random_input_len`, and `random_output_len` arguments.
Materialize once and pass `workload_path` when comparing separate CLI runs.

## Deployment steps

Install updated hf-model-watch and llm-optimizer together. The preparation host
needs `datasets`, `transformers`, and `huggingface-hub` (CPU only; do not install
serving runtimes there). Restart the managed controller with the updated optimizer.
Rebuild each selected worker image with the updated optimizer, including
`hf_workload.py`, `workload_path` client argument, remote manifest contract,
and storage verification. Existing images cannot run custom workloads. Configure
new immutable image tags/digests in execution settings and review the prepared
request before approval. Workers retain their existing S3 and model dependencies;
they do not require `datasets` when consuming a frozen workload.

No GPU jobs or image publications are required for local tests. Image rebuilding,
publishing, controller rollout and an explicitly approved GPU smoke test remain
operator deployment steps.

Remote API callers must materialize once with `prepare_hf_workload` and supply the returned object as `workload_manifest`. Raw `hf_dataset` client arguments are rejected remotely to prevent independent per-framework loading. Default requests omit the new optional field for compatibility with existing worker images.
