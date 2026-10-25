# Completion stream accounting and rollout

The CPU regression confirms that indexing `choices[0]` fails on trailing usage
with `choices: []`. It does not establish the cause of the production Qwen3.5-4B
99/100 measurements. Collect the new bounded request diagnostics and matching
server logs from a future authorized run before attributing those failures.

OpenAI completion streams now process usage independently of text, ignore empty
choices, recognize finish events and stop at `[DONE]`. Clean EOF requires a finish
event or terminator; partial output followed by a transport error remains failed.
Usage and finish events do not advance TTFT or inter-token timing. Latency includes
stream completion. Nonstream completion responses remain supported.

Numeric `prompt_tokens`, `completion_tokens`, and `total_tokens` are retained in
`RequestFuncOutput.usage`; optional detailed results add `usages`. Reported
completion counts take precedence. Without reported usage, aggregate metrics use
the existing tokenizer on observed output instead of assuming the requested
maximum was generated. Retokenization is an estimate and can differ from the
server tokenizer. The direct request object's fallback length is retained for
compatibility and marked with `output_len_is_estimate`.

Failed and missing responses invalidate the benchmark. Objective extraction also
checks explicit failed/completed counts, preserving compatibility with old metric
records lacking accounting fields. No request retry is added. Warmup is separate;
the full main workload is submitted once, including its first and last requests.

`request_errors` contains at most 20 entries, with zero-based workload index,
category, and a reason of at most 2,000 characters. OpenAI errors use fixed schema
or transport messages and omit arbitrary response bodies and exception text.
Other backend error bodies/tracebacks become category summaries. Raw prompts and
outputs are not added to diagnostics. Live checkpoints persist errors before
objective rejection, using original workload indices even when completion order
differs. Existing telemetry and Optuna failure summaries retain this evidence.
Workers upload `attempts.zip` containing `results-attempts/*/execution.json`, even
when there is no successful metric row or study database. The new archive includes only identity, timings, status, accounting and request
diagnostics, excluding commands, environment, log excerpts and raw server logs; existing execution artifacts retain their existing behavior.

## Required image and process updates

No image was built or published for this change. The package is copied into worker
images, so updating the controller checkout alone cannot fix running GPU workers.

For the current Infiro configuration in the adjacent `hf_validator` checkout,
rebuild the selected package-bearing images:

| Framework | Base worker Dockerfile | W&B worker Dockerfile | Change needed |
| --- | --- | --- | --- |
| vLLM | `docker/vllm.Dockerfile` | `docker/vllm-wandb.Dockerfile` | Parser, accounting, diagnostics, artifact upload |
| SGLang | `docker/sglang.Dockerfile` | `docker/sglang-wandb.Dockerfile` | Shared diagnostics/accounting/artifact upload; native parser unchanged |
| TensorRT-LLM | `docker/trtllm.Dockerfile` | `docker/trtllm-wandb.Dockerfile` | Shared diagnostics/accounting/artifact upload; native parser unchanged |

Build the actual variant deployed for each framework; both base and overlay need
not be deployed. W&B overlays copy and reinstall source themselves.
`docker/sglang-fix.Dockerfile` also copies the package and needs rebuilding only if
that legacy variant is selected. Pin upstream serving bases to reviewed immutable
digests to avoid an incidental runtime upgrade.

The currently configured repositories are
`cr.eu-north1.nebius.cloud/e00p24e8mwqab11v1e/{vllm,sglang,trtllm}`.
Their current configured digest prefixes are respectively `6fee13c1255f`,
`f6b1fc8f9d4b`, and `127113a484eb`; these identify configuration, not a live registry
verification. After a separately authorized build/publish:

1. Record each new registry manifest digest and assign a new short release tag.
   Do not reuse the historical `workload1` aliases.
2. Update `hf_validator/config/platform-images.json` with full
   `repository@sha256:<new-digest>` values, or update the effective
   `INFIRO_SERVING_IMAGES` environment override.
3. Add matching new digest-to-tag entries in
   `hf_validator/config/platform-image-aliases.json`, or the effective
   `INFIRO_IMAGE_ALIASES` override. Keep old entries for old preparations.
   Infiro requires a private Nebius registry alias of at most 64 characters and
   verifies that its manifest bytes hash to the reviewed digest before submission.
4. Restart the Infiro Django/Gunicorn process and the `manage.py platform_worker`
   process using the deployment's process manager. Django loads image settings at
   startup, and queue subprocesses inherit those settings. Restart any separate
   long-running optimizer controller after installing this checkout's package if
   it imports its own copy. No database reset or migration is needed.
5. Create a new preparation and inspect its `images` and request hash. Preparation
   copies `settings.INFIRO_SERVING_IMAGES` into execution configuration. Existing
   prepared payloads and running jobs keep their old image references. Leave their
   stored requests, approvals, databases and historical artifacts intact. Any later
   live submission still follows the normal explicit study approval process.

## CPU validation in this workspace

Using `/home/maorb/.venvs/bento_extend/bin/python` with `PYTHONPATH=src`, the
focused suite of completion streams, failure diagnostics, benchmark accounting,
telemetry, live pruning, multi-objective scoring, optimization and remote workers
passed **109 tests** with one existing controller test deselected. That test,
`test_uncertain_cancellation_does_not_consume_concurrency_slot`, failed in the
initial run because the pre-existing uncommitted controller guard rejects a second
study while the cancelled study remains active. No controller behavior was changed
for this fix. The two artifact checkpoint tests also passed independently.

Coverage includes normal and nonstream responses, trailing usage, empty choices,
finish-only events, terminators, malformed events, server errors, clean premature
EOF, transport failure after partial text, timing unaffected by metadata, usage
preservation on failure, missing outputs, bounded diagnostics, first/middle/last
failures, out-of-order checkpoint rejection, and separate warmup accounting.

The broader CPU suite (483 collected tests, integration tests excluded) was
attempted and interrupted after slow progress and multiple failures; it is not a
passing full-suite result. The two new/extended diagnostic test files pass Ruff,
and `git diff --check` passes. Module-wide Ruff reports 55 issues in existing code;
the installed mypy rejects the repository's configured Python 3.9 target. An
attempt with a newer type-check target was interrupted without a result. No GPU
behavior or live registry/deployment behavior was validated.
