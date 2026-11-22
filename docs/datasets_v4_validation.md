# Phase 1 v4 validation

CPU validation used Python 3.12 and the repository's development and notebook
dependencies, plus jsonschema, nbformat and pytest-timeout.

| Check | Result |
|---|---|
| Dataset and campaign v4 suites | 65 passed |
| Full workspace suite | 681 passed, 37 failed, 14 skipped |
| Clean committed baseline at `068af2ad6958080bbc6e35c062b3dfe3719860da` | 599 passed, 37 failed, 14 skipped |
| Ruff on the v4 package and tests | Passed |

The full suite and the clean `git archive HEAD` checkout ran
`python -m pytest tests --timeout=15 -q` with their own source directories.
The 37 failed test IDs match exactly. The workspace includes 17 passing
current-results analysis tests already present outside the committed baseline,
in addition to the 65 v4 tests.

| Existing test file | Failures in each checkout |
|---|---:|
| `test_lab_v2_notebook.py` | 1 |
| `test_optuna_notebook.py` | 28 |
| `test_phase1_experiments.py` | 1 |
| `test_remote.py` | 1 |
| `test_suggested_vllm_baselines_cell.py` | 3 |
| `test_worker_cpu_check.py` | 3 |

All three notebooks passed nbformat validation and executed cell by cell on
their default CPU path. The parity tests compare the v3 search and study cells
against the v4 copies. Managed integration tests exercise Phase A and expanded,
median-pruned Phase B, the normal v3 sampler and serializer path, SQLite resume,
quality gating, shortlist repeats, and the held-out winner at its study
concurrency. They mock server launch, HTTP inference and metric scraping. They
check that telemetry has stopped and cleanup has completed before scoring, and
advance the scoring clock independently to verify the measured window is
unchanged. The dataset suite covers payload projection, conversation history,
sampling and manifests, runtime control rejection, scorers and task accounting.

No GPU campaign, serving process, W&B session, or Docker container was started.
Docker command construction and the evaluator's trusted fixtures are tested on
CPU. Executable scoring inside Docker, native serving compatibility and hosted
tracking remain unverified. Hugging Face metadata confirmed the exact Qwen and
GPQA commits. GPQA data access requires approval and was not available
anonymously. [Usage, pins, and parity checklist](datasets_v4.md)
