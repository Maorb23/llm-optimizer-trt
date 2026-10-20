# Expanded remote search profile

`llm_optimizer.search_profiles.expanded_server_args(framework, concurrency, input_tokens)`
returns native server argument domains and worker-local config files. It shares
`EXPANDED_CHOICES` with `experiment.lab`; importing the profile does not import Optuna,
SciPy or the experiment execution machinery. Existing capped lab grids are preserved.

The full remote domains are 1024..8192 step 1024 at concurrency 1, and 1024..32768
step 256 plus capacities 8..128 step 8 for any concurrency greater than 1. As with
`Lab.regime_key`, concurrency 32 uses the batched regime without changing its workload.

vLLM booleans use coupled positive/negative flags, and graph modes become shell-quoted
`compilation_config` JSON. SGLang uses the existing valid cache-policy combinations
and native graph backend flags. TRT-LLM nested settings become JSON documents (valid
YAML) in `OptimizationRequest.server_config_files`. The batch-size/config domain keeps
graph capture sizes within the selected batch size. TRT token budgets are filtered to
fit the input length so chunking-off choices are meaningful.

The remote worker materializes config files under its output directory and replaces
`@LLM_OPTIMIZER_CONFIG@/filename` placeholders with quoted local paths. Filenames cannot
contain traversal components. Requests exceeding 48 KB are transported with
`LLM_OPTIMIZER_PAYLOAD_ZLIB_B64`; smaller legacy payloads retain the existing format.

Updated controller and worker packages are required, including rebuilt serving images.
No existing published image capability is implied by its tag. Optuna still runs through
the normal optimizer CLI. Client arguments and multi-objective metrics are unchanged;
no second optimization loop is introduced. hf-model-watch supplies 30 TPE trials by
default and preserves explicit execution defaults and parameter overrides.
