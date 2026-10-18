# GP failures and TRT-LLM cleanup

TRT-LLM managed trials run sequentially. Each trial owns its serving process,
MPI executor workers, and launcher helpers. Worker handles are collected while
the server is alive so cleanup can still find workers after reparenting.
Shutdown first signals the serving parent, allowing it to coordinate executor
shutdown. Remaining owned processes receive TERM and then KILL. The run refuses
to continue if live workers remain, worker discovery could not be verified, or
the serving port remains occupied. Readiness also verifies that an owned process
listens on the serving port. Startup interruptions invoke cleanup. These checks
apply to TRT-LLM; other backends retain their existing shutdown sequence.

GP keeps the selected sampler. Before a benchmark starts, its effective config
is canonicalized and fingerprinted, including fixed server settings and model
identity. YAML settings are compared by contents rather than filenames, and
floating-point representation noise is normalized. Previously failed configs
are rejected before server launch. A fresh Optuna trial requests the next GP
proposal. After 10 consecutive rejected proposals the study stops and reports
the reason; it never substitutes a random sampler. Rejected proposals are
marked with `gp_rejected=True` and do not consume the executed-trial budget.
Completed configurations remain eligible for repeat measurements in the lab.

Each GP study writes `gp-failures-<study-hash>.json` under its run/output directory.
The study's `gp_failure_log` attribute contains the exact path. Writes replace
the JSON file atomically; the log and failed study trials restore the blocklist
when resuming. The JSON contains:

- `identity`: model/framework and execution identity.
- `max_consecutive_rejections`: the bounded rejection limit (10).
- `failures`: trial number, canonical effective config, fingerprint, reason,
  and whether the failure blocks that config.
- `rejections`: proposal trial number, config, fingerprint, reason, and
  `server_launched: false`.
- `stops`: rejection-limit diagnostics or fatal interruption/error reasons.

An infrastructure cleanup failure stops execution instead of becoming a
recoverable benchmark failure. Fatal errors and interruptions are logged but
do not block their configurations. Benchmark failures retain their failure
state; no artificial objective scores or penalties are introduced.

Both inference optimization lab notebooks, the extracted experiment workflow,
and the regular CLI's `GPSampler` search use the policy. The CLI also retains
its existing completed-configuration deduplication.

CPU tests cover detached-worker cleanup, occupied-port rejection, startup
interruption, blocked proposals, resumption, and rejection-limit stops. Validate
the process lifecycle on the pinned TRT-LLM container before trusting new GPU
benchmark comparisons; these CPU checks do not establish GPU performance.
