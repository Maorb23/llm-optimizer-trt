# GP proposal rejection with pruning

The CLI and pruning notebook reject exact repeats of previously failed or pruned
GP configurations before starting a server. The GP sampler remains unchanged;
there is no random recovery sampler. This prevents duplicate GPU execution but
does not teach the GP model from partial benchmark scores.

Rejection history is scoped to the model/draft, fixed settings, workload,
objective, search space, and pruning settings. Increasing trial or GPU budgets
does not invalidate history. Changing the execution contract requires a different
study/output directory. Resume restores history from the audit log and Optuna
trials, including trials from before this protection was introduced.

Rejected proposals are stored as `FAIL` with `gp_rejected=True`,
`benchmark_executed=False`, `gp_rejection_reason`, and
`gp_original_trial_number`. They have no objective or intermediate scores, do not
invoke measurement callbacks, and do not consume benchmark execution budgets.
They still occupy Optuna trial numbers. Actual early-stopped benchmarks retain
`PRUNED` status and consume one benchmark attempt.

The audit file linked by the study's `gp_failure_log` attribute contains separate
`failures`, `pruned`, `rejections`, and `stops` lists. Rejections link to the original
trial and state. After ten consecutive rejected proposals, the run stops and
sets `gp_stop_reason`. The counter resets after an accepted execution. Restarting
the run starts a fresh rejection counter while retaining blocked history.

Restart the notebook kernel and rerun its setup/definition cells after updating
the repository so existing in-memory policy definitions are replaced. No GPU
measurements or search-space changes are required for enabling this guard.
