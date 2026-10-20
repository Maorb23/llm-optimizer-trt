# Managed controller execution

`Controller.tick()` and `python -m llm_optimizer.remote ... serve` use the same
nonblocking process lease beside the database. Serve holds the lease for its entire
lifetime, including provider waits and sleep intervals. A competing tick/serve fails
before provider calls. Submission, status and cancellation remain available.

`remote.service.run_forever` replaces the original CLI loop. It emits an atomic
`.controller-health.json` heartbeat every two seconds while holding the lease and
records successful tick completion or a degraded state. Exceptions are logged by type
to avoid exposing credentials. Unexpected process death releases the OS lease.

Optional host management uses a `.controller-managed.json` sidecar containing
`{"schema_version":1,"managed":true,"concurrency":3,"startup_timeout_seconds":20}`.
The application calls `ensure_running(database)` after validating explicit approval.
A fresh heartbeat plus a held lease returns immediately; otherwise it creates
`.controller-start` for an operator-installed systemd path watcher and waits for
readiness. Without readiness it raises and the application must not submit.
Missing registration preserves unmanaged operation; malformed registration fails.

The host service must be installed separately with the intended credentials, database
and concurrency. A file signal alone cannot start a process without that host watcher.
No host credentials or systemd bus mounts are required in the chat agent container.
This controller-side feature requires no GPU serving-image rebuild.

Tests are in `tests/test_controller_service.py`. The HF Model Watch installation
is documented in its `docs/managed-optimizer-controller.md`.
