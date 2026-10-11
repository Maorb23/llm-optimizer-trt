# Multi-objective optimization

Existing scalar `--optuna-objective output_throughput` requests retain their behavior. For a
multi-objective study, supply comma-separated metrics:

```bash
llm-optimizer --framework vllm --model org/model --gpus 1 \
  --server-args 'max_num_seqs=[1,2,4,8]' \
  --client-args 'dataset_name=random;random_input_len=128;random_output_len=32;max_concurrency=4' \
  --search-strategy optuna --optuna-objective output_throughput,mean_e2e_latency_ms \
  --optuna-trials 30 --output-json results.json
```

Remote requests alternatively use `optuna.objectives: ["output_throughput", "mean_e2e_latency_ms"]`.
Omit scalar `objective` when supplying `objectives`. Framework overrides follow the same contract.

The study uses separate Optuna directions and returns an objective vector for every successful
trial. `optimization.study_summary` adds the complete Pareto front, best trial per objective,
completed trial count, and parameter importance per objective. The frontier includes infeasible
complete trials with their `slo_feasible` flag, so measurement tradeoffs are retained. Existing
throughput best-configuration fields remain readable. Importance uses seeded fANOVA with one
explicit target per objective. Install `scikit-learn` to enable importance; insufficient data or a
missing dependency produces an explicit unavailable reason.

Remote manifests add optional `semantic_context`, `test_results`, and `study_summary`. Successful
trial records are recovered from JSONL even when a framework process fails. Study analysis is
independent of controller lifecycle operations. Existing checkpoints and `--continue` remain the
explicit path for continuing trials; reanalyzing saved measurements never resumes them.
