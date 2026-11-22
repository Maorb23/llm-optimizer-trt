import json
from collections import Counter

import pandas as pd
import wandb

RUN = "maorblumberg-tel-aviv-university/inference-optimization/ljejf1b0"
run = wandb.Api(timeout=60).run(RUN)

def flatten(obj, prefix=""):
    out = {}
    for key, value in obj.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            out.update(flatten(value, name))
        else:
            out[name] = value
    return out

# Merge only records belonging to the same trial; never forward-fill trials.
trials = {}
for raw in run.scan_history(page_size=1000):
    row = flatten(raw)
    number = row.get("trial/number")
    if number is None:
        continue
    number = int(number)
    trials.setdefault(number, {}).update(
        {k: v for k, v in row.items() if v is not None}
    )

if not trials:
    raise RuntimeError("No trial/number records found in this run.")

def metric(row, name):
    for prefix in ("benchmark", "objective"):
        value = row.get(f"{prefix}/{name}")
        if value is not None:
            return value
    return None

records, reasons = [], {}
for number, row in sorted(trials.items()):
    errors = {
        k: v for k, v in row.items()
        if any(word in k.lower() for word in
               ("failure", "error", "exception", "prune_reason"))
        and v not in (None, "", False)
    }
    reason = row.get("trial/failure_reason")
    if not reason and errors:
        reason = json.dumps(errors, default=str, ensure_ascii=False)
    reason = " ".join(str(reason or "").split())
    if reason:
        reasons[number] = reason

    records.append({
        "trial": number,
        "state": row.get("trial/state", "UNKNOWN"),
        "tok/s": metric(row, "output_throughput"),
        "E2E_p95_ms": metric(row, "p95_e2e_latency_ms"),
        "TTFT_p95_ms": metric(row, "p95_ttft_ms"),
        "reason": reason[:110],
        **{
            k.removeprefix("parameters/"): v
            for k, v in row.items()
            if k.startswith("parameters/")
        },
    })

df = pd.DataFrame(records)
base = ["trial", "state", "tok/s", "E2E_p95_ms", "TTFT_p95_ms", "reason"]
params = [c for c in df.columns if c not in base]
varying = [
    c for c in params
    if df[c].map(lambda x: json.dumps(x, default=str)).nunique() > 1
]
fixed = {c: df[c].iloc[0] for c in params if c not in varying}

print(f"\nRUN: {run.name} | state={run.state}")
print("TRIAL STATES:", dict(Counter(df["state"])))
print("\nFIXED SAMPLED PARAMETERS:")
print(json.dumps(fixed, indent=2, default=str))

print("\nRESULTS:")
print(df[base].round(2).to_string(index=False, na_rep="-"))

# Split parameters into manageable tables.
for start in range(0, len(varying), 5):
    columns = varying[start:start + 5]
    print("\nPARAMETERS:")
    print(df[["trial", "state", *columns]].to_string(index=False, na_rep="-"))

print("\nFULL LOGGED FAILURE / PRUNING REASONS:")
if reasons:
    for number, reason in reasons.items():
        print(f"Trial {number}: {reason}")
else:
    print("No reasons logged in W&B history; server logs/sidecars may be needed.")

df.to_csv("/tmp/gemma-trial-diagnostics.csv", index=False)
