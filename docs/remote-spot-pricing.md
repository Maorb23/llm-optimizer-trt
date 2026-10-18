# Remote spot pricing

Remote optimization requests support scheduler-only pricing fields:

```json
{"preemptible": true, "follows_spot_price": true}
```

For a configured pricing policy, use:

```json
{"preemptible": true, "spot_pricing_policy_id": "YOUR_POLICY_ID", "follows_spot_price": false}
```

Choose exactly one pricing mode for preemptible jobs. Nonpreemptible jobs cannot
set a policy or follow spot price. The remote API retains its nonpreemptible default;
hf-model-watch now defaults its prepared requests to preemptible execution.

Legacy requests without either pricing field still read
`NEBIUS_SPOT_PRICING_POLICY_ID`, or follow spot price when unset. The controller
saves the resolved choice in its request JSON at submission. Explicit JSON pricing
fields take precedence over environment changes. Nebius receives
`--spot-pricing-policy-id` or `--follows-spot-price` accordingly.

Scheduler pricing fields are stripped from the worker payload to preserve
compatibility with existing worker images. No Optuna execution changes are needed.
