# Autoresearch Program: Probing Strategy Optimization

## Objective

Find the probing configuration that maximizes AUROC for detecting deceptive intent
in Pythia model activations extracted from Secret Hitler gameplay.

## Metric

**Primary metric**: AUROC on held-out test set (from probe_results.json → best_auroc)
**Threshold**: Keep if AUROC > previous best. Discard otherwise.

## What to Modify

You may modify the following in each iteration. Change ONE variable at a time:

1. **Layer selection** (`--layers` flag): Try `all`, `every4`, `every2`, or specific layers like `0,6,12,18,23`
2. **Aggregation method** (`--aggregation` flag): Try `mean_pool` or `last_token`
3. **Label field** (`--label` flag): Try `deceptive_intent` (primary) or `deception_type` (secondary)
4. **Regularization**: Edit `mi/probe.py` → `LogisticRegression(C=...)` to try different regularization strengths (default C=1.0, try 0.1, 0.01, 10.0)
5. **Layer subsets**: Focus on specific layer ranges (early: 0-8, middle: 8-16, late: 16-24)

## How to Run

```bash
# Step 1: Extract activations (if not already done)
python -m mi.extract --dataset data/game_LATEST.json --model pythia-410m --layers LAYER_SPEC --output data/activations/

# Step 2: Run probing
python -m mi.probe --activations data/activations/ --aggregation AGGREGATION --label LABEL --output results/probe_run/
```

## How to Evaluate

1. Read `results/probe_run/probe_results.json`
2. Check `best_auroc` field
3. Compare to previous best AUROC (stored in `results/experiments/best_probe.json`)

## Decision Rules

- **Keep**: If best_auroc > previous best_auroc by at least 0.01
- **Discard**: If best_auroc <= previous best_auroc or within 0.01 margin
- **Always keep**: If this is the first experiment (no previous best)
- When keeping: copy probe_results.json to `results/experiments/best_probe.json`
- When keeping: record the configuration that produced the improvement
- After 3 consecutive discards on the same variable, move to the next variable

## Iteration Log Format

Each iteration should produce a JSON entry:
```json
{
  "iteration": 1,
  "timestamp": "2024-01-01T00:00:00",
  "variable_changed": "aggregation",
  "old_value": "mean_pool",
  "new_value": "last_token",
  "auroc": 0.72,
  "previous_best": 0.65,
  "decision": "keep"
}
```
