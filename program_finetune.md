# Autoresearch Program: Fine-tuning Recipe Discovery

## Objective

Find the LoRA fine-tuning recipe that produces a Pythia model with the best
internal representation of deception (measured by downstream probe performance
on the fine-tuned model's activations).

## Metric

**Primary metric**: Validation loss from fine-tuning (finetune/checkpoints/run_*/metrics.json → final_val_loss)
**Secondary metric**: Downstream probe AUROC (run probing on fine-tuned model's activations)
**Threshold**: Keep if val_loss < previous best. Discard otherwise.

## What to Modify

You may modify the following hyperparameters at the top of `finetune/train.py`.
Change ONE variable at a time:

1. **Learning rate** (`LEARNING_RATE`): Try 1e-4, 5e-5, 1e-5, 5e-6
2. **LoRA rank** (`LORA_RANK`): Try 4, 8, 16, 32
3. **LoRA alpha** (`LORA_ALPHA`): Try 8, 16, 32 (typically 2x rank)
4. **Batch size** (`BATCH_SIZE`): Try 2, 4, 8
5. **LoRA target modules** (`LORA_TARGET_MODULES`): Try `["query_key_value"]`, `["query_key_value", "dense"]`
6. **LoRA dropout** (`LORA_DROPOUT`): Try 0.0, 0.05, 0.1
7. **Training budget** (`BUDGET_MINUTES`): Default 5, try 10 for promising recipes

## How to Run

```bash
# Step 1: Prepare data (if not already done)
python -m finetune.prepare --dataset data/game_LATEST.json --output finetune/data/

# Step 2: Run fine-tuning
python finetune/train.py --budget_minutes 5 --data finetune/data/

# Step 3: Extract activations from fine-tuned model (for downstream probe)
python -m mi.extract --dataset data/game_LATEST.json --model finetune/checkpoints/run_LATEST/ --output data/activations_finetuned/

# Step 4: Compare base vs fine-tuned probing
python -m mi.probe --compare data/activations_base/ data/activations_finetuned/ --output results/comparison/
```

## How to Evaluate

1. Read `finetune/checkpoints/run_LATEST/metrics.json`
2. Check `final_val_loss` field
3. Optionally check `results/comparison/comparison_results.json` for probe improvement
4. Compare to previous best (stored in `results/experiments/best_finetune.json`)

## Decision Rules

- **Keep**: If final_val_loss < previous best_val_loss by at least 0.01
- **Discard**: If final_val_loss >= previous best_val_loss or within 0.01 margin
- **Always keep**: If this is the first experiment
- When keeping: copy metrics.json to `results/experiments/best_finetune.json`
- When keeping: record the hyperparameter change that produced the improvement
- After 3 consecutive discards on the same hyperparameter, move to the next one

## Iteration Log Format

Each iteration should produce a JSON entry:
```json
{
  "iteration": 1,
  "timestamp": "2024-01-01T00:00:00",
  "variable_changed": "LEARNING_RATE",
  "old_value": "5e-5",
  "new_value": "1e-4",
  "val_loss": 2.34,
  "val_perplexity": 10.38,
  "previous_best_loss": null,
  "decision": "keep"
}
```
