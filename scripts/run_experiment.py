"""Thin experiment runner for autoresearch loops.

Executes one iteration of a probe or fine-tuning experiment,
captures metrics, and logs results.

Usage:
    python scripts/run_experiment.py --type probe --results results/probe_run/probe_results.json
    python scripts/run_experiment.py --type finetune --results finetune/checkpoints/run_*/metrics.json
"""
import argparse
import json
import os
from datetime import datetime


EXPERIMENTS_DIR = "results/experiments"


def parse_probe_metrics(results_path: str) -> dict:
    """Parse probe metrics from probe_results.json."""
    with open(results_path) as f:
        data = json.load(f)
    return {
        "best_auroc": data.get("best_auroc", 0.0),
        "best_layer": data.get("best_layer", -1),
        "aggregation": data.get("aggregation", "unknown"),
        "num_layers": data.get("num_layers", 0),
        "total_samples": data.get("total_samples", 0),
    }


def parse_finetune_metrics(results_path: str) -> dict:
    """Parse fine-tuning metrics from metrics.json."""
    with open(results_path) as f:
        data = json.load(f)
    return {
        "final_val_loss": data.get("final_val_loss", float("inf")),
        "final_val_perplexity": data.get("final_val_perplexity", float("inf")),
        "total_steps": data.get("total_steps", 0),
        "total_epochs": data.get("total_epochs", 0),
        "training_time_seconds": data.get("training_time_seconds", 0),
        "hyperparameters": data.get("hyperparameters", {}),
    }


def load_previous_best(experiment_type: str) -> dict | None:
    """Load the previous best experiment result."""
    best_path = os.path.join(EXPERIMENTS_DIR, f"best_{experiment_type}.json")
    if os.path.exists(best_path):
        with open(best_path) as f:
            return json.load(f)
    return None


def save_best(experiment_type: str, metrics: dict):
    """Save as the new best experiment result."""
    best_path = os.path.join(EXPERIMENTS_DIR, f"best_{experiment_type}.json")
    with open(best_path, "w") as f:
        json.dump(metrics, f, indent=2)


def decide_keep_or_discard(
    experiment_type: str,
    current_metrics: dict,
    previous_best: dict | None,
    threshold: float = 0.01,
) -> tuple[str, str]:
    """Decide whether to keep or discard the current experiment.

    Returns:
        Tuple of (decision, reason).
    """
    if previous_best is None:
        return "keep", "First experiment — no previous best"

    if experiment_type == "probe":
        current_val = current_metrics.get("best_auroc", 0)
        prev_val = previous_best.get("best_auroc", 0)
        if current_val > prev_val + threshold:
            return "keep", f"AUROC improved: {prev_val:.3f} -> {current_val:.3f}"
        return "discard", f"AUROC not improved: {prev_val:.3f} -> {current_val:.3f}"

    elif experiment_type == "finetune":
        current_val = current_metrics.get("final_val_loss", float("inf"))
        prev_val = previous_best.get("final_val_loss", float("inf"))
        if current_val < prev_val - threshold:
            return "keep", f"Val loss improved: {prev_val:.4f} -> {current_val:.4f}"
        return "discard", f"Val loss not improved: {prev_val:.4f} -> {current_val:.4f}"

    return "discard", f"Unknown experiment type: {experiment_type}"


def log_experiment(
    experiment_type: str,
    metrics: dict,
    decision: str,
    reason: str,
    config_diff: dict = None,
):
    """Log an experiment iteration to results/experiments/."""
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "type": experiment_type,
        "config_diff": config_diff or {},
        "metrics": metrics,
        "decision": decision,
        "reason": reason,
    }

    log_path = os.path.join(EXPERIMENTS_DIR, f"experiment_{ts}.json")
    with open(log_path, "w") as f:
        json.dump(log_entry, f, indent=2)

    print(f"Experiment logged to {log_path}")
    return log_path


def run_experiment(
    experiment_type: str,
    results_path: str,
    config_diff: dict = None,
):
    """Run one experiment iteration: parse metrics, decide, log.

    Args:
        experiment_type: "probe" or "finetune"
        results_path: Path to results JSON
        config_diff: Dict describing what changed in this iteration

    Returns:
        Dict with decision and metrics.
    """
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

    # Parse metrics
    if experiment_type == "probe":
        metrics = parse_probe_metrics(results_path)
    elif experiment_type == "finetune":
        metrics = parse_finetune_metrics(results_path)
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")

    # Load previous best
    previous_best = load_previous_best(experiment_type)

    # Decide
    decision, reason = decide_keep_or_discard(experiment_type, metrics, previous_best)

    # Log
    log_path = log_experiment(experiment_type, metrics, decision, reason, config_diff)

    # Save if keeping
    if decision == "keep":
        save_best(experiment_type, metrics)
        print(f"  KEEP: {reason}")
    else:
        print(f"  DISCARD: {reason}")

    return {
        "decision": decision,
        "reason": reason,
        "metrics": metrics,
        "log_path": log_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Run one autoresearch experiment iteration")
    parser.add_argument("--type", required=True, choices=["probe", "finetune"])
    parser.add_argument("--results", required=True, help="Path to results JSON")
    parser.add_argument("--config-diff", default=None, help="JSON string describing config changes")
    args = parser.parse_args()

    config_diff = json.loads(args.config_diff) if args.config_diff else None
    run_experiment(args.type, args.results, config_diff)


if __name__ == "__main__":
    main()
