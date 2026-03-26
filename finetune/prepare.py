"""Fine-tuning data preparation with stratified splits.

Converts MI export JSON into prompt/completion pairs for HuggingFace training.

Usage:
    python -m finetune.prepare --dataset data/game.json --model pythia-410m --output finetune/data/
"""
import argparse
import json
import os
from collections import Counter

from sklearn.model_selection import train_test_split


def format_prompt(sample: dict) -> str:
    """Format game context + role into a training prompt."""
    game_state = sample.get("game_state", {})
    parts = [
        f"Game: Secret Hitler | Round {sample.get('round', '?')}",
        f"Role: {sample.get('player_role', 'Unknown')}",
        f"Action: {sample.get('action_type', 'unknown')}",
    ]
    if game_state:
        parts.append(
            f"State: Liberal={game_state.get('liberal_policies', 0)}, "
            f"Fascist={game_state.get('fascist_policies', 0)}, "
            f"Tracker={game_state.get('election_tracker', 0)}, "
            f"Alive={','.join(game_state.get('players_alive', []))}"
        )
    return "\n".join(parts)


def format_completion(sample: dict) -> str:
    """Format hidden thought + public statement as a training completion."""
    hidden = sample.get("hidden_thought", "")
    public = sample.get("public_statement", "")
    parts = []
    if hidden:
        parts.append(f"<hidden_thought>{hidden}</hidden_thought>")
    if public:
        parts.append(f"<public_statement>{public}</public_statement>")
    return "\n".join(parts)


def stratification_key(sample: dict) -> str:
    """Create stratification key from deceptive_intent and role."""
    intent = "deceptive" if sample.get("deceptive_intent", False) else "truthful"
    role = sample.get("player_role", "unknown").replace(" ", "_")
    return f"{intent}_{role}"


def prepare_dataset(
    dataset_path: str,
    output_dir: str = "finetune/data/",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
):
    """Convert MI export JSON into stratified train/val/test JSONL splits.

    Args:
        dataset_path: Path to exported game JSON.
        output_dir: Directory to save train.jsonl, val.jsonl, test.jsonl.
        train_ratio: Fraction for training (default 0.8).
        val_ratio: Fraction for validation (default 0.1).
        test_ratio: Fraction for test (default 0.1).
        seed: Random seed for reproducibility.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(dataset_path) as f:
        dataset = json.load(f)
    samples = dataset.get("samples", [])

    if not samples:
        raise ValueError("No samples found in dataset")

    # Create formatted records
    records = []
    for i, sample in enumerate(samples):
        prompt = format_prompt(sample)
        completion = format_completion(sample)
        if not completion.strip():
            continue
        records.append({
            "id": i,
            "prompt": prompt,
            "completion": completion,
            "deceptive_intent": sample.get("deceptive_intent", False),
            "deception_type": sample.get("deception_type", "UNKNOWN"),
            "player_role": sample.get("player_role", "unknown"),
            "action_type": sample.get("action_type", "unknown"),
        })

    # Create stratification labels
    strat_labels = [stratification_key(r) for r in records]

    # Check if stratification is possible (need at least 2 samples per class)
    label_counts = Counter(strat_labels)
    min_count = min(label_counts.values())

    if min_count < 3:
        # Fall back to deceptive_intent only for stratification
        strat_labels = [
            "deceptive" if r["deceptive_intent"] else "truthful"
            for r in records
        ]

    # First split: train vs (val+test)
    val_test_ratio = val_ratio + test_ratio
    train_records, val_test_records, train_labels, val_test_labels = train_test_split(
        records, strat_labels,
        test_size=val_test_ratio,
        stratify=strat_labels,
        random_state=seed,
    )

    # Second split: val vs test
    relative_test = test_ratio / val_test_ratio
    # Check if stratification is possible for the second split
    vt_label_counts = Counter(val_test_labels)
    can_stratify_vt = all(c >= 2 for c in vt_label_counts.values())
    try:
        val_records, test_records = train_test_split(
            val_test_records,
            test_size=relative_test,
            stratify=val_test_labels if can_stratify_vt else None,
            random_state=seed,
        )
    except ValueError:
        # Fallback: split without stratification if too few samples
        val_records, test_records = train_test_split(
            val_test_records,
            test_size=relative_test,
            random_state=seed,
        )

    # Save splits
    splits = {
        "train": train_records,
        "val": val_records,
        "test": test_records,
    }

    for split_name, split_records in splits.items():
        path = os.path.join(output_dir, f"{split_name}.jsonl")
        with open(path, "w") as f:
            for record in split_records:
                f.write(json.dumps(record) + "\n")

    # Print summary
    total = len(records)
    print(f"Prepared {total} samples into splits:")
    print(f"  train: {len(train_records)} ({len(train_records)/total*100:.1f}%)")
    print(f"  val:   {len(val_records)} ({len(val_records)/total*100:.1f}%)")
    print(f"  test:  {len(test_records)} ({len(test_records)/total*100:.1f}%)")

    # Label distribution per split
    for split_name, split_records in splits.items():
        deceptive = sum(1 for r in split_records if r["deceptive_intent"])
        truthful = len(split_records) - deceptive
        print(f"  {split_name}: deceptive={deceptive}, truthful={truthful}")

    return splits


def main():
    parser = argparse.ArgumentParser(description="Prepare fine-tuning data from MI export")
    parser.add_argument("--dataset", required=True, help="Path to MI export JSON")
    parser.add_argument("--model", default="pythia-410m", help="Model for tokenizer validation")
    parser.add_argument("--output", default="finetune/data/", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    prepare_dataset(
        dataset_path=args.dataset,
        output_dir=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
