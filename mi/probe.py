"""Linear probing module for deception detection.

Trains logistic regression probes on extracted activations to detect
deception directions in activation space.

Usage:
    python -m mi.probe --activations data/activations/ --aggregation mean_pool --output results/
"""
import argparse
import json
import os

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def aggregate_activations(
    activations: list[dict[int, torch.Tensor]],
    layer: int,
    method: str = "mean_pool",
) -> np.ndarray:
    """Aggregate token-level activations into a single vector per interaction.

    Args:
        activations: List of dicts mapping layer -> tensor [seq_len, hidden_dim].
        layer: Layer index to extract.
        method: "mean_pool" (average across tokens) or "last_token" (final token only).

    Returns:
        numpy array of shape [num_interactions, hidden_dim].
    """
    vectors = []
    for act_dict in activations:
        tensor = act_dict[layer]  # [seq_len, hidden_dim]
        if method == "mean_pool":
            vec = tensor.mean(dim=0)  # [hidden_dim]
        elif method == "last_token":
            vec = tensor[-1]  # [hidden_dim]
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        vectors.append(vec.numpy())
    return np.stack(vectors)


def train_probe(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict:
    """Train a logistic regression probe and evaluate.

    Args:
        X: Feature matrix [n_samples, hidden_dim].
        y: Binary labels [n_samples].
        test_size: Fraction for test split.
        seed: Random seed.

    Returns:
        Dict with auroc, accuracy, f1, and trained model.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed,
    )

    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    return {
        "auroc": float(roc_auc_score(y_test, y_prob)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "model": clf,
        "n_train": len(y_train),
        "n_test": len(y_test),
    }


def run_probing(
    activations_path: str,
    metadata_path: str,
    aggregation: str = "mean_pool",
    output_dir: str = "results/",
    label_field: str = "deceptive_intent",
    seed: int = 42,
):
    """Run linear probing across all layers.

    Args:
        activations_path: Path to .pt file with activations.
        metadata_path: Path to metadata JSON sidecar.
        aggregation: "mean_pool" or "last_token".
        output_dir: Directory for results JSON and plots.
        label_field: Which label to probe for (default: deceptive_intent).
        seed: Random seed.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    activations = torch.load(activations_path, weights_only=False)
    with open(metadata_path) as f:
        metadata = json.load(f)

    layers = metadata["layer_indices"]
    samples = metadata["samples"]

    # Extract labels
    labels = np.array([s[label_field] for s in samples], dtype=np.float64)

    # Check we have both classes
    if len(np.unique(labels)) < 2:
        print(f"Warning: only one class present in labels. Skipping probing.")
        return {}

    print(f"Running probing: {len(activations)} samples, {len(layers)} layers, "
          f"aggregation={aggregation}, label={label_field}")
    print(f"  Class balance: {labels.sum():.0f} positive, {(1 - labels).sum():.0f} negative")

    # Probe each layer
    results = {}
    for layer in layers:
        X = aggregate_activations(activations, layer, method=aggregation)
        probe_result = train_probe(X, labels, seed=seed)
        results[layer] = {
            "auroc": probe_result["auroc"],
            "accuracy": probe_result["accuracy"],
            "f1": probe_result["f1"],
            "n_train": probe_result["n_train"],
            "n_test": probe_result["n_test"],
        }
        print(f"  Layer {layer:2d}: AUROC={probe_result['auroc']:.3f}, "
              f"Acc={probe_result['accuracy']:.3f}, F1={probe_result['f1']:.3f}")

    # Save results
    results_output = {
        "model": metadata.get("model", "unknown"),
        "aggregation": aggregation,
        "label_field": label_field,
        "total_samples": len(activations),
        "num_layers": len(layers),
        "per_layer_metrics": {str(k): v for k, v in results.items()},
        "best_layer": max(results, key=lambda k: results[k]["auroc"]),
        "best_auroc": max(r["auroc"] for r in results.values()),
    }

    results_path = os.path.join(output_dir, "probe_results.json")
    with open(results_path, "w") as f:
        json.dump(results_output, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Generate plot
    try:
        _plot_auroc_curve(layers, results, output_dir, aggregation)
    except ImportError:
        print("  matplotlib not available, skipping plot")

    return results


def _plot_auroc_curve(layers, results, output_dir, aggregation):
    """Generate AUROC vs layer number plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    aurocs = [results[layer]["auroc"] for layer in layers]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(layers, aurocs, "b-o", markersize=4, linewidth=1.5)
    ax.axhline(y=0.5, color="r", linestyle="--", alpha=0.5, label="Chance")
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC")
    ax.set_title(f"Deception Probe AUROC by Layer ({aggregation})")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plot_path = os.path.join(output_dir, "probe_auroc_curve.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved to {plot_path}")


def compare_models(
    base_activations_path: str,
    base_metadata_path: str,
    finetuned_activations_path: str,
    finetuned_metadata_path: str,
    aggregation: str = "mean_pool",
    output_dir: str = "results/",
    label_field: str = "deceptive_intent",
    seed: int = 42,
):
    """Compare probe results between base and fine-tuned models.

    Runs probes on both activation sets, computes per-layer AUROC delta,
    and generates comparative visualization.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=== Base Model Probing ===")
    base_results = run_probing(
        base_activations_path, base_metadata_path,
        aggregation=aggregation, output_dir=os.path.join(output_dir, "base"),
        label_field=label_field, seed=seed,
    )

    print("\n=== Fine-tuned Model Probing ===")
    ft_results = run_probing(
        finetuned_activations_path, finetuned_metadata_path,
        aggregation=aggregation, output_dir=os.path.join(output_dir, "finetuned"),
        label_field=label_field, seed=seed,
    )

    # Compute deltas on shared layers
    shared_layers = sorted(set(base_results.keys()) & set(ft_results.keys()))
    comparison = {}
    for layer in shared_layers:
        base_auroc = base_results[layer]["auroc"]
        ft_auroc = ft_results[layer]["auroc"]
        comparison[layer] = {
            "base_auroc": base_auroc,
            "finetuned_auroc": ft_auroc,
            "delta": ft_auroc - base_auroc,
        }

    # Find best improvement
    if comparison:
        best_layer = max(comparison, key=lambda k: comparison[k]["delta"])
        max_delta = comparison[best_layer]["delta"]
    else:
        best_layer = None
        max_delta = 0.0

    # Save comparison results
    comparison_output = {
        "aggregation": aggregation,
        "label_field": label_field,
        "shared_layers": shared_layers,
        "base_only_layers": sorted(set(base_results.keys()) - set(ft_results.keys())),
        "finetuned_only_layers": sorted(set(ft_results.keys()) - set(base_results.keys())),
        "per_layer_comparison": {str(k): v for k, v in comparison.items()},
        "best_improvement_layer": best_layer,
        "max_auroc_delta": max_delta,
    }

    comp_path = os.path.join(output_dir, "comparison_results.json")
    with open(comp_path, "w") as f:
        json.dump(comparison_output, f, indent=2)

    print(f"\n=== Comparison Summary ===")
    print(f"  Shared layers: {len(shared_layers)}")
    if best_layer is not None:
        print(f"  Best improvement: Layer {best_layer} (delta={max_delta:+.3f})")
    print(f"  Results saved to {comp_path}")

    # Generate comparison plot
    try:
        _plot_comparison(shared_layers, comparison, output_dir, aggregation)
    except ImportError:
        print("  matplotlib not available, skipping plot")

    return comparison


def _plot_comparison(layers, comparison, output_dir, aggregation):
    """Generate comparative AUROC plot with both models."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base_aurocs = [comparison[l]["base_auroc"] for l in layers]
    ft_aurocs = [comparison[l]["finetuned_auroc"] for l in layers]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [3, 1]})

    # Top: Both AUROC curves
    ax1.plot(layers, base_aurocs, "b-o", markersize=4, linewidth=1.5, label="Base")
    ax1.plot(layers, ft_aurocs, "r-s", markersize=4, linewidth=1.5, label="Fine-tuned")
    ax1.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax1.set_ylabel("AUROC")
    ax1.set_title(f"Deception Probe: Base vs Fine-tuned ({aggregation})")
    ax1.legend()
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # Bottom: Delta
    deltas = [comparison[l]["delta"] for l in layers]
    colors = ["green" if d > 0 else "red" for d in deltas]
    ax2.bar(layers, deltas, color=colors, alpha=0.7)
    ax2.axhline(y=0, color="gray", linestyle="-", alpha=0.5)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("AUROC Delta")
    ax2.set_title("Fine-tuned - Base AUROC")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "comparison_auroc.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Comparison plot saved to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Train linear probes on extracted activations")
    parser.add_argument("--activations", help="Path to activations directory or .pt file")
    parser.add_argument("--metadata", default=None, help="Path to metadata JSON (auto-detected if omitted)")
    parser.add_argument("--aggregation", default="mean_pool", choices=["mean_pool", "last_token"])
    parser.add_argument("--output", default="results/", help="Output directory")
    parser.add_argument("--label", default="deceptive_intent", help="Label field to probe")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compare", nargs=2, metavar=("BASE_DIR", "FINETUNED_DIR"),
                        help="Compare two activation directories: base vs fine-tuned")
    args = parser.parse_args()

    def _resolve_paths(act_input):
        """Resolve activation and metadata paths from input."""
        if act_input.endswith(".pt"):
            a = act_input
            m = a.replace("activations_", "metadata_").replace(".pt", ".json")
        else:
            pt_files = [f for f in os.listdir(act_input) if f.startswith("activations_") and f.endswith(".pt")]
            if not pt_files:
                raise FileNotFoundError(f"No activation .pt files in {act_input}")
            a = os.path.join(act_input, pt_files[0])
            m = os.path.join(act_input, pt_files[0].replace("activations_", "metadata_").replace(".pt", ".json"))
        return a, m

    if args.compare:
        base_act, base_meta = _resolve_paths(args.compare[0])
        ft_act, ft_meta = _resolve_paths(args.compare[1])
        compare_models(
            base_activations_path=base_act,
            base_metadata_path=base_meta,
            finetuned_activations_path=ft_act,
            finetuned_metadata_path=ft_meta,
            aggregation=args.aggregation,
            output_dir=args.output,
            label_field=args.label,
            seed=args.seed,
        )
    else:
        if not args.activations:
            parser.error("--activations is required when not using --compare")
        act_path = args.activations
        if args.metadata:
            meta_path = args.metadata
        else:
            act_path, meta_path = _resolve_paths(act_path)

        run_probing(
            activations_path=act_path,
            metadata_path=meta_path,
            aggregation=args.aggregation,
            output_dir=args.output,
            label_field=args.label,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
