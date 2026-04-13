"""Base vs. fine-tuned model comparison for deception vectors.

Compares deception vectors from two models (base and fine-tuned) across
three axes: cosine similarity, AUROC deltas, and steering effectiveness.

Usage:
    python -m mi.analysis --base-vectors results/vectors/base/ --ft-vectors results/vectors/ft/ --output results/comparison/
"""
import argparse
import json
import os
from typing import Optional

import numpy as np
import torch

from mi.vectors import DeceptionVectors, extract_vectors


def compare_vectors(
    base_vectors_path: str,
    ft_vectors_path: str,
) -> dict:
    """Compare deception vectors between base and fine-tuned models.

    Computes per-layer cosine similarity and magnitude comparison.

    Args:
        base_vectors_path: Path to base model's deception_vectors.pt.
        ft_vectors_path: Path to fine-tuned model's deception_vectors.pt.

    Returns:
        Dict with per-layer cosine similarity and magnitude comparison.
    """
    base_pt = torch.load(base_vectors_path, weights_only=False)
    ft_pt = torch.load(ft_vectors_path, weights_only=False)

    base_layers = sorted(int(k) for k in base_pt.keys())
    ft_layers = sorted(int(k) for k in ft_pt.keys())
    shared_layers = sorted(set(base_layers) & set(ft_layers))

    results = {
        "base_layers": base_layers,
        "ft_layers": ft_layers,
        "shared_layers": shared_layers,
        "base_only_layers": sorted(set(base_layers) - set(ft_layers)),
        "ft_only_layers": sorted(set(ft_layers) - set(base_layers)),
        "per_layer": {},
    }

    for layer in shared_layers:
        base_vec = base_pt[str(layer)].numpy() if isinstance(base_pt[str(layer)], torch.Tensor) else base_pt[str(layer)]
        ft_vec = ft_pt[str(layer)].numpy() if isinstance(ft_pt[str(layer)], torch.Tensor) else ft_pt[str(layer)]

        base_norm = np.linalg.norm(base_vec)
        ft_norm = np.linalg.norm(ft_vec)

        if base_norm > 0 and ft_norm > 0:
            cosine_sim = float(np.dot(base_vec, ft_vec) / (base_norm * ft_norm))
        else:
            cosine_sim = 0.0

        results["per_layer"][str(layer)] = {
            "cosine_similarity": cosine_sim,
            "base_norm": float(base_norm),
            "ft_norm": float(ft_norm),
            "norm_ratio": float(ft_norm / base_norm) if base_norm > 0 else float("inf"),
            "direction_similarity": "same" if cosine_sim > 0.8 else ("opposite" if cosine_sim < -0.5 else "different"),
        }

    cosine_values = [v["cosine_similarity"] for v in results["per_layer"].values()]
    if cosine_values:
        results["mean_cosine_similarity"] = float(np.mean(cosine_values))
        results["max_cosine_similarity"] = float(np.max(cosine_values))
        results["min_cosine_similarity"] = float(np.min(cosine_values))
    else:
        results["mean_cosine_similarity"] = 0.0
        results["max_cosine_similarity"] = 0.0
        results["min_cosine_similarity"] = 0.0

    return results


def compare_probes(
    base_vector_results_path: str,
    ft_vector_results_path: str,
) -> dict:
    """Compare vector-AUROC results between base and fine-tuned models.

    Args:
        base_vector_results_path: Path to base model's vector_results.json.
        ft_vector_results_path: Path to fine-tuned model's vector_results.json.

    Returns:
        Dict with per-layer AUROC comparison and deltas.
    """
    with open(base_vector_results_path) as f:
        base_results = json.load(f)
    with open(ft_vector_results_path) as f:
        ft_results = json.load(f)

    base_aurocs = base_results["per_layer_auroc"]
    ft_aurocs = ft_results["per_layer_auroc"]

    base_layers = set(int(k) for k in base_aurocs.keys())
    ft_layers = set(int(k) for k in ft_aurocs.keys())
    shared_layers = sorted(base_layers & ft_layers)

    results = {
        "base_model": base_results.get("model", "unknown"),
        "ft_model": ft_results.get("model", "unknown"),
        "shared_layers": shared_layers,
        "per_layer": {},
    }

    for layer in shared_layers:
        base_auroc = base_aurocs[str(layer)]
        ft_auroc = ft_aurocs[str(layer)]
        results["per_layer"][str(layer)] = {
            "base_auroc": base_auroc,
            "ft_auroc": ft_auroc,
            "delta": ft_auroc - base_auroc,
        }

    if shared_layers:
        deltas = [results["per_layer"][str(l)]["delta"] for l in shared_layers]
        results["mean_delta"] = float(np.mean(deltas))
        results["max_delta_layer"] = shared_layers[int(np.argmax(deltas))]
        results["max_delta"] = float(max(deltas))

    return results


def run_full_comparison(
    base_vectors_path: str,
    ft_vectors_path: str,
    base_vector_results_path: Optional[str] = None,
    ft_vector_results_path: Optional[str] = None,
    output_dir: str = "results/comparison/",
) -> dict:
    """Run full comparison between base and fine-tuned deception vectors.

    Combines vector comparison and (optionally) probe comparison.

    Args:
        base_vectors_path: Path to base model's deception_vectors.pt.
        ft_vectors_path: Path to fine-tuned model's deception_vectors.pt.
        base_vector_results_path: Path to base model's vector_results.json (optional).
        ft_vector_results_path: Path to fine-tuned model's vector_results.json (optional).
        output_dir: Directory to save results.

    Returns:
        Dict with all comparison results.
    """
    os.makedirs(output_dir, exist_ok=True)

    vector_comparison = compare_vectors(base_vectors_path, ft_vectors_path)

    full_results = {"vector_comparison": vector_comparison}

    if base_vector_results_path and ft_vector_results_path:
        probe_comparison = compare_probes(
            base_vector_results_path, ft_vector_results_path
        )
        full_results["probe_comparison"] = probe_comparison

    results_path = os.path.join(output_dir, "comparison_results.json")
    serializable = json.loads(json.dumps(full_results, default=str))
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"Comparison results saved to {results_path}")

    try:
        _plot_comparison(full_results, output_dir)
    except ImportError:
        print("  matplotlib not available, skipping plots")

    return full_results


def _plot_comparison(full_results: dict, output_dir: str):
    """Generate comparison plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vector_comp = full_results["vector_comparison"]

    if vector_comp["shared_layers"]:
        layers = vector_comp["shared_layers"]
        cos_sims = [vector_comp["per_layer"][str(l)]["cosine_similarity"] for l in layers]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(layers, cos_sims, color="steelblue", alpha=0.8)
        ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
        ax.axhline(y=0.8, color="green", linestyle="--", alpha=0.5, label="Similar (>0.8)")
        ax.axhline(y=-0.5, color="red", linestyle="--", alpha=0.5, label="Opposite (<-0.5)")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Cosine Similarity")
        ax.set_title("Deception Vector: Base vs. Fine-tuned Cosine Similarity")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plot_path = os.path.join(output_dir, "comparison_vectors.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Vector comparison plot saved to {plot_path}")

    if "probe_comparison" in full_results:
        probe_comp = full_results["probe_comparison"]

        if probe_comp["shared_layers"]:
            layers = probe_comp["shared_layers"]
            base_aurocs = [probe_comp["per_layer"][str(l)]["base_auroc"] for l in layers]
            ft_aurocs = [probe_comp["per_layer"][str(l)]["ft_auroc"] for l in layers]
            deltas = [probe_comp["per_layer"][str(l)]["delta"] for l in layers]

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [3, 1]})

            ax1.plot(layers, base_aurocs, "b-o", markersize=4, linewidth=1.5, label="Base")
            ax1.plot(layers, ft_aurocs, "r-s", markersize=4, linewidth=1.5, label="Fine-tuned")
            ax1.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
            ax1.set_ylabel("AUROC")
            ax1.set_title("Deception Vector AUROC: Base vs. Fine-tuned")
            ax1.legend()
            ax1.set_ylim(0, 1)
            ax1.grid(True, alpha=0.3)

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
            print(f"  AUROC comparison plot saved to {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare deception vectors between base and fine-tuned models"
    )
    parser.add_argument("--base-vectors", required=True,
                        help="Path to base model's deception_vectors.pt or directory")
    parser.add_argument("--ft-vectors", required=True,
                        help="Path to fine-tuned model's deception_vectors.pt or directory")
    parser.add_argument("--base-results", default=None,
                        help="Path to base model's vector_results.json (for AUROC comparison)")
    parser.add_argument("--ft-results", default=None,
                        help="Path to fine-tuned model's vector_results.json (for AUROC comparison)")
    parser.add_argument("--output", default="results/comparison/",
                        help="Output directory")
    args = parser.parse_args()

    def _resolve_vector_path(path):
        if os.path.isdir(path):
            return os.path.join(path, "deception_vectors.pt")
        return path

    base_vec = _resolve_vector_path(args.base_vectors)
    ft_vec = _resolve_vector_path(args.ft_vectors)

    run_full_comparison(
        base_vectors_path=base_vec,
        ft_vectors_path=ft_vec,
        base_vector_results_path=args.base_results,
        ft_vector_results_path=args.ft_results,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()