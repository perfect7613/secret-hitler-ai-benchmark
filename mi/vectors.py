"""Deception vector extraction using Anthropic-style difference-of-means methodology.

Extracts deception concept vectors from pre-extracted activations, evaluates them
via projection AUROC, and validates them through logit lens analysis.

Inspired by: "Emotion Concepts and their Function in a Large Language Model"
(Sofroniew et al., 2026) — adapted for deception detection in game-playing LLMs.

Usage:
    python -m mi.vectors --activations data/activations/ --metadata data/activations/metadata_pythia-410m.json --output results/vectors/
    python -m mi.vectors --activations data/activations/ --model pythia-410m --output results/vectors/
"""
import argparse
import json
import os
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


@dataclass
class DeceptionVectors:
    vectors: dict[int, np.ndarray]
    layer_indices: list[int]
    hidden_dim: int
    positive_mean: dict[int, np.ndarray]
    negative_mean: dict[int, np.ndarray]
    per_layer_auroc: dict[int, float]
    best_layer: int
    best_auroc: float
    label_field: str
    num_positive: int
    num_negative: int
    metadata: dict


def extract_vectors(
    activations_path: str,
    metadata_path: str,
    label_field: str = "deceptive_intent",
    output_dir: Optional[str] = None,
    model_name: Optional[str] = None,
    top_k_logit_lens: int = 20,
) -> DeceptionVectors:
    """Extract deception concept vectors using difference-of-means.

    For each layer, computes:
        vector = mean(positive activations) - mean(negative activations)
    then L2-normalizes and evaluates via projection AUROC.

    Args:
        activations_path: Path to .pt file with activations.
        metadata_path: Path to metadata JSON sidecar.
        label_field: Field to use for positive/negative split.
            'deceptive_intent' (bool) for binary,
            'deception_type' (str) for future multi-class.
        output_dir: If provided, save results to this directory.
        model_name: Model name for logit lens (loads unembedding matrix).
            If None, skip logit lens.
        top_k_logit_lens: Number of top/bottom tokens for logit lens.

    Returns:
        DeceptionVectors dataclass with vectors, AUROC scores, and metadata.
    """
    activations = torch.load(activations_path, weights_only=False)
    with open(metadata_path) as f:
        metadata = json.load(f)

    layers = metadata["layer_indices"]
    samples = metadata["samples"]
    hidden_dim = metadata.get("hidden_dim", activations[0][layers[0]].shape[-1])

    if label_field == "deceptive_intent":
        labels = np.array([bool(s[label_field]) for s in samples], dtype=bool)
    else:
        label_values = [s.get(label_field, "UNKNOWN") for s in samples]
        labels = np.array([v == "DECEPTIVE" for v in label_values], dtype=bool)

    num_positive = int(labels.sum())
    num_negative = int((~labels).sum())

    if num_positive < 5 or num_negative < 5:
        warnings.warn(
            f"Few samples per class: positive={num_positive}, negative={num_negative}. "
            f"Results may be unreliable. Consider collecting more data."
        )

    if num_positive == 0 or num_negative == 0:
        raise ValueError(
            f"Cannot extract vectors: need both classes. "
            f"positive={num_positive}, negative={num_negative}"
        )

    positive_indices = np.where(labels)[0]
    negative_indices = np.where(~labels)[0]

    vectors = {}
    positive_means = {}
    negative_means = {}
    per_layer_auroc = {}

    for layer in layers:
        last_token_acts = _get_last_token_activations(activations, layer)

        positive_mean = last_token_acts[positive_indices].mean(axis=0)
        negative_mean = last_token_acts[negative_indices].mean(axis=0)

        positive_means[layer] = positive_mean
        negative_means[layer] = negative_mean

        raw_vector = positive_mean - negative_mean
        vector_norm = np.linalg.norm(raw_vector)
        if vector_norm > 0:
            normalized_vector = raw_vector / vector_norm
        else:
            normalized_vector = raw_vector
        vectors[layer] = normalized_vector

        projections = last_token_acts @ normalized_vector

        if len(np.unique(labels)) >= 2:
            auroc = float(roc_auc_score(labels, projections))
        else:
            auroc = 0.5
        per_layer_auroc[layer] = auroc

    best_layer = max(per_layer_auroc, key=per_layer_auroc.get)
    best_auroc = per_layer_auroc[best_layer]

    result = DeceptionVectors(
        vectors=vectors,
        layer_indices=layers,
        hidden_dim=hidden_dim,
        positive_mean=positive_means,
        negative_mean=negative_means,
        per_layer_auroc=per_layer_auroc,
        best_layer=best_layer,
        best_auroc=best_auroc,
        label_field=label_field,
        num_positive=num_positive,
        num_negative=num_negative,
        metadata=metadata,
    )

    if output_dir:
        _save_vectors(result, output_dir)

        if model_name:
            logit_lens_results = logit_lens(result, model_name, top_k=top_k_logit_lens)
            _save_logit_lens(logit_lens_results, output_dir)

    return result


def _get_last_token_activations(activations: list, layer: int) -> np.ndarray:
    """Extract last-token activations for a given layer across all samples.

    Args:
        activations: List of dicts mapping layer -> tensor [seq_len, hidden_dim].
        layer: Layer index to extract.

    Returns:
        numpy array of shape [num_samples, hidden_dim].
    """
    vectors = []
    for act_dict in activations:
        tensor = act_dict[layer]
        last_token = tensor[-1].numpy()
        vectors.append(last_token)
    return np.stack(vectors)


def logit_lens(
    deception_vectors: DeceptionVectors,
    model_name: str,
    top_k: int = 20,
) -> dict:
    """Project deception vectors through the model's unembedding matrix.

    For each layer, projects the normalized deception vector through W_U
    (the unembedding matrix) to find which tokens are most upweighted
    and downweighted by the vector.

    Args:
        deception_vectors: DeceptionVectors dataclass with per-layer vectors.
        model_name: TransformerLens model name (e.g., 'pythia-410m').
        top_k: Number of top/bottom tokens to return per layer.

    Returns:
        Dict with per-layer top-k upweighted and downweighted tokens.
    """
    from transformer_lens import HookedTransformer

    resolved = PYTHIA_MODELS.get(model_name, model_name)
    model = HookedTransformer.from_pretrained(resolved)
    model.eval()

    W_U = model.W_U.detach().cpu().numpy()
    vocab_size = W_U.shape[1] if W_U.shape[0] < W_U.shape[1] else W_U.shape[0]

    results = {"model": model_name, "layers": {}, "top_k": top_k}

    for layer, vector in deception_vectors.vectors.items():
        logits = vector @ W_U

        if W_U.shape[0] < W_U.shape[1]:
            logits_for_topk = logits
        else:
            logits_for_topk = logits

        top_indices = np.argsort(logits_for_topk)[-top_k:][::-1]
        bottom_indices = np.argsort(logits_for_topk)[:top_k]

        top_tokens = []
        for idx in top_indices:
            token_str = model.tokenizer.decode([int(idx)])
            score = float(logits_for_topk[idx])
            top_tokens.append({"token": token_str, "token_id": int(idx), "score": score})

        bottom_tokens = []
        for idx in bottom_indices:
            token_str = model.tokenizer.decode([int(idx)])
            score = float(logits_for_topk[idx])
            bottom_tokens.append({"token": token_str, "token_id": int(idx), "score": score})

        results["layers"][str(layer)] = {
            "top_tokens": top_tokens,
            "bottom_tokens": bottom_tokens,
        }

    return results


PYTHIA_MODELS = {
    "pythia-410m": "EleutherAI/pythia-410m",
    "pythia-1.4b": "EleutherAI/pythia-1.4b",
    "pythia-70m": "EleutherAI/pythia-70m",
}


def _save_vectors(result: DeceptionVectors, output_dir: str):
    """Save vectors and results to output directory."""
    os.makedirs(output_dir, exist_ok=True)

    vectors_torch = {str(k): torch.tensor(v) for k, v in result.vectors.items()}
    torch.save(vectors_torch, os.path.join(output_dir, "deception_vectors.pt"))

    results_json = {
        "model": result.metadata.get("model", "unknown"),
        "label_field": result.label_field,
        "layer_indices": result.layer_indices,
        "hidden_dim": result.hidden_dim,
        "num_positive": result.num_positive,
        "num_negative": result.num_negative,
        "per_layer_auroc": {str(k): v for k, v in result.per_layer_auroc.items()},
        "best_layer": result.best_layer,
        "best_auroc": result.best_auroc,
        "vector_norms": {
            str(k): float(np.linalg.norm(v))
            for k, v in result.vectors.items()
        },
    }

    results_path = os.path.join(output_dir, "vector_results.json")
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)

    try:
        _plot_vector_auroc(result, output_dir)
    except ImportError:
        print("matplotlib not available, skipping plot")

    print(f"Vectors saved to {output_dir}")
    print(f"  Best layer: {result.best_layer} (AUROC={result.best_auroc:.3f})")
    print(f"  Positive samples: {result.num_positive}, Negative: {result.num_negative}")


def _save_logit_lens(logit_lens_results: dict, output_dir: str):
    """Save logit lens results to output directory."""
    lens_path = os.path.join(output_dir, "logit_lens.json")
    with open(lens_path, "w") as f:
        json.dump(logit_lens_results, f, indent=2)
    print(f"Logit lens results saved to {lens_path}")


def _plot_vector_auroc(result: DeceptionVectors, output_dir: str):
    """Generate AUROC vs layer number plot for vector-based probing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = result.layer_indices
    aurocs = [result.per_layer_auroc[l] for l in layers]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(layers, aurocs, "b-o", markersize=4, linewidth=1.5, label="Vector AUROC")
    ax.axhline(y=0.5, color="r", linestyle="--", alpha=0.5, label="Chance")

    best_layer = result.best_layer
    best_auroc = result.best_auroc
    ax.scatter([best_layer], [best_auroc], color="red", s=100, zorder=5,
               label=f"Best: L{best_layer} ({best_auroc:.3f})")

    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC (vector projection)")
    ax.set_title(f"Deception Vector AUROC by Layer ({result.label_field})")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plot_path = os.path.join(output_dir, "vector_auroc_curve.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved to {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract deception vectors using difference-of-means"
    )
    parser.add_argument("--activations", required=True,
                        help="Path to activations directory or .pt file")
    parser.add_argument("--metadata", default=None,
                        help="Path to metadata JSON (auto-detected if omitted)")
    parser.add_argument("--model", default=None,
                        help="Model name for logit lens (e.g., pythia-410m). Skips logit lens if omitted.")
    parser.add_argument("--label", default="deceptive_intent",
                        help="Label field: deceptive_intent (binary) or deception_type")
    parser.add_argument("--output", default="results/vectors/",
                        help="Output directory")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Number of top/bottom tokens for logit lens")
    args = parser.parse_args()

    act_path, meta_path = _resolve_paths(args.activations, args.metadata)
    extract_vectors(
        activations_path=act_path,
        metadata_path=meta_path,
        label_field=args.label,
        output_dir=args.output,
        model_name=args.model,
        top_k_logit_lens=args.top_k,
    )


def _resolve_paths(act_input: str, metadata_input: Optional[str] = None):
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

    if metadata_input:
        m = metadata_input

    return a, m


if __name__ == "__main__":
    main()