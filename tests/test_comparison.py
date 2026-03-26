"""Tests for base vs. fine-tuned model probe comparison (Issue #10)."""
import json
import os
import tempfile

import numpy as np
import pytest
import torch

from mi.probe import compare_models


def make_activations_and_metadata(
    n_samples=100, layers=None, hidden_dim=32, separable=False, seed=42,
):
    """Create activations .pt and metadata .json files."""
    np.random.seed(seed)
    if layers is None:
        layers = [0, 5, 10]

    activations = []
    labels = []
    for i in range(n_samples):
        label = i % 2
        labels.append(label)
        act_dict = {}
        for layer in layers:
            t = torch.randn(20, hidden_dim)
            if separable:
                t[:, 0] += 3.0 if label == 1 else -3.0
            act_dict[layer] = t
        activations.append(act_dict)

    metadata = {
        "model": "test",
        "layer_indices": layers,
        "hidden_dim": hidden_dim,
        "total_samples": n_samples,
        "skipped_samples": 0,
        "samples": [
            {
                "interaction_id": i,
                "deceptive_intent": bool(labels[i]),
                "deception_type": "DECEPTIVE" if labels[i] else "TRUTHFUL",
                "action_type": "discussion",
                "player_role": "a Fascist" if labels[i] else "a Liberal",
                "num_tokens": 20,
                "layer_indices": layers,
            }
            for i in range(n_samples)
        ],
    }
    return activations, metadata


def save_to_dir(tmpdir, name, activations, metadata):
    """Save activations and metadata to a subdirectory."""
    d = os.path.join(tmpdir, name)
    os.makedirs(d, exist_ok=True)
    act_path = os.path.join(d, f"activations_{name}.pt")
    meta_path = os.path.join(d, f"metadata_{name}.json")
    torch.save(activations, act_path)
    with open(meta_path, "w") as f:
        json.dump(metadata, f)
    return act_path, meta_path


class TestCompareModels:
    def test_comparison_produces_valid_deltas(self):
        """Comparison should produce per-layer delta metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_acts, base_meta = make_activations_and_metadata(
                n_samples=100, separable=False, seed=42
            )
            ft_acts, ft_meta = make_activations_and_metadata(
                n_samples=100, separable=True, seed=42
            )

            base_act_path, base_meta_path = save_to_dir(tmpdir, "base", base_acts, base_meta)
            ft_act_path, ft_meta_path = save_to_dir(tmpdir, "finetuned", ft_acts, ft_meta)

            output_dir = os.path.join(tmpdir, "results")
            comparison = compare_models(
                base_act_path, base_meta_path,
                ft_act_path, ft_meta_path,
                output_dir=output_dir,
            )

            assert len(comparison) == 3  # 3 shared layers
            for layer, metrics in comparison.items():
                assert "base_auroc" in metrics
                assert "finetuned_auroc" in metrics
                assert "delta" in metrics
                assert abs(metrics["delta"] - (metrics["finetuned_auroc"] - metrics["base_auroc"])) < 1e-6

    def test_finetuned_improves_on_separable(self):
        """Fine-tuned (separable) should have higher AUROC than base (random)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_acts, base_meta = make_activations_and_metadata(
                n_samples=100, separable=False, seed=42
            )
            ft_acts, ft_meta = make_activations_and_metadata(
                n_samples=100, separable=True, seed=42
            )

            base_act_path, base_meta_path = save_to_dir(tmpdir, "base", base_acts, base_meta)
            ft_act_path, ft_meta_path = save_to_dir(tmpdir, "finetuned", ft_acts, ft_meta)

            output_dir = os.path.join(tmpdir, "results")
            comparison = compare_models(
                base_act_path, base_meta_path,
                ft_act_path, ft_meta_path,
                output_dir=output_dir,
            )

            # At least one layer should show positive delta
            max_delta = max(m["delta"] for m in comparison.values())
            assert max_delta > 0, "Fine-tuned should improve on at least one layer"

    def test_comparison_json_schema(self):
        """comparison_results.json should have correct schema."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_acts, base_meta = make_activations_and_metadata(n_samples=60, seed=1)
            ft_acts, ft_meta = make_activations_and_metadata(n_samples=60, seed=2)

            base_act_path, base_meta_path = save_to_dir(tmpdir, "base", base_acts, base_meta)
            ft_act_path, ft_meta_path = save_to_dir(tmpdir, "finetuned", ft_acts, ft_meta)

            output_dir = os.path.join(tmpdir, "results")
            compare_models(base_act_path, base_meta_path, ft_act_path, ft_meta_path, output_dir=output_dir)

            comp_path = os.path.join(output_dir, "comparison_results.json")
            assert os.path.exists(comp_path)

            with open(comp_path) as f:
                data = json.load(f)

            assert "per_layer_comparison" in data
            assert "best_improvement_layer" in data
            assert "max_auroc_delta" in data
            assert "shared_layers" in data
            assert "aggregation" in data

    def test_different_layer_counts(self):
        """Should handle activation sets with different layers gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_acts, base_meta = make_activations_and_metadata(
                n_samples=60, layers=[0, 5, 10, 15], seed=1
            )
            ft_acts, ft_meta = make_activations_and_metadata(
                n_samples=60, layers=[0, 5, 10], seed=2
            )

            base_act_path, base_meta_path = save_to_dir(tmpdir, "base", base_acts, base_meta)
            ft_act_path, ft_meta_path = save_to_dir(tmpdir, "finetuned", ft_acts, ft_meta)

            output_dir = os.path.join(tmpdir, "results")
            comparison = compare_models(base_act_path, base_meta_path, ft_act_path, ft_meta_path, output_dir=output_dir)

            # Should only compare shared layers (0, 5, 10)
            assert len(comparison) == 3
            assert 15 not in comparison

            with open(os.path.join(output_dir, "comparison_results.json")) as f:
                data = json.load(f)
            assert 15 in data["base_only_layers"]
