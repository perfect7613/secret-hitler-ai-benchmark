"""Tests for base vs. fine-tuned vector comparison (Issue #16).

Verifies cosine similarity calculation, AUROC delta computation,
shared-layer handling, and comparison JSON schema.
Uses synthetic vectors — no model downloads needed.
"""
import json
import os
import tempfile

import numpy as np
import pytest
import torch

from mi.analysis import compare_vectors, compare_probes, run_full_comparison


def _make_vectors(layers, hidden_dim, direction="same", scale=1.0):
    """Create synthetic deception vectors.

    Args:
        direction: 'same' (parallel), 'opposite' (anti-parallel),
                   'orthogonal' (perpendicular), 'random'.
        scale: Scale factor for magnitude.
    """
    vectors = {}
    for layer in layers:
        if direction == "same":
            base = np.random.randn(hidden_dim)
            base /= np.linalg.norm(base)
            vectors[layer] = scale * base
        elif direction == "opposite":
            base = np.random.randn(hidden_dim)
            base /= np.linalg.norm(base)
            vectors[layer] = -scale * base
        elif direction == "orthogonal":
            base = np.random.randn(hidden_dim)
            base /= np.linalg.norm(base)
            vectors[layer] = scale * base
        elif direction == "random":
            vec = np.random.randn(hidden_dim)
            vec /= np.linalg.norm(vec)
            vectors[layer] = scale * vec
    return vectors


def _save_vectors(vectors, tmpdir, prefix="model"):
    """Save vectors to a .pt file, return path."""
    pt_data = {str(k): torch.tensor(v) for k, v in vectors.items()}
    path = os.path.join(tmpdir, f"deception_vectors_{prefix}.pt")
    torch.save(pt_data, path)
    return path


def _make_vector_results(layers, auroc_range=(0.6, 0.9), model="test-model"):
    """Create synthetic vector_results.json."""
    aurocs = {}
    for i, layer in enumerate(layers):
        t = i / max(len(layers) - 1, 1)
        auroc = auroc_range[0] + t * (auroc_range[1] - auroc_range[0])
        aurocs[str(layer)] = round(auroc, 4)
    best_layer = max(aurocs, key=aurocs.get)
    return {
        "model": model,
        "label_field": "deceptive_intent",
        "layer_indices": layers,
        "hidden_dim": 64,
        "num_positive": 50,
        "num_negative": 50,
        "per_layer_auroc": aurocs,
        "best_layer": int(best_layer),
        "best_auroc": aurocs[best_layer],
        "vector_norms": {str(l): 1.0 for l in layers},
    }


class TestCompareVectors:
    def test_parallel_vectors_high_cosine_similarity(self):
        """Vectors in the same direction should have cosine similarity ~1.0."""
        layers = [0, 5, 10]
        hidden_dim = 64
        base = _make_vectors(layers, hidden_dim, direction="same")

        np.random.seed(42)
        base_dir = np.random.randn(hidden_dim)
        base_dir /= np.linalg.norm(base_dir)
        base_parallel = {l: base_dir.copy() * 1.0 for l in layers}
        ft_parallel = {l: base_dir.copy() * 0.95 for l in layers}

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = _save_vectors(base_parallel, tmpdir, "base")
            ft_path = _save_vectors(ft_parallel, tmpdir, "ft")

            results = compare_vectors(base_path, ft_path)

            for layer in layers:
                cos_sim = results["per_layer"][str(layer)]["cosine_similarity"]
                assert cos_sim > 0.99, f"Layer {layer}: cosine sim {cos_sim} should be ~1.0"

    def test_opposite_vectors_negative_cosine_similarity(self):
        """Vectors in opposite directions should have cosine similarity ~-1.0."""
        layers = [0, 5, 10]
        hidden_dim = 64
        np.random.seed(42)
        base_dir = np.random.randn(hidden_dim)
        base_dir /= np.linalg.norm(base_dir)

        base_vecs = {l: base_dir.copy() for l in layers}
        ft_vecs = {l: -base_dir.copy() for l in layers}

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = _save_vectors(base_vecs, tmpdir, "base")
            ft_path = _save_vectors(ft_vecs, tmpdir, "ft")

            results = compare_vectors(base_path, ft_path)

            for layer in layers:
                cos_sim = results["per_layer"][str(layer)]["cosine_similarity"]
                assert cos_sim < -0.99, f"Layer {layer}: cosine sim {cos_sim} should be ~-1.0"

    def test_orthogonal_vectors_near_zero_cosine_similarity(self):
        """Orthogonal vectors should have cosine similarity ~0.0."""
        layers = [0, 5, 10]
        hidden_dim = 128

        np.random.seed(42)
        results_per_layer = []

        base_vecs = {}
        ft_vecs = {}
        for layer in layers:
            v1 = np.random.randn(hidden_dim)
            v1 /= np.linalg.norm(v1)
            v2 = np.random.randn(hidden_dim)
            v2 /= np.linalg.norm(v2)
            v2 -= np.dot(v2, v1) * v1
            v2 /= np.linalg.norm(v2)
            base_vecs[layer] = v1
            ft_vecs[layer] = v2

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = _save_vectors(base_vecs, tmpdir, "base")
            ft_path = _save_vectors(ft_vecs, tmpdir, "ft")

            results = compare_vectors(base_path, ft_path)

            for layer in layers:
                cos_sim = abs(results["per_layer"][str(layer)]["cosine_similarity"])
                assert cos_sim < 0.15, f"Layer {layer}: cosine sim {cos_sim} should be ~0.0"

    def test_magnitude_comparison(self):
        """Should compare vector magnitudes between models."""
        layers = [0, 1]
        hidden_dim = 32

        np.random.seed(42)
        base_dir = np.random.randn(hidden_dim)
        base_dir /= np.linalg.norm(base_dir)

        base_vecs = {l: base_dir.copy() * 1.0 for l in layers}
        ft_vecs = {l: base_dir.copy() * 2.0 for l in layers}

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = _save_vectors(base_vecs, tmpdir, "base")
            ft_path = _save_vectors(ft_vecs, tmpdir, "ft")

            results = compare_vectors(base_path, ft_path)

            for layer in layers:
                result = results["per_layer"][str(layer)]
                assert abs(result["norm_ratio"] - 2.0) < 0.1

    def test_different_layer_counts(self):
        """Should compare only shared layers when models have different depths."""
        base_layers = [0, 5, 10, 15]
        ft_layers = [0, 5, 10]
        hidden_dim = 32

        np.random.seed(42)
        base_dir = np.random.randn(hidden_dim)
        base_dir /= np.linalg.norm(base_dir)

        base_vecs = {l: base_dir.copy() for l in base_layers}
        ft_vecs = {l: base_dir.copy() * 0.9 for l in ft_layers}

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = _save_vectors(base_vecs, tmpdir, "base")
            ft_path = _save_vectors(ft_vecs, tmpdir, "ft")

            results = compare_vectors(base_path, ft_path)

            assert set(results["shared_layers"]) == {0, 5, 10}
            assert set(results["base_only_layers"]) == {15}
            assert set(results["ft_only_layers"]) == set()

    def test_mean_cosine_similarity(self):
        """Should compute mean/min/max cosine similarity across layers."""
        layers = [0, 1, 2]
        hidden_dim = 32

        np.random.seed(42)
        base_dir = np.random.randn(hidden_dim)
        base_dir /= np.linalg.norm(base_dir)

        base_vecs = {l: base_dir.copy() for l in layers}
        ft_vecs = {l: base_dir.copy() * 1.1 for l in layers}

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = _save_vectors(base_vecs, tmpdir, "base")
            ft_path = _save_vectors(ft_vecs, tmpdir, "ft")

            results = compare_vectors(base_path, ft_path)

            assert "mean_cosine_similarity" in results
            assert "max_cosine_similarity" in results
            assert "min_cosine_similarity" in results
            assert results["mean_cosine_similarity"] > 0.9


class TestCompareProbes:
    def test_auroc_delta_computation(self):
        """Should compute per-layer AUROC deltas."""
        layers = [0, 5, 10]

        base_results = _make_vector_results(layers, auroc_range=(0.6, 0.75), model="base")
        ft_results = _make_vector_results(layers, auroc_range=(0.7, 0.85), model="ft")

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = os.path.join(tmpdir, "base_vector_results.json")
            ft_path = os.path.join(tmpdir, "ft_vector_results.json")

            with open(base_path, "w") as f:
                json.dump(base_results, f)
            with open(ft_path, "w") as f:
                json.dump(ft_results, f)

            results = compare_probes(base_path, ft_path)

            assert "per_layer" in results
            for layer in layers:
                delta = results["per_layer"][str(layer)]["delta"]
                assert delta > 0, f"Fine-tuned should have higher AUROC, got delta={delta}"

    def test_shared_layers_only(self):
        """Should only compare shared layers."""
        base_layers = [0, 5, 10, 15]
        ft_layers = [0, 5, 10]

        base_results = _make_vector_results(base_layers, model="base")
        ft_results = _make_vector_results(ft_layers, model="ft")

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = os.path.join(tmpdir, "base_vector_results.json")
            ft_path = os.path.join(tmpdir, "ft_vector_results.json")

            with open(base_path, "w") as f:
                json.dump(base_results, f)
            with open(ft_path, "w") as f:
                json.dump(ft_results, f)

            results = compare_probes(base_path, ft_path)
            assert set(results["shared_layers"]) == {0, 5, 10}


class TestRunFullComparison:
    def test_saves_comparison_results_json(self):
        """Should save comparison_results.json."""
        layers = [0, 5, 10]
        hidden_dim = 32

        np.random.seed(42)
        base_dir = np.random.randn(hidden_dim)
        base_dir /= np.linalg.norm(base_dir)

        base_vecs = {l: base_dir.copy() for l in layers}
        ft_vecs = {l: base_dir.copy() * 0.95 for l in layers}

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = _save_vectors(base_vecs, tmpdir, "base")
            ft_path = _save_vectors(ft_vecs, tmpdir, "ft")
            output_dir = os.path.join(tmpdir, "comparison_output")

            results = run_full_comparison(base_path, ft_path, output_dir=output_dir)

            results_file = os.path.join(output_dir, "comparison_results.json")
            assert os.path.exists(results_file)

            with open(results_file) as f:
                saved = json.load(f)
            assert "vector_comparison" in saved
            assert "per_layer" in saved["vector_comparison"]

    def test_with_probe_comparison(self):
        """Should include probe comparison when results are provided."""
        layers = [0, 1, 2]

        np.random.seed(42)
        base_dir = np.random.randn(32)
        base_dir /= np.linalg.norm(base_dir)

        base_vecs = {l: base_dir.copy() for l in layers}
        ft_vecs = {l: base_dir.copy() * 0.95 for l in layers}

        base_results = _make_vector_results(layers, auroc_range=(0.6, 0.7), model="base")
        ft_results = _make_vector_results(layers, auroc_range=(0.7, 0.8), model="ft")

        with tempfile.TemporaryDirectory() as tmpdir:
            base_vec_path = _save_vectors(base_vecs, tmpdir, "base")
            ft_vec_path = _save_vectors(ft_vecs, tmpdir, "ft")
            base_res_path = os.path.join(tmpdir, "base_vector_results.json")
            ft_res_path = os.path.join(tmpdir, "ft_vector_results.json")

            with open(base_res_path, "w") as f:
                json.dump(base_results, f)
            with open(ft_res_path, "w") as f:
                json.dump(ft_results, f)

            output_dir = os.path.join(tmpdir, "comparison_output")

            results = run_full_comparison(
                base_vec_path, ft_vec_path,
                base_vector_results_path=base_res_path,
                ft_vector_results_path=ft_res_path,
                output_dir=output_dir,
            )

            assert "probe_comparison" in results
            assert "per_layer" in results["probe_comparison"]