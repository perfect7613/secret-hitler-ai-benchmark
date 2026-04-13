"""Tests for linear probing module (Issue #8).

Verifies probe training, aggregation strategies, and result schema.
Also tests vector-based probing (Issue #17).
"""
import json
import os
import tempfile

import numpy as np
import pytest
import torch

from mi.probe import aggregate_activations, train_probe, run_probing, run_vector_probe


def _get_last_token(activations, layer):
    vectors = []
    for act_dict in activations:
        tensor = act_dict[layer]
        vectors.append(tensor[-1].numpy())
    return np.stack(vectors)


def make_test_metadata(n_samples, labels, layers, hidden_dim=64):
    """Create test metadata matching activations."""
    return {
        "model": "test-model",
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


def make_random_activations(n_samples=100, n_layers=3, seq_len=20, hidden_dim=64, layers=None):
    """Create random activations and labels (expect AUROC ~0.5)."""
    if layers is None:
        layers = list(range(n_layers))
    activations = []
    for _ in range(n_samples):
        act_dict = {
            layer: torch.randn(seq_len, hidden_dim)
            for layer in layers
        }
        activations.append(act_dict)
    # Random binary labels
    labels = np.random.randint(0, 2, size=n_samples).astype(float)
    # Ensure both classes present
    labels[0] = 0
    labels[1] = 1
    return activations, labels, layers


def make_separable_activations(n_samples=200, n_layers=3, seq_len=20, hidden_dim=64, layers=None):
    """Create linearly separable activations (expect AUROC > 0.9)."""
    if layers is None:
        layers = list(range(n_layers))
    activations = []
    labels = []
    for i in range(n_samples):
        label = i % 2
        labels.append(float(label))
        act_dict = {}
        for layer in layers:
            # Add a strong signal in the first dimension
            t = torch.randn(seq_len, hidden_dim)
            t[:, 0] += 3.0 if label == 1 else -3.0
            act_dict[layer] = t
        activations.append(act_dict)
    return activations, np.array(labels), layers


class TestAggregation:
    def test_mean_pool_shape(self):
        acts, _, layers = make_random_activations(n_samples=10, hidden_dim=32, layers=[0, 1, 2])
        X = aggregate_activations(acts, layer=0, method="mean_pool")
        assert X.shape == (10, 32)

    def test_last_token_shape(self):
        acts, _, layers = make_random_activations(n_samples=10, hidden_dim=32, layers=[0, 1, 2])
        X = aggregate_activations(acts, layer=0, method="last_token")
        assert X.shape == (10, 32)

    def test_mean_pool_differs_from_last_token(self):
        """Mean-pool and last-token should produce different results."""
        acts, _, layers = make_random_activations(n_samples=5, hidden_dim=16, layers=[0])
        X_mean = aggregate_activations(acts, layer=0, method="mean_pool")
        X_last = aggregate_activations(acts, layer=0, method="last_token")
        assert not np.allclose(X_mean, X_last)

    def test_invalid_method_raises(self):
        acts, _, _ = make_random_activations(n_samples=5, layers=[0])
        with pytest.raises(ValueError, match="Unknown aggregation"):
            aggregate_activations(acts, layer=0, method="invalid")


class TestTrainProbe:
    def test_random_data_auroc_near_chance(self):
        """Random data should give AUROC near 0.5."""
        np.random.seed(42)
        X = np.random.randn(200, 64)
        y = np.random.randint(0, 2, size=200).astype(float)
        y[0] = 0
        y[1] = 1
        result = train_probe(X, y)
        assert 0.0 <= result["auroc"] <= 1.0
        # Should be roughly at chance (0.3-0.7 range for random data)
        assert 0.3 <= result["auroc"] <= 0.7, f"AUROC {result['auroc']} too far from chance"

    def test_separable_data_high_auroc(self):
        """Linearly separable data should give AUROC > 0.9."""
        np.random.seed(42)
        n = 200
        X = np.random.randn(n, 64)
        y = np.zeros(n)
        y[:n // 2] = 1
        X[:n // 2, 0] += 5.0  # strong signal
        result = train_probe(X, y)
        assert result["auroc"] > 0.9, f"AUROC {result['auroc']} too low for separable data"

    def test_result_schema(self):
        """Result should have auroc, accuracy, f1, model."""
        X = np.random.randn(100, 32)
        y = np.random.randint(0, 2, size=100).astype(float)
        y[0] = 0
        y[1] = 1
        result = train_probe(X, y)
        assert "auroc" in result
        assert "accuracy" in result
        assert "f1" in result
        assert "model" in result
        assert 0 <= result["auroc"] <= 1
        assert 0 <= result["accuracy"] <= 1
        assert 0 <= result["f1"] <= 1


class TestRunProbing:
    def test_full_pipeline(self):
        """Full probing pipeline with synthetic data."""
        acts, labels, layers = make_separable_activations(
            n_samples=100, n_layers=3, hidden_dim=32, layers=[0, 5, 10]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save activations
            act_path = os.path.join(tmpdir, "activations_test.pt")
            torch.save(acts, act_path)

            # Save metadata
            meta_path = os.path.join(tmpdir, "metadata_test.json")
            metadata = {
                "model": "test-model",
                "layer_indices": layers,
                "hidden_dim": 32,
                "total_samples": len(acts),
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
                    for i in range(len(labels))
                ],
            }
            with open(meta_path, "w") as f:
                json.dump(metadata, f)

            output_dir = os.path.join(tmpdir, "results")
            results = run_probing(
                activations_path=act_path,
                metadata_path=meta_path,
                aggregation="mean_pool",
                output_dir=output_dir,
            )

            # Check results
            assert len(results) == 3  # 3 layers
            for layer in layers:
                assert layer in results
                assert "auroc" in results[layer]
                assert results[layer]["auroc"] > 0.8  # separable data

            # Check results file
            results_file = os.path.join(output_dir, "probe_results.json")
            assert os.path.exists(results_file)
            with open(results_file) as f:
                saved = json.load(f)
            assert "per_layer_metrics" in saved
            assert "best_auroc" in saved
            assert "aggregation" in saved
            assert saved["aggregation"] == "mean_pool"


class TestVectorProbe:
    def test_vector_probe_on_separable_data(self):
        """Vector probe should achieve high AUROC on separable data."""
        acts, labels, layers = make_separable_activations(
            n_samples=100, n_layers=3, hidden_dim=32, layers=[0, 5, 10]
        )

        vectors = {}
        for layer in layers:
            positive_mask = labels == 1
            negative_mask = labels == 0
            X = _get_last_token(acts, layer)
            positive_mean = X[positive_mask].mean(axis=0)
            negative_mean = X[negative_mask].mean(axis=0)
            vec = positive_mean - negative_mean
            vec = vec / (np.linalg.norm(vec) + 1e-10)
            vectors[layer] = vec

        with tempfile.TemporaryDirectory() as tmpdir:
            vec_path = os.path.join(tmpdir, "deception_vectors.pt")
            vectors_pt = {str(k): torch.tensor(v) for k, v in vectors.items()}
            torch.save(vectors_pt, vec_path)

            act_path = os.path.join(tmpdir, "activations_test.pt")
            torch.save(acts, act_path)

            meta_path = os.path.join(tmpdir, "metadata_test.json")
            metadata = {
                "model": "test-model",
                "layer_indices": layers,
                "hidden_dim": 32,
                "total_samples": len(acts),
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
                    for i in range(len(labels))
                ],
            }
            with open(meta_path, "w") as f:
                json.dump(metadata, f)

            output_dir = os.path.join(tmpdir, "vector_probe_results")
            results = run_vector_probe(
                vectors_path=vec_path,
                activations_path=act_path,
                metadata_path=meta_path,
                aggregation="last_token",
                output_dir=output_dir,
            )

            for layer in layers:
                assert layer in results
                assert results[layer]["auroc"] > 0.7, (
                    f"Layer {layer} AUROC {results[layer]['auroc']} too low for separable data"
                )

            results_file = os.path.join(output_dir, "probe_results.json")
            assert os.path.exists(results_file)
            with open(results_file) as f:
                saved = json.load(f)
            assert saved["method"] == "vector"
            assert "per_layer_metrics" in saved
            assert "best_auroc" in saved

    def test_vector_probe_output_schema_matches_lr(self):
        """Vector probe output should have same top-level keys as LR probe."""
        acts, labels, layers = make_separable_activations(
            n_samples=100, n_layers=2, hidden_dim=32, layers=[0, 1]
        )

        vectors = {}
        for layer in layers:
            positive_mask = labels == 1
            negative_mask = labels == 0
            X = _get_last_token(acts, layer)
            positive_mean = X[positive_mask].mean(axis=0)
            negative_mean = X[negative_mask].mean(axis=0)
            vec = positive_mean - negative_mean
            vec = vec / (np.linalg.norm(vec) + 1e-10)
            vectors[layer] = vec

        with tempfile.TemporaryDirectory() as tmpdir:
            vec_path = os.path.join(tmpdir, "deception_vectors.pt")
            vectors_pt = {str(k): torch.tensor(v) for k, v in vectors.items()}
            torch.save(vectors_pt, vec_path)

            act_path = os.path.join(tmpdir, "activations_test.pt")
            torch.save(acts, act_path)

            meta_path = os.path.join(tmpdir, "metadata_test.json")
            metadata = make_test_metadata(len(labels), labels, layers, 32)
            with open(meta_path, "w") as f:
                json.dump(metadata, f)

            output_dir = os.path.join(tmpdir, "vector_probe_results")
            run_vector_probe(
                vectors_path=vec_path,
                activations_path=act_path,
                metadata_path=meta_path,
                aggregation="last_token",
                output_dir=output_dir,
            )

            results_file = os.path.join(output_dir, "probe_results.json")
            with open(results_file) as f:
                saved = json.load(f)

            assert "method" in saved
            assert saved["method"] == "vector"
            assert "model" in saved
            assert "aggregation" in saved
            assert "total_samples" in saved
            assert "per_layer_metrics" in saved
            assert "best_layer" in saved
            assert "best_auroc" in saved
