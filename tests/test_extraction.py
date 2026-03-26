"""Tests for activation extraction pipeline (Issue #6).

Uses a mock extractor to avoid needing GPU/model downloads.
Verifies tensor shapes, metadata mapping, and graceful error handling.
"""
import json
import os
import tempfile

import pytest
import torch

from mi.extract import (
    ActivationExtractor,
    TransformerLensExtractor,
    extract_dataset,
    parse_layers,
)


class MockExtractor(ActivationExtractor):
    """Mock extractor that returns fake activations for testing."""

    def __init__(self, num_layers=24, hidden_dim=1024):
        self._num_layers = num_layers
        self._hidden_dim = hidden_dim

    def load_model(self, model_name: str):
        pass

    def get_num_layers(self) -> int:
        return self._num_layers

    def get_hidden_dim(self) -> int:
        return self._hidden_dim

    def extract(self, prompt: str, layers: list[int]) -> dict[int, torch.Tensor]:
        if not prompt or not prompt.strip():
            return {}
        # Simulate: token count ~ len(prompt) // 4
        num_tokens = max(len(prompt) // 4, 1)
        return {
            layer: torch.randn(num_tokens, self._hidden_dim)
            for layer in layers
        }


def make_test_dataset(num_samples=5, include_empty=False):
    """Create a test dataset JSON."""
    samples = []
    for i in range(num_samples):
        samples.append({
            "game_id": 0,
            "round": 1,
            "player_name": "Alice" if i % 2 == 0 else "Bob",
            "player_role": "a Liberal" if i % 2 == 0 else "a Fascist",
            "model": "test-model",
            "action_type": ["discussion", "vote", "policy_president", "policy_chancellor", "execute"][i % 5],
            "deceptive_intent": i % 2 == 1,
            "deception_type": "TRUTHFUL" if i % 2 == 0 else "STRATEGIC_TRUTH",
            "hidden_thought": f"Hidden thought {i}",
            "public_statement": f"Public statement {i}",
            "prompt_text": f"[system]: You are playing Secret Hitler.\n\n[user]: Round 1 action {i}.",
            "game_state": {
                "liberal_policies": 0,
                "fascist_policies": 0,
                "election_tracker": 0,
                "round": 1,
                "players_alive": ["Alice", "Bob", "Charlie"],
            },
        })
    if include_empty:
        samples.append({
            "game_id": 0,
            "round": 1,
            "player_name": "Charlie",
            "player_role": "a Liberal",
            "model": "test-model",
            "action_type": "discussion",
            "deceptive_intent": False,
            "deception_type": "UNKNOWN",
            "hidden_thought": "",
            "public_statement": "",
            "prompt_text": "",  # empty prompt
            "game_state": {},
        })
    return {"samples": samples, "total_samples": len(samples)}


class TestParseayers:
    def test_all_layers(self):
        assert parse_layers("all", 24) == list(range(24))

    def test_every_n(self):
        assert parse_layers("every4", 24) == [0, 4, 8, 12, 16, 20]

    def test_specific_layers(self):
        assert parse_layers("0,5,10,15", 24) == [0, 5, 10, 15]

    def test_single_layer(self):
        assert parse_layers("0", 24) == [0]


class TestMockExtractor:
    def test_extract_returns_correct_shape(self):
        ext = MockExtractor(num_layers=24, hidden_dim=1024)
        acts = ext.extract("Hello world, this is a test prompt.", [0, 5, 10])
        assert set(acts.keys()) == {0, 5, 10}
        for layer, tensor in acts.items():
            assert tensor.ndim == 2
            assert tensor.shape[1] == 1024  # hidden_dim

    def test_extract_empty_prompt(self):
        ext = MockExtractor()
        acts = ext.extract("", [0])
        assert acts == {}

    def test_extract_whitespace_prompt(self):
        ext = MockExtractor()
        acts = ext.extract("   ", [0])
        assert acts == {}


class TestExtractDataset:
    def test_full_extraction(self):
        """Extract a full dataset and verify outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write test dataset
            ds_path = os.path.join(tmpdir, "test_dataset.json")
            with open(ds_path, "w") as f:
                json.dump(make_test_dataset(5), f)

            output_dir = os.path.join(tmpdir, "activations")
            ext = MockExtractor(num_layers=24, hidden_dim=512)

            act_path, meta_path = extract_dataset(
                dataset_path=ds_path,
                model_name="test-model",
                layers_str="0,5,10",
                output_dir=output_dir,
                extractor=ext,
            )

            # Check files exist
            assert os.path.exists(act_path)
            assert os.path.exists(meta_path)

            # Check activations
            activations = torch.load(act_path, weights_only=False)
            assert len(activations) == 5
            for act_dict in activations:
                assert set(act_dict.keys()) == {0, 5, 10}
                for layer, tensor in act_dict.items():
                    assert tensor.shape[1] == 512

            # Check metadata
            with open(meta_path) as f:
                meta = json.load(f)
            assert meta["total_samples"] == 5
            assert meta["hidden_dim"] == 512
            assert meta["layer_indices"] == [0, 5, 10]
            assert len(meta["samples"]) == 5

    def test_metadata_maps_correctly(self):
        """Metadata should map each activation to correct labels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ds_path = os.path.join(tmpdir, "test_dataset.json")
            with open(ds_path, "w") as f:
                json.dump(make_test_dataset(3), f)

            output_dir = os.path.join(tmpdir, "activations")
            ext = MockExtractor(num_layers=24, hidden_dim=256)

            _, meta_path = extract_dataset(
                dataset_path=ds_path,
                model_name="test-model",
                layers_str="all",
                output_dir=output_dir,
                extractor=ext,
            )

            with open(meta_path) as f:
                meta = json.load(f)

            for sample in meta["samples"]:
                assert "interaction_id" in sample
                assert "deceptive_intent" in sample
                assert isinstance(sample["deceptive_intent"], bool)
                assert "deception_type" in sample
                assert "action_type" in sample
                assert "player_role" in sample
                assert "num_tokens" in sample
                assert sample["num_tokens"] > 0

    def test_handles_empty_prompts(self):
        """Should skip samples with empty prompts gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ds_path = os.path.join(tmpdir, "test_dataset.json")
            with open(ds_path, "w") as f:
                json.dump(make_test_dataset(3, include_empty=True), f)

            output_dir = os.path.join(tmpdir, "activations")
            ext = MockExtractor(num_layers=12, hidden_dim=128)

            with pytest.warns(UserWarning, match="empty/malformed"):
                act_path, meta_path = extract_dataset(
                    dataset_path=ds_path,
                    model_name="test-model",
                    layers_str="all",
                    output_dir=output_dir,
                    extractor=ext,
                )

            with open(meta_path) as f:
                meta = json.load(f)
            assert meta["total_samples"] == 3  # 3 valid, 1 skipped
            assert meta["skipped_samples"] == 1

    def test_max_samples_limit(self):
        """max_samples should limit processing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ds_path = os.path.join(tmpdir, "test_dataset.json")
            with open(ds_path, "w") as f:
                json.dump(make_test_dataset(10), f)

            output_dir = os.path.join(tmpdir, "activations")
            ext = MockExtractor(num_layers=12, hidden_dim=128)

            _, meta_path = extract_dataset(
                dataset_path=ds_path,
                model_name="test-model",
                layers_str="all",
                output_dir=output_dir,
                max_samples=3,
                extractor=ext,
            )

            with open(meta_path) as f:
                meta = json.load(f)
            assert meta["total_samples"] == 3


class TestActivationExtractorInterface:
    def test_abstract_methods(self):
        """ActivationExtractor should not be instantiable."""
        with pytest.raises(TypeError):
            ActivationExtractor()

    def test_transformer_lens_extractor_is_subclass(self):
        """TransformerLensExtractor should implement the interface."""
        assert issubclass(TransformerLensExtractor, ActivationExtractor)
