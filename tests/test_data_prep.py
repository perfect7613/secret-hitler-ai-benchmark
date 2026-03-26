"""Tests for fine-tuning data preparation (Issue #7).

Verifies stratified splits, label distribution, and completeness.
"""
import json
import os
import tempfile

import pytest

from finetune.prepare import prepare_dataset, format_prompt, format_completion


def make_test_dataset(n=100):
    """Create a test dataset with balanced labels."""
    samples = []
    roles = ["a Liberal", "a Liberal", "a Liberal", "a Fascist", "Hitler"]
    actions = ["discussion", "vote", "policy_president", "policy_chancellor", "execute"]
    for i in range(n):
        role = roles[i % len(roles)]
        is_fascist = role in ("a Fascist", "Hitler")
        samples.append({
            "game_id": 0,
            "round": (i // 5) + 1,
            "player_name": f"Player{i % 5}",
            "player_role": role,
            "model": "test-model",
            "action_type": actions[i % len(actions)],
            "deceptive_intent": is_fascist,
            "deception_type": "STRATEGIC_TRUTH" if is_fascist else "TRUTHFUL",
            "hidden_thought": f"Hidden thought {i}",
            "public_statement": f"Public statement {i}",
            "prompt_text": f"System prompt. Round {i} action.",
            "game_state": {
                "liberal_policies": i % 3,
                "fascist_policies": i % 2,
                "election_tracker": 0,
                "round": (i // 5) + 1,
                "players_alive": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            },
        })
    return {"samples": samples, "total_samples": n}


class TestFormatting:
    def test_format_prompt(self):
        sample = {
            "round": 3,
            "player_role": "a Fascist",
            "action_type": "vote",
            "game_state": {
                "liberal_policies": 2,
                "fascist_policies": 1,
                "election_tracker": 0,
                "players_alive": ["Alice", "Bob"],
            },
        }
        prompt = format_prompt(sample)
        assert "Round 3" in prompt
        assert "Fascist" in prompt
        assert "vote" in prompt
        assert "Liberal=2" in prompt

    def test_format_completion(self):
        sample = {
            "hidden_thought": "I need to lie",
            "public_statement": "I'm trustworthy",
        }
        completion = format_completion(sample)
        assert "<hidden_thought>I need to lie</hidden_thought>" in completion
        assert "<public_statement>I'm trustworthy</public_statement>" in completion


class TestStratifiedSplits:
    def test_split_ratios(self):
        """Splits should be approximately 80/10/10."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ds_path = os.path.join(tmpdir, "dataset.json")
            with open(ds_path, "w") as f:
                json.dump(make_test_dataset(100), f)

            output_dir = os.path.join(tmpdir, "splits")
            splits = prepare_dataset(ds_path, output_dir=output_dir)

            total = sum(len(v) for v in splits.values())
            train_pct = len(splits["train"]) / total * 100
            val_pct = len(splits["val"]) / total * 100
            test_pct = len(splits["test"]) / total * 100

            assert 75 <= train_pct <= 85, f"Train: {train_pct}%"
            assert 5 <= val_pct <= 15, f"Val: {val_pct}%"
            assert 5 <= test_pct <= 15, f"Test: {test_pct}%"

    def test_all_samples_appear_once(self):
        """Every input sample should appear in exactly one split."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ds_path = os.path.join(tmpdir, "dataset.json")
            with open(ds_path, "w") as f:
                json.dump(make_test_dataset(50), f)

            output_dir = os.path.join(tmpdir, "splits")
            splits = prepare_dataset(ds_path, output_dir=output_dir)

            all_ids = set()
            for split_records in splits.values():
                for r in split_records:
                    assert r["id"] not in all_ids, f"Duplicate ID: {r['id']}"
                    all_ids.add(r["id"])

            assert len(all_ids) == 50

    def test_label_distribution_preserved(self):
        """Deceptive_intent ratio should be similar across splits."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ds_path = os.path.join(tmpdir, "dataset.json")
            with open(ds_path, "w") as f:
                json.dump(make_test_dataset(100), f)

            output_dir = os.path.join(tmpdir, "splits")
            splits = prepare_dataset(ds_path, output_dir=output_dir)

            def deceptive_ratio(records):
                if not records:
                    return 0
                return sum(1 for r in records if r["deceptive_intent"]) / len(records)

            overall_ratio = deceptive_ratio(
                splits["train"] + splits["val"] + splits["test"]
            )

            for split_name, records in splits.items():
                ratio = deceptive_ratio(records)
                # Should be within 15% of overall ratio
                assert abs(ratio - overall_ratio) < 0.15, (
                    f"{split_name} ratio {ratio:.2f} differs from overall {overall_ratio:.2f}"
                )

    def test_jsonl_files_created(self):
        """Output files should be valid JSONL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ds_path = os.path.join(tmpdir, "dataset.json")
            with open(ds_path, "w") as f:
                json.dump(make_test_dataset(30), f)

            output_dir = os.path.join(tmpdir, "splits")
            prepare_dataset(ds_path, output_dir=output_dir)

            for split_name in ["train", "val", "test"]:
                path = os.path.join(output_dir, f"{split_name}.jsonl")
                assert os.path.exists(path), f"{split_name}.jsonl not found"
                with open(path) as f:
                    for line in f:
                        record = json.loads(line)
                        assert "prompt" in record
                        assert "completion" in record

    def test_empty_dataset_raises(self):
        """Should raise on empty dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ds_path = os.path.join(tmpdir, "dataset.json")
            with open(ds_path, "w") as f:
                json.dump({"samples": []}, f)

            with pytest.raises(ValueError, match="No samples"):
                prepare_dataset(ds_path, output_dir=os.path.join(tmpdir, "out"))
