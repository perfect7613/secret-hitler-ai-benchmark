"""Tests for LoRA fine-tuning pipeline (Issue #9).

Tests that don't require downloading large models. Integration tests
are skipped if transformers/peft are not installed.
"""
import json
import os
import tempfile

import pytest


class TestFineTuneDataset:
    def test_dataset_loads_jsonl(self):
        """Dataset should load and parse JSONL files."""
        try:
            from transformers import AutoTokenizer
        except ImportError:
            pytest.skip("transformers not installed")

        from finetune.train import FineTuneDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = os.path.join(tmpdir, "train.jsonl")
            with open(jsonl_path, "w") as f:
                for i in range(5):
                    record = {
                        "prompt": f"Game context {i}",
                        "completion": f"<hidden_thought>Thought {i}</hidden_thought>",
                    }
                    f.write(json.dumps(record) + "\n")

            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            dataset = FineTuneDataset(jsonl_path, tokenizer, max_length=64)
            assert len(dataset) == 5

            sample = dataset[0]
            assert "input_ids" in sample
            assert "attention_mask" in sample
            assert "labels" in sample
            assert sample["input_ids"].shape[0] == 64  # max_length


class TestHyperparameters:
    def test_hyperparams_declared_at_top(self):
        """All key hyperparameters should be importable from module."""
        from finetune.train import (
            MODEL_NAME,
            LORA_RANK,
            LORA_ALPHA,
            LORA_TARGET_MODULES,
            LEARNING_RATE,
            BATCH_SIZE,
            BUDGET_MINUTES,
            SEED,
        )
        assert isinstance(LORA_RANK, int) and LORA_RANK > 0
        assert isinstance(LORA_ALPHA, int) and LORA_ALPHA > 0
        assert isinstance(LEARNING_RATE, float) and LEARNING_RATE > 0
        assert isinstance(BATCH_SIZE, int) and BATCH_SIZE > 0
        assert isinstance(BUDGET_MINUTES, (int, float)) and BUDGET_MINUTES > 0


class TestLoRAIntegration:
    """Integration tests using the tiny pythia-70m model."""

    @pytest.fixture
    def tiny_model(self):
        try:
            from finetune.train import load_model_with_lora
        except ImportError:
            pytest.skip("transformers/peft not installed")
        model, tokenizer = load_model_with_lora(
            model_name="EleutherAI/pythia-70m",
            lora_rank=4,
            lora_alpha=8,
        )
        return model, tokenizer

    def test_lora_adapter_applied(self, tiny_model):
        """LoRA adapter params should exist in the model."""
        model, _ = tiny_model
        # Check that LoRA params exist
        lora_params = [n for n, _ in model.named_parameters() if "lora" in n.lower()]
        assert len(lora_params) > 0, "No LoRA parameters found"

    def test_model_generates_text(self, tiny_model):
        """Model with LoRA should be able to generate text."""
        import torch

        model, tokenizer = tiny_model
        model.eval()
        inputs = tokenizer("Hello, I am", return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
            )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        assert len(text) > len("Hello, I am")

    def test_checkpoint_save_load(self, tiny_model):
        """Checkpoint should save and load correctly."""
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        model, tokenizer = tiny_model

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            tokenizer.save_pretrained(tmpdir)

            # Verify checkpoint files exist
            assert os.path.exists(os.path.join(tmpdir, "adapter_config.json"))

            # Load back
            base_model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m")
            loaded = PeftModel.from_pretrained(base_model, tmpdir)
            assert loaded is not None

    def test_training_with_budget(self, tiny_model):
        """Training should stop after time budget and save checkpoint."""
        from finetune.train import train

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal training data
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)
            for split in ["train", "val"]:
                path = os.path.join(data_dir, f"{split}.jsonl")
                with open(path, "w") as f:
                    for i in range(20):
                        record = {
                            "prompt": f"Game: Secret Hitler | Round {i}\nRole: a Liberal\nAction: discussion",
                            "completion": f"<hidden_thought>I am Liberal</hidden_thought>\n<public_statement>Let's cooperate</public_statement>",
                        }
                        f.write(json.dumps(record) + "\n")

            checkpoint_dir = os.path.join(tmpdir, "checkpoint")
            result_dir = train(
                data_dir=data_dir,
                budget_minutes=0.1,  # 6 seconds
                model_name="EleutherAI/pythia-70m",
                checkpoint_dir=checkpoint_dir,
            )

            assert os.path.exists(result_dir)
            assert os.path.exists(os.path.join(result_dir, "adapter_config.json"))
            assert os.path.exists(os.path.join(result_dir, "metrics.json"))

            with open(os.path.join(result_dir, "metrics.json")) as f:
                metrics = json.load(f)
            assert "final_val_loss" in metrics
            assert "total_steps" in metrics
            assert metrics["total_steps"] > 0
