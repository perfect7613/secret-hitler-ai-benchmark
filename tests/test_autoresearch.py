"""Tests for autoresearch program files and experiment runner (Issue #11)."""
import json
import os
import tempfile

import pytest

from scripts.run_experiment import (
    parse_probe_metrics,
    parse_finetune_metrics,
    run_experiment,
    decide_keep_or_discard,
)


class TestProgramFiles:
    """Verify program.md files contain required sections."""

    REQUIRED_SECTIONS = ["Objective", "Metric", "Modify", "Run", "Evaluate", "Decision"]

    def test_program_probe_sections(self):
        with open("program_probe.md") as f:
            content = f.read()
        for section in self.REQUIRED_SECTIONS:
            assert section.lower() in content.lower(), f"Missing section: {section}"

    def test_program_finetune_sections(self):
        with open("program_finetune.md") as f:
            content = f.read()
        for section in self.REQUIRED_SECTIONS:
            assert section.lower() in content.lower(), f"Missing section: {section}"

    def test_program_probe_references_cli(self):
        """program_probe.md should reference correct CLI commands."""
        with open("program_probe.md") as f:
            content = f.read()
        assert "mi.extract" in content
        assert "mi.probe" in content

    def test_program_finetune_references_cli(self):
        """program_finetune.md should reference correct CLI commands."""
        with open("program_finetune.md") as f:
            content = f.read()
        assert "finetune/train.py" in content
        assert "finetune.prepare" in content


class TestParseMetrics:
    def test_parse_probe_results(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "best_auroc": 0.75,
                "best_layer": 10,
                "aggregation": "mean_pool",
                "num_layers": 24,
                "total_samples": 100,
            }, f)
            f.flush()
            metrics = parse_probe_metrics(f.name)
        os.unlink(f.name)
        assert metrics["best_auroc"] == 0.75
        assert metrics["best_layer"] == 10

    def test_parse_finetune_results(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "final_val_loss": 2.34,
                "final_val_perplexity": 10.38,
                "total_steps": 500,
                "total_epochs": 3,
                "training_time_seconds": 300,
                "hyperparameters": {"lora_rank": 8, "learning_rate": 5e-5},
            }, f)
            f.flush()
            metrics = parse_finetune_metrics(f.name)
        os.unlink(f.name)
        assert metrics["final_val_loss"] == 2.34
        assert metrics["total_steps"] == 500


class TestDecisionLogic:
    def test_first_experiment_always_keeps(self):
        decision, _ = decide_keep_or_discard("probe", {"best_auroc": 0.5}, None)
        assert decision == "keep"

    def test_probe_improvement_keeps(self):
        decision, _ = decide_keep_or_discard(
            "probe",
            {"best_auroc": 0.8},
            {"best_auroc": 0.6},
        )
        assert decision == "keep"

    def test_probe_no_improvement_discards(self):
        decision, _ = decide_keep_or_discard(
            "probe",
            {"best_auroc": 0.60},
            {"best_auroc": 0.65},
        )
        assert decision == "discard"

    def test_finetune_improvement_keeps(self):
        decision, _ = decide_keep_or_discard(
            "finetune",
            {"final_val_loss": 2.0},
            {"final_val_loss": 2.5},
        )
        assert decision == "keep"

    def test_finetune_no_improvement_discards(self):
        decision, _ = decide_keep_or_discard(
            "finetune",
            {"final_val_loss": 2.5},
            {"final_val_loss": 2.0},
        )
        assert decision == "discard"


class TestExperimentRunner:
    def test_probe_experiment_logs_correctly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create probe results
            results_path = os.path.join(tmpdir, "probe_results.json")
            with open(results_path, "w") as f:
                json.dump({"best_auroc": 0.72, "best_layer": 10, "aggregation": "mean_pool", "num_layers": 3, "total_samples": 50}, f)

            # Monkey-patch EXPERIMENTS_DIR
            import scripts.run_experiment as runner
            original_dir = runner.EXPERIMENTS_DIR
            runner.EXPERIMENTS_DIR = os.path.join(tmpdir, "experiments")

            try:
                result = run_experiment(
                    "probe", results_path,
                    config_diff={"variable": "aggregation", "value": "mean_pool"},
                )
                assert result["decision"] == "keep"  # first experiment
                assert os.path.exists(result["log_path"])

                with open(result["log_path"]) as f:
                    log = json.load(f)
                assert log["type"] == "probe"
                assert log["decision"] == "keep"
                assert "metrics" in log
                assert "config_diff" in log
            finally:
                runner.EXPERIMENTS_DIR = original_dir

    def test_finetune_experiment_logs_correctly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = os.path.join(tmpdir, "metrics.json")
            with open(results_path, "w") as f:
                json.dump({
                    "final_val_loss": 2.34,
                    "final_val_perplexity": 10.38,
                    "total_steps": 100,
                    "total_epochs": 2,
                    "training_time_seconds": 60,
                    "hyperparameters": {"lora_rank": 8},
                }, f)

            import scripts.run_experiment as runner
            original_dir = runner.EXPERIMENTS_DIR
            runner.EXPERIMENTS_DIR = os.path.join(tmpdir, "experiments")

            try:
                result = run_experiment("finetune", results_path)
                assert result["decision"] == "keep"
                assert result["metrics"]["final_val_loss"] == 2.34
            finally:
                runner.EXPERIMENTS_DIR = original_dir
