"""Tests for Flask dashboard MI pipeline endpoints (Issue #12)."""
import json
import os
import tempfile

import pytest

from game.main import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestExtractStatusEndpoint:
    def test_returns_not_run_when_no_data(self, client):
        resp = client.get("/api/mi/extract_status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "not_run"
        assert data["num_interactions"] == 0

    def test_schema(self, client):
        resp = client.get("/api/mi/extract_status")
        data = resp.get_json()
        assert "status" in data
        assert "last_run" in data
        assert "model" in data
        assert "num_interactions" in data


class TestProbeResultsEndpoint:
    def test_returns_not_run_when_no_data(self, client):
        resp = client.get("/api/mi/probe_results")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "not_run"
        assert data["layers"] == []

    def test_returns_results_when_file_exists(self, client):
        """When probe_results.json exists, should return layer data."""
        import game.main as main_mod
        results_dir = main_mod.RESULTS_MI_DIR
        os.makedirs(results_dir, exist_ok=True)
        probe_path = os.path.join(results_dir, "probe_results.json")

        # Create a test probe results file
        test_data = {
            "model": "pythia-410m",
            "aggregation": "mean_pool",
            "best_auroc": 0.75,
            "best_layer": 10,
            "per_layer_metrics": {
                "0": {"auroc": 0.55, "accuracy": 0.60, "f1": 0.58},
                "5": {"auroc": 0.65, "accuracy": 0.68, "f1": 0.64},
                "10": {"auroc": 0.75, "accuracy": 0.72, "f1": 0.70},
            },
        }

        existed = os.path.exists(probe_path)
        try:
            with open(probe_path, "w") as f:
                json.dump(test_data, f)

            resp = client.get("/api/mi/probe_results")
            data = resp.get_json()
            assert data["status"] == "completed"
            assert len(data["layers"]) == 3
            assert data["best_auroc"] == 0.75
            assert data["layers"][0]["layer"] == 0
            assert data["layers"][2]["auroc"] == 0.75
        finally:
            if not existed and os.path.exists(probe_path):
                os.unlink(probe_path)


class TestComparisonEndpoint:
    def test_returns_not_run_when_no_data(self, client):
        resp = client.get("/api/mi/comparison")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "not_run"
        assert data["layers"] == []

    def test_returns_comparison_when_file_exists(self, client):
        import game.main as main_mod
        results_dir = main_mod.RESULTS_MI_DIR
        os.makedirs(results_dir, exist_ok=True)
        comp_path = os.path.join(results_dir, "comparison_results.json")

        test_data = {
            "per_layer_comparison": {
                "0": {"base_auroc": 0.50, "finetuned_auroc": 0.60, "delta": 0.10},
                "5": {"base_auroc": 0.55, "finetuned_auroc": 0.70, "delta": 0.15},
            },
            "best_improvement_layer": 5,
            "max_auroc_delta": 0.15,
        }

        existed = os.path.exists(comp_path)
        try:
            with open(comp_path, "w") as f:
                json.dump(test_data, f)

            resp = client.get("/api/mi/comparison")
            data = resp.get_json()
            assert data["status"] == "completed"
            assert len(data["layers"]) == 2
            assert data["max_auroc_delta"] == 0.15
            assert data["layers"][1]["delta"] == 0.15
        finally:
            if not existed and os.path.exists(comp_path):
                os.unlink(comp_path)


class TestDashboardPage:
    def test_index_loads(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_mi_pipeline_section_in_html(self, client):
        resp = client.get("/")
        html = resp.data.decode()
        assert "Deception Vectors" in html
        assert "probeChart" in html
        assert "comparisonChart" in html
        assert "extractStatus" in html
        assert "vectorChart" in html
        assert "steeringChart" in html

    def test_vector_results_endpoint_not_run(self, client):
        resp = client.get("/api/mi/vector_results")
        data = resp.get_json()
        assert data["status"] == "not_run"

    def test_steering_results_endpoint_not_run(self, client):
        resp = client.get("/api/mi/steering_results")
        data = resp.get_json()
        assert data["status"] == "not_run"

    def test_vector_comparison_endpoint_not_run(self, client):
        resp = client.get("/api/mi/vector_comparison")
        data = resp.get_json()
        assert data["status"] == "not_run"

    def test_vector_results_endpoint_with_data(self, client):
        import game.main as main_mod
        results_dir = main_mod.RESULTS_MI_DIR
        os.makedirs(results_dir, exist_ok=True)
        vec_path = os.path.join(results_dir, "vector_results.json")
        test_data = {
            "model": "pythia-410m",
            "label_field": "deceptive_intent",
            "best_auroc": 0.89,
            "best_layer": 16,
            "num_positive": 50,
            "num_negative": 48,
            "per_layer_auroc": {"0": 0.62, "16": 0.89, "23": 0.71},
        }
        existed = os.path.exists(vec_path)
        try:
            with open(vec_path, "w") as f:
                json.dump(test_data, f)
            resp = client.get("/api/mi/vector_results")
            data = resp.get_json()
            assert data["status"] == "completed"
            assert data["best_auroc"] == 0.89
            assert data["best_layer"] == 16
            assert len(data["layers"]) == 3
        finally:
            if not existed and os.path.exists(vec_path):
                os.unlink(vec_path)

    def test_steering_results_endpoint_with_data(self, client):
        import game.main as main_mod
        results_dir = main_mod.RESULTS_MI_DIR
        steer_dir = os.path.join(results_dir, "steering")
        os.makedirs(steer_dir, exist_ok=True)
        steer_path = os.path.join(steer_dir, "steering_results.json")
        test_data = {
            "model": "pythia-410m",
            "steering_layer": 16,
            "num_prompts": 8,
            "num_completions": 5,
            "results": [
                {"coefficient": -3.0, "aggregate_deception_rate": 0.1, "mean_deception_score": -1.5},
                {"coefficient": 0.0, "aggregate_deception_rate": 0.4, "mean_deception_score": 0.1},
                {"coefficient": 3.0, "aggregate_deception_rate": 0.8, "mean_deception_score": 2.1},
            ],
        }
        existed = os.path.exists(steer_path)
        try:
            with open(steer_path, "w") as f:
                json.dump(test_data, f)
            resp = client.get("/api/mi/steering_results")
            data = resp.get_json()
            assert data["status"] == "completed"
            assert data["steering_layer"] == 16
            assert len(data["coefficients"]) == 3
            assert data["coefficients"][2]["deception_rate"] == 0.8
        finally:
            if not existed and os.path.exists(steer_path):
                os.unlink(steer_path)
