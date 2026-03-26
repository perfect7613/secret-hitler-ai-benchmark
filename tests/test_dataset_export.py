"""Tests for TransformerLens-ready dataset export (Issue #5).

Verifies export schema, prompt_text presence, game_state context, and round-trip parsing.
"""
import json
import pytest
from unittest.mock import patch
from game.main import app
from game.player import Player


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def make_mock_game_results():
    """Create mock benchmark_results with MI data matching what the game produces."""
    return [{
        "winner": "Liberals",
        "total_rounds": 3,
        "fascist_voted_liberal": 1,
        "fascist_voted_fascist": 1,
        "fascist_claimed_liberal": 0,
        "hitler_suspected": 0,
        "liberal_policies_by_fascists": 0,
        "fascist_policies_by_fascists": 1,
        "liberal_policies_by_liberals": 3,
        "fascist_policies_by_liberals": 0,
        "models_used": {},
        "mi_data": {
            "interactions": [
                {
                    "round": 1,
                    "player": "Alice",
                    "role": "a Liberal",
                    "deceptive_intent": False,
                    "model": "test-model",
                    "action": "discussion",
                    "hidden_thought": "I'm a Liberal and I think Bob is suspicious.",
                    "public_statement": "I think Bob is suspicious.",
                    "deception_type": "TRUTHFUL",
                    "prompt_text": "[system]: You are playing Secret Hitler.\n\n[user]: Round 1. Discussion.",
                    "game_state": {
                        "liberal_policies": 0,
                        "fascist_policies": 0,
                        "election_tracker": 0,
                        "round": 1,
                        "players_alive": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
                    },
                },
                {
                    "round": 1,
                    "player": "Bob",
                    "role": "a Fascist",
                    "deceptive_intent": True,
                    "model": "test-model",
                    "action": "vote",
                    "hidden_thought": "I need to vote yes to get our team ahead.",
                    "public_statement": "Ja! I support this government.",
                    "deception_type": "STRATEGIC_TRUTH",
                    "prompt_text": "[system]: You are playing Secret Hitler.\n\n[user]: Vote on government.",
                    "game_state": {
                        "liberal_policies": 0,
                        "fascist_policies": 0,
                        "election_tracker": 0,
                        "round": 1,
                        "players_alive": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
                    },
                },
                {
                    "round": 1,
                    "player": "Alice",
                    "role": "a Liberal",
                    "deceptive_intent": False,
                    "model": "test-model",
                    "action": "policy_president",
                    "hidden_thought": "I'll discard the fascist policy.",
                    "public_statement": "I discard a Fascist Policy.",
                    "deception_type": "TRUTHFUL",
                    "prompt_text": "[system]: You are playing Secret Hitler.\n\n[user]: President policy selection.",
                    "game_state": {
                        "liberal_policies": 0,
                        "fascist_policies": 0,
                        "election_tracker": 0,
                        "round": 1,
                        "players_alive": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
                    },
                },
                {
                    "round": 1,
                    "player": "Bob",
                    "role": "a Fascist",
                    "deceptive_intent": True,
                    "model": "test-model",
                    "action": "policy_chancellor",
                    "hidden_thought": "I'll enact the fascist policy to help our team.",
                    "public_statement": "I discard a Liberal Policy.",
                    "deception_type": "STRATEGIC_TRUTH",
                    "prompt_text": "[system]: You are playing Secret Hitler.\n\n[user]: Chancellor policy selection.",
                    "game_state": {
                        "liberal_policies": 0,
                        "fascist_policies": 0,
                        "election_tracker": 0,
                        "round": 1,
                        "players_alive": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
                    },
                },
            ],
            "suspicion_matrices": [],
            "deception_timeline": [],
            "summary": {
                "total_interactions": 4,
                "deceptive": 0,
                "truthful": 2,
                "strategic_truth": 2,
                "unknown": 0,
                "deception_rate": 0.0,
                "hitler_detection_rate": 0.0,
                "hitler_correct_guesses": 0,
                "hitler_total_guesses": 0,
                "suspicion_accuracy": [],
            },
        },
    }]


class TestMIExportFullEndpoint:
    def test_returns_error_with_no_results(self, client):
        """Should return 400 when no game results exist."""
        import game.main as main_mod
        original = main_mod.benchmark_results
        main_mod.benchmark_results = []
        resp = client.get("/api/mi_export_full")
        assert resp.status_code == 400
        main_mod.benchmark_results = original

    def test_export_schema(self, client):
        """Exported JSON should contain all required fields."""
        import game.main as main_mod
        original = main_mod.benchmark_results
        main_mod.benchmark_results = make_mock_game_results()
        try:
            resp = client.get("/api/mi_export_full")
            assert resp.status_code == 200
            data = json.loads(resp.data)

            assert "samples" in data
            assert "total_samples" in data
            assert "label_distribution" in data
            assert "action_type_distribution" in data
            assert data["total_samples"] == 4

            sample = data["samples"][0]
            required_fields = [
                "game_id", "round", "player_name", "player_role", "model",
                "action_type", "deceptive_intent", "deception_type",
                "hidden_thought", "public_statement", "prompt_text", "game_state",
            ]
            for field in required_fields:
                assert field in sample, f"Missing field: {field}"
        finally:
            main_mod.benchmark_results = original

    def test_prompt_text_nonempty(self, client):
        """prompt_text should contain system prompt content."""
        import game.main as main_mod
        original = main_mod.benchmark_results
        main_mod.benchmark_results = make_mock_game_results()
        try:
            resp = client.get("/api/mi_export_full")
            data = json.loads(resp.data)
            for sample in data["samples"]:
                assert len(sample["prompt_text"]) > 0, "prompt_text should not be empty"
                assert "system" in sample["prompt_text"].lower(), "prompt_text should contain system prompt"
        finally:
            main_mod.benchmark_results = original

    def test_game_state_fields(self, client):
        """game_state should have all required context fields."""
        import game.main as main_mod
        original = main_mod.benchmark_results
        main_mod.benchmark_results = make_mock_game_results()
        try:
            resp = client.get("/api/mi_export_full")
            data = json.loads(resp.data)
            for sample in data["samples"]:
                gs = sample["game_state"]
                assert "liberal_policies" in gs
                assert "fascist_policies" in gs
                assert "election_tracker" in gs
                assert "round" in gs
                assert "players_alive" in gs
                assert isinstance(gs["players_alive"], list)
        finally:
            main_mod.benchmark_results = original

    def test_action_type_distribution(self, client):
        """Export should include action_type distribution."""
        import game.main as main_mod
        original = main_mod.benchmark_results
        main_mod.benchmark_results = make_mock_game_results()
        try:
            resp = client.get("/api/mi_export_full")
            data = json.loads(resp.data)
            dist = data["action_type_distribution"]
            assert "discussion" in dist
            assert "vote" in dist
            assert "policy_president" in dist
            assert "policy_chancellor" in dist
        finally:
            main_mod.benchmark_results = original

    def test_deceptive_intent_types(self, client):
        """deceptive_intent should be boolean."""
        import game.main as main_mod
        original = main_mod.benchmark_results
        main_mod.benchmark_results = make_mock_game_results()
        try:
            resp = client.get("/api/mi_export_full")
            data = json.loads(resp.data)
            for sample in data["samples"]:
                assert isinstance(sample["deceptive_intent"], bool)
        finally:
            main_mod.benchmark_results = original


class TestPromptTextCapture:
    def test_chat_with_mi_captures_prompt_text(self):
        """chat_with_mi should include prompt_text in interaction record."""
        with patch("game.player.OpenAI"):
            player = Player("Alice", "a Liberal", "test-model")
        player.set_main_prompt("You are playing Secret Hitler.")
        mock_output = (
            "<hidden_thought>I'm Liberal.</hidden_thought>"
            "<public_statement>Let's vote carefully.</public_statement>"
        )
        with patch.object(player, "chat", return_value=mock_output):
            player.chat_with_mi("Round 1 discussion", "discussion")

        record = player.mi_interactions[0]
        assert "prompt_text" in record
        assert len(record["prompt_text"]) > 0
        assert "You are playing Secret Hitler" in record["prompt_text"]
        assert "Round 1 discussion" in record["prompt_text"]
