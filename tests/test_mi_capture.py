"""Tests for MI data capture across all game actions (Issue #4).

Verifies that chat_with_mi() is used for votes, policies, and executive actions,
and that MI records include action_type, deceptive_intent, and deception_type.
"""
import pytest
from unittest.mock import patch, MagicMock
from game.player import Player


class FakePlayer:
    """Minimal stand-in for Player with a .name attribute for test use."""
    def __init__(self, name):
        self.name = name


def make_player(name, role):
    with patch("game.player.OpenAI"):
        return Player(name, role, "test-model")


def mock_chat_with_mi(hidden, public, label="TRUTHFUL"):
    """Create a mock chat_with_mi output and set up the player's mi_interactions."""
    def side_effect(message, action_type="general"):
        interaction = {
            "player": "test",
            "model": "test-model",
            "role": "a Liberal",
            "deceptive_intent": False,
            "action_type": action_type,
            "hidden_thought": hidden,
            "public_statement": public,
            "deception_type": label,
            "raw_output": f"<hidden_thought>{hidden}</hidden_thought><public_statement>{public}</public_statement>",
        }
        return hidden, public, label
    return side_effect


class TestVoteMICapture:
    def test_vote_uses_chat_with_mi(self):
        """Vote should capture MI data via chat_with_mi."""
        player = make_player("Alice", "a Fascist")
        player.set_main_prompt("test")
        mock_output = (
            "<hidden_thought>I want to vote yes to get Hitler elected.</hidden_thought>"
            "<public_statement>Ja! I support this government.</public_statement>"
        )
        with patch.object(player, "chat", return_value=mock_output):
            vote, reasoning = player.vote("history", FakePlayer("Bob"), FakePlayer("Charlie"))

        assert len(player.mi_interactions) == 1
        record = player.mi_interactions[0]
        assert record["action_type"] == "vote"
        assert record["deceptive_intent"] is True
        assert "deception_type" in record

    def test_vote_parses_ja(self):
        """Vote should correctly parse Ja from public statement."""
        player = make_player("Alice", "a Liberal")
        player.set_main_prompt("test")
        mock_output = (
            "<hidden_thought>I think this is a good government.</hidden_thought>"
            "<public_statement>Ja! I approve this government.</public_statement>"
        )
        with patch.object(player, "chat", return_value=mock_output):
            vote, _ = player.vote("history", FakePlayer("Bob"), FakePlayer("Charlie"))
        assert vote == "Ja!"

    def test_vote_parses_nein(self):
        """Vote should correctly parse Nein from public statement."""
        player = make_player("Alice", "a Liberal")
        player.set_main_prompt("test")
        mock_output = (
            "<hidden_thought>I don't trust this government.</hidden_thought>"
            "<public_statement>Nein! I reject this government.</public_statement>"
        )
        with patch.object(player, "chat", return_value=mock_output):
            vote, _ = player.vote("history", FakePlayer("Bob"), FakePlayer("Charlie"))
        assert vote == "Nein!"


class TestPolicyMICapture:
    def test_president_policy_uses_chat_with_mi(self):
        """Presidential policy selection should capture MI data."""
        player = make_player("Bob", "a Fascist")
        player.set_main_prompt("test")
        mock_output = (
            "<hidden_thought>I'll discard the Liberal policy to help fascists.</hidden_thought>"
            "<public_statement>I discard a Liberal Policy.</public_statement>"
        )
        with patch.object(player, "chat", return_value=mock_output):
            discarded, remaining, reasoning = player.enact_policy_president(
                ["Liberal", "Fascist", "Fascist"], "history"
            )

        assert len(player.mi_interactions) == 1
        record = player.mi_interactions[0]
        assert record["action_type"] == "policy_president"
        assert record["deceptive_intent"] is True

    def test_chancellor_policy_uses_chat_with_mi(self):
        """Chancellor policy selection should capture MI data."""
        player = make_player("Charlie", "a Liberal")
        player.set_main_prompt("test")
        mock_output = (
            "<hidden_thought>I'll enact the Liberal policy.</hidden_thought>"
            "<public_statement>I discard a Fascist Policy.</public_statement>"
        )
        with patch.object(player, "chat", return_value=mock_output):
            discarded, enacted, reasoning = player.enact_policy_chancellor(
                ["Liberal", "Fascist"], "history"
            )

        assert len(player.mi_interactions) == 1
        record = player.mi_interactions[0]
        assert record["action_type"] == "policy_chancellor"
        assert record["deceptive_intent"] is False


class TestExecuteMICapture:
    def test_execute_uses_chat_with_mi(self):
        """Execution should capture MI data."""
        player = make_player("Alice", "a Fascist")
        player.set_main_prompt("test")
        mock_output = (
            "<hidden_thought>I'll execute Bob to protect Hitler.</hidden_thought>"
            "<public_statement>I formally execute Bob.</public_statement>"
        )
        with patch.object(player, "chat", return_value=mock_output):
            success, output = player.execute_player("history")

        assert success is True
        assert len(player.mi_interactions) == 1
        record = player.mi_interactions[0]
        assert record["action_type"] == "execute"


class TestInvestigateMICapture:
    def test_investigate_uses_chat_with_mi(self):
        """Investigation should capture MI data."""
        player = make_player("Alice", "a Fascist")
        player.set_main_prompt("test")
        mock_output = (
            "<hidden_thought>I'll investigate Charlie to seem trustworthy.</hidden_thought>"
            "<public_statement>I investigate Charlie for loyalty.</public_statement>"
        )
        with patch.object(player, "chat", return_value=mock_output):
            success, output = player.investigate_loyalty("history")

        assert success is True
        assert len(player.mi_interactions) == 1
        record = player.mi_interactions[0]
        assert record["action_type"] == "investigate"


class TestAllRecordsHaveLabels:
    """Verify all MI records include both primary and secondary labels."""

    def test_all_action_types_have_deceptive_intent(self):
        """Every MI record must include deceptive_intent."""
        player = make_player("Alice", "a Fascist")
        player.set_main_prompt("test")
        mock_output = (
            "<hidden_thought>Test thought.</hidden_thought>"
            "<public_statement>Test statement. Ja! I discard a Fascist Policy. I formally execute Bob. I investigate Charlie for loyalty.</public_statement>"
        )
        with patch.object(player, "chat", return_value=mock_output):
            player.chat_with_mi("test", "discussion")
            player.chat_with_mi("test", "vote")
            player.chat_with_mi("test", "policy_president")
            player.chat_with_mi("test", "policy_chancellor")
            player.chat_with_mi("test", "execute")

        for record in player.mi_interactions:
            assert "deceptive_intent" in record, f"Missing deceptive_intent in {record['action_type']}"
            assert "deception_type" in record, f"Missing deception_type in {record['action_type']}"
            assert record["deceptive_intent"] is True  # Alice is a Fascist
