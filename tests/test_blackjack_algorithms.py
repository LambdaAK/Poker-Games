import tempfile
import unittest
import random

from blackjack import Action, BlackjackGame, BasicStrategyPolicy, TabularActionValuePolicy, parse_card
from blackjack.algorithms import info_state_key


class BlackjackAlgorithmTests(unittest.TestCase):
    def test_info_state_key(self) -> None:
        game = BlackjackGame()
        state = game.reset(
            player_hand=(parse_card("As"), parse_card("7d")),
            dealer_hand=(parse_card("Th"), parse_card("5s")),
        )
        self.assertEqual(info_state_key(state), "p18s|dT|n2|dbl1")

    def test_basic_strategy_rules(self) -> None:
        policy = BasicStrategyPolicy()

        game = BlackjackGame()
        state = game.reset(
            player_hand=(parse_card("6c"), parse_card("5d")),
            dealer_hand=(parse_card("9h"), parse_card("7s")),
        )
        self.assertEqual(policy.choose_action(state, game.legal_actions(), random.Random()), Action.DOUBLE)

        state = game.reset(
            player_hand=(parse_card("Td"), parse_card("6c")),
            dealer_hand=(parse_card("Ah"), parse_card("7s")),
        )
        self.assertEqual(policy.choose_action(state, game.legal_actions(), random.Random()), Action.HIT)

        state = game.reset(
            player_hand=(parse_card("Td"), parse_card("7c")),
            dealer_hand=(parse_card("6h"), parse_card("7s")),
        )
        self.assertEqual(policy.choose_action(state, game.legal_actions(), random.Random()), Action.STAND)

    def test_monte_carlo_update_moves_value(self) -> None:
        policy = TabularActionValuePolicy()
        key = "p11h|d9|n2|dbl1"
        legal = (Action.HIT, Action.STAND, Action.DOUBLE)
        before = policy.action_values(key, legal)[Action.DOUBLE]
        policy.update_monte_carlo(key, legal, Action.DOUBLE, 1.0)
        after = policy.action_values(key, legal)[Action.DOUBLE]
        self.assertGreater(after, before)

    def test_q_learning_update_moves_value(self) -> None:
        policy = TabularActionValuePolicy()
        key = "p16h|dT|n2|dbl0"
        legal = (Action.HIT, Action.STAND)
        before = policy.action_values(key, legal)[Action.HIT]
        policy.update_q_learning(key, legal, Action.HIT, 1.0, 0.5)
        after = policy.action_values(key, legal)[Action.HIT]
        self.assertGreater(after, before)

    def test_policy_roundtrip(self) -> None:
        policy = TabularActionValuePolicy()
        key = "p12h|d6|n2|dbl1"
        legal = (Action.HIT, Action.STAND, Action.DOUBLE)
        policy.update_monte_carlo(key, legal, Action.STAND, 1.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/policy.json"
            policy.save(path)
            loaded = TabularActionValuePolicy.load(path)
            self.assertEqual(policy.to_json_dict(), loaded.to_json_dict())


if __name__ == "__main__":
    unittest.main()
