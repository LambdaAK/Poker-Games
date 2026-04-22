import unittest

from blackjack import Action, BlackjackGame, card_label, hand_value, is_blackjack, parse_card


class BlackjackGameTests(unittest.TestCase):
    def test_hand_value_and_card_labels(self) -> None:
        hand = (parse_card("As"), parse_card("9d"))
        total, soft = hand_value(hand)
        self.assertEqual(total, 20)
        self.assertTrue(soft)
        self.assertEqual(card_label(hand[0]), "As")
        self.assertEqual(card_label(hand[1]), "9d")

    def test_reset_with_natural_blackjacks(self) -> None:
        game = BlackjackGame()
        state = game.reset(
            player_hand=(parse_card("As"), parse_card("Kd")),
            dealer_hand=(parse_card("Ah"), parse_card("Ts")),
        )
        self.assertTrue(state.terminal)
        self.assertTrue(is_blackjack(state.player_hand))
        self.assertTrue(is_blackjack(state.dealer_hand))
        self.assertEqual(state.rewards, (0.0, 0.0))
        self.assertEqual(game.legal_actions(), ())

    def test_stand_resolves_dealer_bust(self) -> None:
        game = BlackjackGame()
        game.reset(
            player_hand=(parse_card("Td"), parse_card("7c")),
            dealer_hand=(parse_card("9h"), parse_card("7s")),
            deck=(parse_card("Tc"),),
        )
        state, rewards, done = game.step(Action.STAND)
        self.assertTrue(done)
        self.assertTrue(state.terminal)
        self.assertEqual(state.dealer_hand, (parse_card("9h"), parse_card("7s"), parse_card("Tc")))
        self.assertEqual(rewards, (1.0, -1.0))

    def test_hit_to_21_resolves_immediately(self) -> None:
        game = BlackjackGame()
        game.reset(
            player_hand=(parse_card("9c"), parse_card("2d")),
            dealer_hand=(parse_card("Th"), parse_card("5s")),
            deck=(parse_card("Ts"), parse_card("2c")),
        )
        state, rewards, done = game.step(Action.HIT)
        self.assertTrue(done)
        self.assertTrue(state.terminal)
        self.assertEqual(hand_value(state.player_hand)[0], 21)
        self.assertEqual(hand_value(state.dealer_hand)[0], 17)
        self.assertEqual(rewards, (1.0, -1.0))

    def test_double_down_bets_two_units(self) -> None:
        game = BlackjackGame()
        game.reset(
            player_hand=(parse_card("9c"), parse_card("2d")),
            dealer_hand=(parse_card("Th"), parse_card("6s")),
            deck=(parse_card("Ts"), parse_card("2c")),
        )
        state, rewards, done = game.step(Action.DOUBLE)
        self.assertTrue(done)
        self.assertTrue(state.terminal)
        self.assertEqual(state.stake, 2)
        self.assertEqual(rewards, (2.0, -2.0))

    def test_double_illegal_after_hit(self) -> None:
        game = BlackjackGame()
        game.reset(
            player_hand=(parse_card("9c"), parse_card("2d")),
            dealer_hand=(parse_card("Th"), parse_card("5s")),
            deck=(parse_card("4c"), parse_card("2c")),
        )
        state, rewards, done = game.step(Action.HIT)
        self.assertFalse(done)
        self.assertEqual(rewards, (0.0, 0.0))
        with self.assertRaises(ValueError):
            game.step(Action.DOUBLE)


if __name__ == "__main__":
    unittest.main()
