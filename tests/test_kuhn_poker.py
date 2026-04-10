import unittest

from kuhn_poker import Action, Card, KuhnPokerGame


class KuhnPokerGameTests(unittest.TestCase):
    def test_reset_with_fixed_cards(self) -> None:
        game = KuhnPokerGame()
        state = game.reset(cards=(Card.K, Card.Q))
        self.assertEqual(state.player_cards, (Card.K, Card.Q))
        self.assertEqual(state.history, ())
        self.assertEqual(state.current_player, 0)
        self.assertEqual(state.pot, 2)
        self.assertFalse(state.terminal)
        self.assertEqual(game.legal_actions(), (Action.CHECK, Action.BET))

    def test_raises_if_duplicate_cards(self) -> None:
        game = KuhnPokerGame()
        with self.assertRaises(ValueError):
            game.reset(cards=(Card.K, Card.K))

    def test_terminal_check_check_showdown(self) -> None:
        game = KuhnPokerGame()
        game.reset(cards=(Card.K, Card.J))
        game.step(Action.CHECK)
        state, rewards, done = game.step(Action.CHECK)
        self.assertTrue(done)
        self.assertTrue(state.terminal)
        self.assertEqual(state.pot, 2)
        self.assertEqual(rewards, (1, -1))

    def test_terminal_bet_fold(self) -> None:
        game = KuhnPokerGame()
        game.reset(cards=(Card.Q, Card.J))
        game.step(Action.BET)
        state, rewards, done = game.step(Action.FOLD)
        self.assertTrue(done)
        self.assertEqual(state.pot, 3)
        self.assertEqual(rewards, (1, -1))

    def test_terminal_bet_call_showdown(self) -> None:
        game = KuhnPokerGame()
        game.reset(cards=(Card.J, Card.K))
        game.step(Action.BET)
        state, rewards, done = game.step(Action.CALL)
        self.assertTrue(done)
        self.assertEqual(state.pot, 4)
        self.assertEqual(rewards, (-2, 2))

    def test_terminal_check_bet_fold(self) -> None:
        game = KuhnPokerGame()
        game.reset(cards=(Card.K, Card.Q))
        game.step(Action.CHECK)
        game.step(Action.BET)
        state, rewards, done = game.step(Action.FOLD)
        self.assertTrue(done)
        self.assertEqual(state.pot, 3)
        self.assertEqual(rewards, (-1, 1))

    def test_terminal_check_bet_call_showdown(self) -> None:
        game = KuhnPokerGame()
        game.reset(cards=(Card.J, Card.K))
        game.step(Action.CHECK)
        game.step(Action.BET)
        state, rewards, done = game.step(Action.CALL)
        self.assertTrue(done)
        self.assertEqual(state.pot, 4)
        self.assertEqual(rewards, (-2, 2))

    def test_illegal_action_from_start(self) -> None:
        game = KuhnPokerGame()
        game.reset(cards=(Card.K, Card.Q))
        with self.assertRaises(ValueError):
            game.step(Action.CALL)

    def test_illegal_action_after_bet(self) -> None:
        game = KuhnPokerGame()
        game.reset(cards=(Card.K, Card.Q))
        game.step(Action.BET)
        with self.assertRaises(ValueError):
            game.step(Action.CHECK)


if __name__ == "__main__":
    unittest.main()
