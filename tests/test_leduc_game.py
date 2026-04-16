import unittest

from leduc_poker.game import Action, Card, LeducPokerGame


class LeducPokerGameTests(unittest.TestCase):
    def test_reset_with_fixed_cards_and_board(self) -> None:
        game = LeducPokerGame()
        state = game.reset(cards=(Card.K1, Card.Q1), board_card=Card.J1)
        self.assertEqual(state.player_cards, (Card.K1, Card.Q1))
        self.assertIsNone(state.board_card)
        self.assertEqual(state.round_index, 0)
        self.assertEqual(state.pot, 2)
        self.assertEqual(state.contributions, (1, 1))
        self.assertEqual(game.legal_actions(), (Action.CHECK, Action.BET))

    def test_preflop_check_check_reveals_board(self) -> None:
        game = LeducPokerGame()
        game.reset(cards=(Card.K1, Card.Q1), board_card=Card.J1)
        game.step(Action.CHECK)
        state, rewards, done = game.step(Action.CHECK)
        self.assertFalse(done)
        self.assertEqual(rewards, (0, 0))
        self.assertEqual(state.round_index, 1)
        self.assertEqual(state.board_card, Card.J1)
        self.assertEqual(state.current_player, 0)
        self.assertEqual(state.to_call, 0)
        self.assertEqual(game.legal_actions(), (Action.CHECK, Action.BET))

    def test_fold_after_bet(self) -> None:
        game = LeducPokerGame()
        game.reset(cards=(Card.K1, Card.Q1), board_card=Card.J1)
        game.step(Action.BET)
        state, rewards, done = game.step(Action.FOLD)
        self.assertTrue(done)
        self.assertTrue(state.terminal)
        self.assertEqual(state.pot, 3)
        self.assertEqual(rewards, (1, -1))

    def test_raise_then_call_closes_round(self) -> None:
        game = LeducPokerGame()
        game.reset(cards=(Card.K1, Card.Q1), board_card=Card.J1)
        game.step(Action.BET)
        state, _, _ = game.step(Action.RAISE)
        self.assertEqual(state.to_call, 1)
        self.assertEqual(game.legal_actions(), (Action.CALL, Action.FOLD))
        state, rewards, done = game.step(Action.CALL)
        self.assertFalse(done)
        self.assertEqual(rewards, (0, 0))
        self.assertEqual(state.round_index, 1)
        self.assertEqual(state.pot, 6)
        self.assertEqual(state.contributions, (3, 3))

    def test_showdown_high_card_win(self) -> None:
        game = LeducPokerGame()
        game.reset(cards=(Card.K1, Card.Q1), board_card=Card.J1)
        game.step(Action.CHECK)
        game.step(Action.CHECK)
        game.step(Action.CHECK)
        state, rewards, done = game.step(Action.CHECK)
        self.assertTrue(done)
        self.assertEqual(state.board_card, Card.J1)
        self.assertEqual(rewards, (1, -1))

    def test_showdown_pair_beats_high_card(self) -> None:
        game = LeducPokerGame()
        game.reset(cards=(Card.Q1, Card.K1), board_card=Card.Q2)
        game.step(Action.CHECK)
        game.step(Action.CHECK)
        game.step(Action.BET)
        state, rewards, done = game.step(Action.CALL)
        self.assertTrue(done)
        self.assertTrue(state.terminal)
        self.assertEqual(state.pot, 6)
        self.assertEqual(rewards, (3, -3))
        self.assertEqual(game.state.contributions, (3, 3))
        self.assertEqual(game.state.terminal, True)
        self.assertEqual(game.state.folded_player, None)
        self.assertEqual(game.state.board_card, Card.Q2)
        self.assertEqual(game.state.pot, 6)
        self.assertEqual(game.state.round_index, 1)
        self.assertEqual(game.state.current_player, -1)
        self.assertEqual(game.state.to_call, 0)
        self.assertEqual(game.state.raises_in_round, 0)
        self.assertEqual(game.state.histories[0], (Action.CHECK, Action.CHECK))
        self.assertEqual(game.state.histories[1], (Action.BET, Action.CALL))

    def test_showdown_tie_splits_pot(self) -> None:
        game = LeducPokerGame()
        game.reset(cards=(Card.J1, Card.J2), board_card=Card.K1)
        game.step(Action.CHECK)
        game.step(Action.CHECK)
        game.step(Action.CHECK)
        state, rewards, done = game.step(Action.CHECK)
        self.assertTrue(done)
        self.assertEqual(state.pot, 2)
        self.assertEqual(rewards, (0, 0))

    def test_illegal_action(self) -> None:
        game = LeducPokerGame()
        game.reset(cards=(Card.K1, Card.Q1), board_card=Card.J1)
        with self.assertRaises(ValueError):
            game.step(Action.CALL)


if __name__ == "__main__":
    unittest.main()
