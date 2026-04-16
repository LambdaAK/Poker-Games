import unittest

from holdem_limit.game import (
    Action,
    HoldemLimitGame,
    evaluate_five,
    parse_card,
)


class HoldemLimitGameTests(unittest.TestCase):
    def test_initial_state_and_legal_actions(self) -> None:
        game = HoldemLimitGame()
        state = game.reset(
            hands=((parse_card("As"), parse_card("Kd")), (parse_card("Qc"), parse_card("Jh"))),
            board_cards=(
                parse_card("2c"),
                parse_card("3d"),
                parse_card("4h"),
                parse_card("5s"),
                parse_card("6c"),
            ),
        )
        self.assertEqual(state.pot, 3)
        self.assertEqual(state.contributions, (1, 2))
        self.assertEqual(state.to_call, 1)
        self.assertEqual(state.current_player, 0)
        self.assertEqual(game.legal_actions(), (Action.CALL, Action.RAISE, Action.FOLD))

    def test_blind_completion_then_big_blind_option(self) -> None:
        game = HoldemLimitGame()
        game.reset(
            hands=((parse_card("As"), parse_card("Kd")), (parse_card("Qc"), parse_card("Jh"))),
            board_cards=(
                parse_card("2c"),
                parse_card("3d"),
                parse_card("4h"),
                parse_card("5s"),
                parse_card("6c"),
            ),
        )
        state, rewards, done = game.step(Action.CALL)
        self.assertFalse(done)
        self.assertEqual(rewards, (0, 0))
        self.assertEqual(state.round_index, 0)
        self.assertEqual(state.current_player, 1)
        self.assertEqual(state.to_call, 0)
        self.assertEqual(game.legal_actions(), (Action.CHECK, Action.BET))

        state, rewards, done = game.step(Action.CHECK)
        self.assertFalse(done)
        self.assertEqual(rewards, (0, 0))
        self.assertEqual(state.round_index, 1)
        self.assertEqual(len(state.board), 3)
        self.assertEqual(state.current_player, 1)

    def test_fold_preflop(self) -> None:
        game = HoldemLimitGame()
        game.reset(
            hands=((parse_card("As"), parse_card("Kd")), (parse_card("Qc"), parse_card("Jh"))),
            board_cards=(
                parse_card("2c"),
                parse_card("3d"),
                parse_card("4h"),
                parse_card("5s"),
                parse_card("6c"),
            ),
        )
        state, rewards, done = game.step(Action.FOLD)
        self.assertTrue(done)
        self.assertTrue(state.terminal)
        self.assertEqual(rewards, (-1, 1))

    def test_raise_cap_enforced(self) -> None:
        game = HoldemLimitGame()
        game.reset(
            hands=((parse_card("As"), parse_card("Kd")), (parse_card("Qc"), parse_card("Jh"))),
            board_cards=(
                parse_card("2c"),
                parse_card("3d"),
                parse_card("4h"),
                parse_card("5s"),
                parse_card("6c"),
            ),
        )
        game.step(Action.RAISE)  # raises=1
        game.step(Action.RAISE)  # raises=2
        state, _, _ = game.step(Action.RAISE)  # raises=3
        self.assertEqual(state.raises_in_round, 3)
        self.assertEqual(game.legal_actions(), (Action.CALL, Action.FOLD))

    def test_showdown_high_card_win(self) -> None:
        game = HoldemLimitGame()
        game.reset(
            hands=((parse_card("As"), parse_card("3d")), (parse_card("Kc"), parse_card("4h"))),
            board_cards=(
                parse_card("2c"),
                parse_card("7d"),
                parse_card("9h"),
                parse_card("Jc"),
                parse_card("Qd"),
            ),
        )
        game.step(Action.CALL)
        game.step(Action.CHECK)
        game.step(Action.CHECK)
        game.step(Action.CHECK)
        game.step(Action.CHECK)
        game.step(Action.CHECK)
        game.step(Action.CHECK)
        state, rewards, done = game.step(Action.CHECK)
        self.assertTrue(done)
        self.assertTrue(state.terminal)
        self.assertEqual(rewards, (2, -2))

    def test_showdown_tie_split_pot(self) -> None:
        game = HoldemLimitGame()
        game.reset(
            hands=((parse_card("As"), parse_card("Kd")), (parse_card("Ah"), parse_card("Kc"))),
            board_cards=(
                parse_card("2c"),
                parse_card("3d"),
                parse_card("4h"),
                parse_card("5s"),
                parse_card("6c"),
            ),
        )
        game.step(Action.CALL)
        game.step(Action.CHECK)
        game.step(Action.CHECK)
        game.step(Action.CHECK)
        game.step(Action.CHECK)
        game.step(Action.CHECK)
        game.step(Action.CHECK)
        _, rewards, done = game.step(Action.CHECK)
        self.assertTrue(done)
        self.assertEqual(rewards, (0, 0))

    def test_evaluator_straight_flush_beats_quads(self) -> None:
        straight_flush = (
            parse_card("Ah"),
            parse_card("Kh"),
            parse_card("Qh"),
            parse_card("Jh"),
            parse_card("Th"),
        )
        quads = (
            parse_card("As"),
            parse_card("Ad"),
            parse_card("Ac"),
            parse_card("Ah"),
            parse_card("2d"),
        )
        self.assertGreater(evaluate_five(straight_flush), evaluate_five(quads))


if __name__ == "__main__":
    unittest.main()
