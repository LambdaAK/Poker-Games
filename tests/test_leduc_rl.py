import tempfile
import unittest

from leduc_poker.game import Action, Card, initial_state
from leduc_poker.rl import Decision, TabularSoftmaxPolicy, info_state_key


class LeducRLTests(unittest.TestCase):
    def test_info_state_key(self) -> None:
        state = initial_state(cards=(Card.K1, Card.Q1), board_card=Card.J1)
        self.assertEqual(info_state_key(state, 0), "K/?|r0||")
        self.assertEqual(info_state_key(state, 1), "Q/?|r0||")

    def test_update_moves_toward_selected_action(self) -> None:
        policy = TabularSoftmaxPolicy()
        key = "K/?|r0||"
        legal = (Action.CHECK, Action.BET)
        before = policy.action_probabilities(key, legal)[Action.BET]
        policy.update_episode(
            decisions=[
                Decision(
                    player=0,
                    info_state=key,
                    legal_actions=legal,
                    action=Action.BET,
                )
            ],
            rewards=(2, -2),
            learning_rate=0.2,
        )
        after = policy.action_probabilities(key, legal)[Action.BET]
        self.assertGreater(after, before)

    def test_save_load_roundtrip(self) -> None:
        policy = TabularSoftmaxPolicy()
        key = "Q/?|r0||"
        policy.update_episode(
            decisions=[
                Decision(
                    player=0,
                    info_state=key,
                    legal_actions=(Action.CHECK, Action.BET),
                    action=Action.CHECK,
                )
            ],
            rewards=(1, -1),
            learning_rate=0.1,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/policy.json"
            policy.save(path)
            loaded = TabularSoftmaxPolicy.load(path)
            self.assertEqual(policy.to_json_dict(), loaded.to_json_dict())


if __name__ == "__main__":
    unittest.main()
