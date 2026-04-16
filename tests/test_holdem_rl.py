import tempfile
import unittest

from holdem_limit.game import Action, HoldemLimitGame
from holdem_limit.rl import Decision, TabularSoftmaxPolicy
from holdem_limit.abstraction import info_state_key


class HoldemRLTests(unittest.TestCase):
    def test_info_state_key_exists(self) -> None:
        game = HoldemLimitGame()
        state = game.reset()
        key0 = info_state_key(state, 0)
        key1 = info_state_key(state, 1)
        self.assertIn("r0|", key0)
        self.assertIn("r0|", key1)
        self.assertTrue(len(key0) > 8)
        self.assertTrue(len(key1) > 8)

    def test_update_moves_probability(self) -> None:
        policy = TabularSoftmaxPolicy()
        key = "r0|pf_pair_hi|pre|tc1|ra0|h-|-|-|-"
        legal = (Action.CALL, Action.RAISE, Action.FOLD)
        before = policy.action_probabilities(key, legal)[Action.RAISE]
        policy.update_episode(
            decisions=[
                Decision(
                    player=0,
                    info_state=key,
                    legal_actions=legal,
                    action=Action.RAISE,
                )
            ],
            rewards=(2, -2),
            learning_rate=0.2,
        )
        after = policy.action_probabilities(key, legal)[Action.RAISE]
        self.assertGreater(after, before)

    def test_save_load_roundtrip(self) -> None:
        policy = TabularSoftmaxPolicy()
        key = "r0|pf_high|pre|tc1|ra0|h-|-|-|-"
        policy.update_episode(
            decisions=[
                Decision(
                    player=0,
                    info_state=key,
                    legal_actions=(Action.CALL, Action.RAISE, Action.FOLD),
                    action=Action.CALL,
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
