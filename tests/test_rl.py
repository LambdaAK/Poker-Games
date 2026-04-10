import tempfile
import unittest

from kuhn_poker import Action, Card
from kuhn_poker.rl import Decision, TabularSoftmaxPolicy, info_state_key


class TabularPolicyTests(unittest.TestCase):
    def test_info_state_key(self) -> None:
        self.assertEqual(info_state_key(Card.K, ""), "K|")
        self.assertEqual(info_state_key(Card.J, "cb"), "J|cb")

    def test_uniform_probabilities_on_unseen_state(self) -> None:
        policy = TabularSoftmaxPolicy()
        probs = policy.action_probabilities("Q|", (Action.CHECK, Action.BET))
        self.assertAlmostEqual(probs[Action.CHECK], 0.5)
        self.assertAlmostEqual(probs[Action.BET], 0.5)

    def test_update_pushes_probability_toward_good_action(self) -> None:
        policy = TabularSoftmaxPolicy()
        info_state = "K|"
        legal = (Action.CHECK, Action.BET)
        before = policy.action_probabilities(info_state, legal)[Action.BET]

        policy.update_episode(
            decisions=[
                Decision(
                    player=0,
                    info_state=info_state,
                    legal_actions=legal,
                    action=Action.BET,
                )
            ],
            rewards=(1, -1),
            learning_rate=0.5,
        )
        after = policy.action_probabilities(info_state, legal)[Action.BET]
        self.assertGreater(after, before)

    def test_save_load_roundtrip(self) -> None:
        policy = TabularSoftmaxPolicy()
        policy.update_episode(
            decisions=[
                Decision(
                    player=0,
                    info_state="Q|c",
                    legal_actions=(Action.CHECK, Action.BET),
                    action=Action.CHECK,
                )
            ],
            rewards=(1, -1),
            learning_rate=0.2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/policy.json"
            policy.save(path)
            loaded = TabularSoftmaxPolicy.load(path)
            self.assertEqual(policy.to_json_dict(), loaded.to_json_dict())


if __name__ == "__main__":
    unittest.main()
