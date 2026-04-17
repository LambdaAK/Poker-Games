import tempfile
import unittest

from holdem_limit.game import Action
from holdem_limit.nfsp import NfspAveragePolicy, evaluate_vs_random, train_self_play


class HoldemNFSPTests(unittest.TestCase):
    def test_observe_biases_policy(self) -> None:
        policy = NfspAveragePolicy(reservoir_capacity=16)
        key = "r0|pf_pair_hi|pre|tc1|ra0|h-|-|-|-"
        legal = (Action.CALL, Action.RAISE, Action.FOLD)

        policy.observe(key, Action.RAISE)
        policy.observe(key, Action.RAISE)
        policy.observe(key, Action.CALL)

        probs = policy.action_probabilities(key, legal)
        self.assertGreater(probs[Action.RAISE], probs[Action.CALL])

    def test_save_load_roundtrip(self) -> None:
        policy = NfspAveragePolicy(reservoir_capacity=8)
        key = "r0|pf_high|pre|tc1|ra0|h-|-|-|-"
        policy.observe(key, Action.CALL)
        policy.observe(key, Action.FOLD)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/holdem_nfsp.json"
            policy.save(path)
            loaded = NfspAveragePolicy.load(path)
            self.assertEqual(policy.to_json_dict(), loaded.to_json_dict())

    def test_train_and_eval_smoke(self) -> None:
        policy, logs = train_self_play(
            episodes=30,
            q_learning_rate=0.1,
            anticipatory=0.5,
            reservoir_capacity=128,
            seed=0,
            log_every=10,
        )
        self.assertGreater(len(logs), 0)
        self.assertGreater(len(policy.info_states()), 0)
        score = evaluate_vs_random(policy, episodes=50, seed=1, as_player=0)
        self.assertIsInstance(score, float)


if __name__ == "__main__":
    unittest.main()
