import tempfile
import unittest

from holdem_limit.cfr import AverageStrategyPolicy, CFRTrainer
from holdem_limit.abstraction import info_state_key
from holdem_limit.game import HoldemLimitGame


class HoldemCFRTests(unittest.TestCase):
    def test_info_state_key(self) -> None:
        game = HoldemLimitGame()
        state = game.reset()
        key = info_state_key(state, 0)
        self.assertTrue(key.startswith("r0|"))

    def test_train_produces_policy(self) -> None:
        trainer = CFRTrainer(seed=0)
        trainer.train(iterations=4, log_every=0)
        policy = trainer.average_policy()
        self.assertGreater(len(policy.info_states()), 0)

    def test_policy_save_load_roundtrip(self) -> None:
        policy = AverageStrategyPolicy.from_json_dict(
            {
                "r0|pf_pair_hi|pre|tc1|ra0|h-|-|-|-": {
                    "call": 0.5,
                    "raise": 0.3,
                    "fold": 0.2,
                }
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/holdem_cfr.json"
            policy.save(path)
            loaded = AverageStrategyPolicy.load(path)
            self.assertEqual(policy.to_json_dict(), loaded.to_json_dict())


if __name__ == "__main__":
    unittest.main()
