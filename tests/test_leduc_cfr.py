import tempfile
import unittest

from leduc_poker.cfr import AverageStrategyPolicy, CFRTrainer, info_state_key
from leduc_poker.game import Card, initial_state


class LeducCFRTests(unittest.TestCase):
    def test_info_state_key(self) -> None:
        state = initial_state(cards=(Card.J1, Card.K1), board_card=Card.Q1)
        self.assertEqual(info_state_key(state, 0), "J/?|r0||")
        self.assertEqual(info_state_key(state, 1), "K/?|r0||")

    def test_train_produces_nonempty_policy(self) -> None:
        trainer = CFRTrainer()
        trainer.train(iterations=2, log_every=0)
        policy = trainer.average_policy()
        self.assertGreater(len(policy.info_states()), 0)

    def test_policy_save_load_roundtrip(self) -> None:
        policy = AverageStrategyPolicy.from_json_dict(
            {
                "K/?|r0||": {"check": 0.4, "bet": 0.6},
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/cfr.json"
            policy.save(path)
            loaded = AverageStrategyPolicy.load(path)
            self.assertEqual(policy.to_json_dict(), loaded.to_json_dict())


if __name__ == "__main__":
    unittest.main()
