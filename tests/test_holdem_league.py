import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from train_holdem_league import parse_int_list


class HoldemLeagueTests(unittest.TestCase):
    def test_parse_int_list(self) -> None:
        self.assertEqual(parse_int_list("0, 2,5"), [0, 2, 5])
        with self.assertRaises(ValueError):
            parse_int_list(" , ")

    def test_cli_smoke_generates_summary_and_best(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "league"
            cmd = [
                sys.executable,
                "train_holdem_league.py",
                "--seeds",
                "0",
                "--benchmark-seeds",
                "7",
                "--out-dir",
                str(out_dir),
                "--eval-episodes",
                "40",
                "--benchmark-episodes",
                "40",
                "--rl-episodes",
                "40",
                "--rl-log-every",
                "20",
                "--cfr-iterations",
                "6",
                "--cfr-log-every",
                "3",
                "--nfsp-episodes",
                "40",
                "--nfsp-log-every",
                "20",
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)

            summary_path = out_dir / "league_summary.json"
            best_path = out_dir / "best_holdem_policy.json"
            meta_path = out_dir / "best_holdem_policy.meta.json"
            self.assertTrue(summary_path.exists())
            self.assertTrue(best_path.exists())
            self.assertTrue(meta_path.exists())

            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertIn("artifacts", payload)
            self.assertIn("benchmark", payload)
            self.assertIn("best", payload)
            self.assertGreaterEqual(len(payload["artifacts"]), 3)
            self.assertIn("play_flag", payload["best"])


if __name__ == "__main__":
    unittest.main()
