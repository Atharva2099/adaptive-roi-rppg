import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.verify_gate8_mcd_oracles import _args, _plan


class Gate8OracleTests(unittest.TestCase):
    def test_cli_freezes_width_and_supports_shards(self):
        value = _args(["--output-dir", "o", "--manifest-tree", "m", "--state-root", "s", "--gt-root", "g", "--code-snapshot-sha256", "a" * 64, "--shard-count", "16", "--shard-index", "3"])
        self.assertEqual((value.shard_count, value.shard_index, value.beam_width), (16, 3, 8))

    def test_launcher_runs_srun_then_merge(self):
        text = Path("slurm/gate8_mcd_oracles_full.slurm").read_text()
        self.assertLess(text.index("srun"), text.index("--merge"))
        self.assertIn("--shard-count 4", text)

    def test_plan_contract_is_eval_only(self):
        self.assertIn("read_mcd_eval_labels", Path("scripts/verify_gate8_mcd_oracles.py").read_text())
        self.assertIn("split\": \"eval\"", Path("scripts/verify_gate8_mcd_oracles.py").read_text())


if __name__ == "__main__": unittest.main()
