from __future__ import annotations

import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from core.wheelchair_outputs import direction_output  # noqa: E402


class FixedWheelchairOutputTests(unittest.TestCase):
    def test_fixed_targets_keep_other_axis_at_neutral(self) -> None:
        self.assertEqual(direction_output("STOP"), (2500, 2500))
        self.assertEqual(direction_output("FORWARD"), (2500, 550))
        self.assertEqual(direction_output("BACKWARD"), (2500, 4450))
        self.assertEqual(direction_output("LEFT"), (550, 2500))
        self.assertEqual(direction_output("RIGHT"), (4450, 2500))

    def test_invalid_direction_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            direction_output("DIAGONAL")


if __name__ == "__main__":
    unittest.main()
