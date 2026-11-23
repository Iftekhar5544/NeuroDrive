from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from core.eeg_ml import grouped_train_test_indices  # noqa: E402


class GroupedEegSplitTests(unittest.TestCase):
    def test_trials_never_cross_train_and_test(self) -> None:
        labels = []
        groups = []
        for class_id in range(3):
            for trial in range(5):
                labels.extend([class_id] * 8)
                groups.extend([f"class{class_id}-trial{trial}"] * 8)
        labels_arr = np.asarray(labels, dtype=np.int32)
        groups_arr = np.asarray(groups, dtype=object)

        train_index, test_index = grouped_train_test_indices(
            labels_arr,
            groups_arr,
            test_size=0.25,
            random_seed=42,
        )

        self.assertTrue(set(groups_arr[train_index]).isdisjoint(set(groups_arr[test_index])))
        self.assertEqual(set(np.unique(labels_arr[train_index])), {0, 1, 2})
        self.assertEqual(set(np.unique(labels_arr[test_index])), {0, 1, 2})

    def test_requires_multiple_trials(self) -> None:
        with self.assertRaises(ValueError):
            grouped_train_test_indices(
                np.asarray([0, 1]),
                np.asarray(["only", "only"], dtype=object),
                test_size=0.25,
                random_seed=1,
            )


if __name__ == "__main__":
    unittest.main()

