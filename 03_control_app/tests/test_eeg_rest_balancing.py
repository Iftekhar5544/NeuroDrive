from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from core import eeg_ml


class RestUndersamplingTests(unittest.TestCase):
    def test_only_excess_rest_is_trimmed_without_replacement(self):
        names = ["Forward", "Left", "Rest", "Right"]
        labels = np.repeat(np.arange(4), [10, 10, 60, 10])
        original = labels.copy()
        selected, details = eeg_ml.undersample_rest_training_indices(labels, names, 42)
        self.assertEqual(details["class_window_counts_after"], dict.fromkeys(names, 10))
        self.assertEqual(details["removed_windows"], 50)
        self.assertEqual(details["target_windows"], 10)
        self.assertTrue(details["applied"])
        self.assertEqual(len(selected), len(np.unique(selected)))
        np.testing.assert_array_equal(selected[labels[selected] != 2], np.flatnonzero(labels != 2))
        np.testing.assert_array_equal(labels, original)

    def test_unequal_movement_classes_set_rest_cap_without_removing_movements(self):
        names = ["Forward", "Left", "Rest", "Right"]
        labels = np.repeat(np.arange(4), [12, 8, 60, 11])
        _, details = eeg_ml.undersample_rest_training_indices(labels, names, 42)
        self.assertEqual(details["class_window_counts_after"], {"Forward": 12, "Left": 8, "Rest": 8, "Right": 11})

    def test_rest_matching_ignores_case_and_surrounding_whitespace(self):
        for label in ("Rest", "rest", "REST", " Rest "):
            with self.subTest(label=label):
                labels = np.array([0] * 10 + [1] * 50)
                _, details = eeg_ml.undersample_rest_training_indices(labels, ["Forward", label], 42)
                self.assertEqual(details["class_window_counts_after"][label], 10)

    def test_no_rest_or_no_excess_leaves_all_windows_in_place(self):
        cases = [
            (["Forward", "Left", "Right"], [10, 50, 10]),
            (["Forward", "Rest"], [10, 10]),
            (["Forward", "Rest"], [10, 4]),
            (["Forward", "Stop"], [10, 50]),
            (["Rest"], [50]),
        ]
        for names, counts in cases:
            with self.subTest(names=names, counts=counts):
                labels = np.repeat(np.arange(len(names)), counts)
                selected, details = eeg_ml.undersample_rest_training_indices(labels, names, 42)
                np.testing.assert_array_equal(selected, np.arange(len(labels)))
                self.assertFalse(details["applied"])
                self.assertEqual(details["class_window_counts_after"], details["class_window_counts_before"])

    def test_seed_repeats_selection_and_changing_seed_changes_rest_subset(self):
        labels = np.array([0] * 10 + [1] * 100)
        first, _ = eeg_ml.undersample_rest_training_indices(labels, ["Forward", "Rest"], 42)
        repeated, _ = eeg_ml.undersample_rest_training_indices(labels, ["Forward", "Rest"], 42)
        different, _ = eeg_ml.undersample_rest_training_indices(labels, ["Forward", "Rest"], 43)
        np.testing.assert_array_equal(first, repeated)
        self.assertFalse(np.array_equal(first, different))


class RestBalancedTrainingTests(unittest.TestCase):
    def test_real_training_and_each_cv_fold_trim_only_training_then_reload_normally(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            dataset_path = root / "synthetic_eeg.csv"
            rng = np.random.default_rng(23)
            with dataset_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["label", "trial_id", "ch1", "ch2"])
                for class_id, label in enumerate(["Forward", "Left", "Rest", "Right"]):
                    for trial in range(5):
                        sample_count = 352 if label == "Rest" else 96
                        t = np.arange(sample_count) / 64
                        signal = np.sin(2 * np.pi * (4 + class_id * 5) * t)
                        noise = rng.normal(0, 0.1, (sample_count, 2))
                        for index in range(sample_count):
                            writer.writerow([label, f"{label}-{trial}", signal[index] + noise[index, 0],
                                             signal[index] * 0.7 + noise[index, 1]])
            source_bytes = dataset_path.read_bytes()
            built = []
            fits = []
            predictions = []
            original_build = eeg_ml.build_dataset
            original_fit = eeg_ml.Pipeline.fit
            original_predict = eeg_ml.Pipeline.predict

            def capture_build(*args, **kwargs):
                value = original_build(*args, **kwargs)
                built.append(value)
                return value

            def capture_fit(model, features, labels, **kwargs):
                fits.append((features.copy(), labels.copy()))
                return original_fit(model, features, labels, **kwargs)

            def capture_predict(model, features, **kwargs):
                predictions.append(features.copy())
                return original_predict(model, features, **kwargs)

            with patch.object(eeg_ml, "build_dataset", side_effect=capture_build), \
                    patch.object(eeg_ml.Pipeline, "fit", new=capture_fit), \
                    patch.object(eeg_ml.Pipeline, "predict", new=capture_predict):
                result = eeg_ml.train_eeg_model(
                    sample_rate=64, window_ms=500, stride_ms=250, n_estimators=8, max_depth=4,
                    test_size=0.2, random_seed=42, csv_path=str(dataset_path),
                    output_dir=str(root / "models"),
                )

            self.assertEqual(dataset_path.read_bytes(), source_bytes)
            features, names, groups = built[0]
            classes = sorted(np.unique(names).tolist())
            labels = np.array([classes.index(name) for name in names])
            holdout_train, holdout_test = eeg_ml.grouped_train_test_indices(labels, groups, 0.2, 42)
            cv = eeg_ml.StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
            partitions = [(holdout_train, holdout_test)] + list(cv.split(features, labels, groups))
            self.assertEqual(len(fits), 6)
            self.assertEqual(len(predictions), 6)
            sampling_reports = [result["rest_undersampling"]] + result["cv_fold_rest_undersampling"]
            for (train, validation), (fit_features, fit_labels), evaluated, details in zip(
                    partitions, fits, predictions, sampling_reports):
                np.testing.assert_array_equal(evaluated, features[validation])
                self.assertTrue(set(groups[train]).isdisjoint(set(groups[validation])))
                training_rows = {row.tobytes() for row in features[train]}
                fitted_rows = {row.tobytes() for row in fit_features}
                self.assertTrue(fitted_rows.issubset(training_rows))
                self.assertTrue(fitted_rows.isdisjoint(row.tobytes() for row in evaluated))
                for index in train[labels[train] != classes.index("Rest")]:
                    self.assertIn(features[index].tobytes(), fitted_rows)
                counts = eeg_ml.class_window_counts(fit_labels, classes)
                self.assertEqual(counts, details["class_window_counts_after"])
                self.assertEqual(counts["Rest"], min(count for name, count in counts.items() if name != "Rest"))

            self.assertEqual(result["train_windows"], len(fits[0][1]))
            self.assertEqual(result["test_windows"], len(holdout_test))
            self.assertEqual(result["test_class_window_counts"], eeg_ml.class_window_counts(labels[holdout_test], classes))
            for fold, (_, validation) in zip(result["cv_fold_rest_undersampling"], partitions[1:]):
                self.assertEqual(fold["validation_class_window_counts"], eeg_ml.class_window_counts(labels[validation], classes))

            run = Path(result["run_dir"])
            saved = json.loads((run / "training_results.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["rest_undersampling"], result["rest_undersampling"])
            self.assertEqual(saved["cv_fold_rest_undersampling"], result["cv_fold_rest_undersampling"])
            self.assertIn("Training windows per class (before -> after)", (run / "training_summary.txt").read_text(encoding="utf-8"))
            bundle = eeg_ml.load_model_bundle(result["model_path"])
            self.assertEqual(bundle.model.named_steps["scaler"].n_samples_seen_, result["train_windows"])
            self.assertEqual(bundle.artifact["metrics"]["train_class_window_counts_after"], result["train_class_window_counts_after"])
            prediction = eeg_ml.predict_from_window(bundle, rng.normal(size=(2, 32)))
            self.assertIn(prediction["label"], classes)
            self.assertEqual(len(prediction["probabilities"]), 4)


if __name__ == "__main__":
    unittest.main()
