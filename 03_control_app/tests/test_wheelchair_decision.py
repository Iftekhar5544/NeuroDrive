from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from PyQt5 import QtCore, QtTest  # noqa: E402

from core.wheelchair_decision import WheelchairDecisionController  # noqa: E402


class WheelchairDecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])

    def setUp(self) -> None:
        self.controller = WheelchairDecisionController(
            minimum_confidence=0.75,
            history_size=5,
            votes_required=2,
        )
        self.decisions: list[dict] = []
        self.controller.decision_ready.connect(self.decisions.append)
        self.controller.set_enabled(True)
        self.decisions.clear()

    def tearDown(self) -> None:
        self.controller.set_enabled(False)

    def test_requires_repeated_predictions_before_motion(self) -> None:
        self.assertEqual(self.controller.votes_required, 2)
        self.controller.process_prediction({"label": "Forward", "confidence": 0.91})
        self.assertTrue(self.decisions)
        self.assertTrue(all(item["direction"] == "STOP" for item in self.decisions))

        self.controller.process_prediction({"label": "Forward", "confidence": 0.91})
        self.assertEqual(self.decisions[-1]["direction"], "FORWARD")
        self.assertEqual(self.decisions[-1]["duration_ms"], 3000)

    def _start_direction(self, label: str = "Forward") -> None:
        for _ in range(self.controller.votes_required):
            self.controller.process_prediction({"label": label, "confidence": 0.95})
        self.assertEqual(self.controller.last_decision, label.upper())

    def test_no_voting_is_default_and_first_qualifying_prediction_starts_run(self) -> None:
        controller = WheelchairDecisionController()
        try:
            self.assertEqual(controller.votes_required, 1)
            controller.set_enabled(True)
            controller.process_prediction({"label": "Left", "confidence": 0.74})
            self.assertEqual(controller.last_decision, "STOP")
            with patch("core.wheelchair_decision.time.monotonic", return_value=100.0):
                controller.process_prediction({"label": "Left", "confidence": 0.75})
            self.assertEqual(controller.last_decision, "LEFT")
            self.assertEqual(controller.movement_deadline_at, 103.0)
            self.assertEqual(len(controller.history), 0)
        finally:
            controller.set_enabled(False)

    def test_each_voting_count_requires_its_selected_number_of_predictions(self) -> None:
        with patch("core.wheelchair_decision.time.monotonic", return_value=100.0):
            for votes in (2, 3, 4, 5):
                for label in ("Forward", "Left", "Right"):
                    with self.subTest(votes=votes, label=label):
                        self.controller.configure(0.75, votes)
                        for _ in range(votes - 1):
                            self.controller.process_prediction({"label": label, "confidence": 0.95})
                            self.assertEqual(self.controller.last_decision, "STOP")
                        self.controller.process_prediction({"label": label, "confidence": 0.95})
                        self.assertEqual(self.controller.last_decision, label.upper())
                        self.assertEqual(self.controller.movement_deadline_at, 103.0)

    def test_changing_voting_stops_run_and_drops_previous_votes(self) -> None:
        self._start_direction()
        self.controller.configure(0.75, 5)
        self.assertEqual(self.controller.last_decision, "STOP")
        self.assertFalse(self.controller.movement_timer.isActive())
        self.assertEqual(len(self.controller.history), 0)
        for _ in range(4):
            self.controller.process_prediction({"label": "Forward", "confidence": 0.95})
            self.assertEqual(self.controller.last_decision, "STOP")
        self.controller.configure(0.75, 1)
        self.assertEqual(len(self.controller.history), 0)
        self.controller.process_prediction({"label": "Right", "confidence": 0.95})
        self.assertEqual(self.controller.last_decision, "RIGHT")

    def test_all_eeg_directions_use_saved_duration(self) -> None:
        with patch("core.wheelchair_decision.time.monotonic", return_value=100.0):
            for duration_ms in (100, 2500, 7500, 60000):
                self.controller.set_movement_duration(duration_ms)
                for label in ("Forward", "Left", "Right"):
                    with self.subTest(duration_ms=duration_ms, label=label):
                        self.controller.reset()
                        self._start_direction(label)
                        self.assertEqual(self.controller.movement_deadline_at, 100.0 + duration_ms / 1000)
                        self.assertEqual(self.decisions[-1]["duration_ms"], duration_ms)
                        self.assertEqual(self.controller.movement_timer.interval(), duration_ms)
                        self.assertTrue(self.controller.movement_timer.isActive())

    def test_saving_duration_stops_active_run_and_requires_fresh_votes(self) -> None:
        self._start_direction()
        self.controller.set_movement_duration(2500)
        self.assertEqual(self.decisions[-1]["direction"], "STOP")
        self.assertFalse(self.controller.movement_timer.isActive())
        self.assertEqual(self.controller.movement_deadline_at, 0.0)
        self.assertEqual(len(self.controller.history), 0)
        self.controller.process_prediction({"label": "Left", "confidence": 0.95})
        self.assertEqual(self.controller.last_decision, "STOP")
        self.controller.process_prediction({"label": "Left", "confidence": 0.95})
        self.assertEqual(self.controller.last_decision, "LEFT")
        self.assertEqual(self.decisions[-1]["duration_ms"], 2500)

    def test_invalid_duration_does_not_change_active_run(self) -> None:
        self._start_direction()
        deadline = self.controller.movement_deadline_at
        for duration_ms in (0, -100, 99, 60001, True, 2500.0, "2500"):
            with self.subTest(duration_ms=duration_ms), self.assertRaises(ValueError):
                self.controller.set_movement_duration(duration_ms)
            self.assertEqual(self.controller.movement_duration_ms, 3000)
            self.assertEqual(self.controller.movement_deadline_at, deadline)
            self.assertEqual(self.controller.last_decision, "FORWARD")

    def test_live_predictions_do_not_extend_or_redirect_run_and_new_votes_are_required(self) -> None:
        with patch("core.wheelchair_decision.time.monotonic", return_value=100.0) as clock:
            self._start_direction()
            accepted_count = len(self.decisions)
            for elapsed in (0.1, 1.0, 2.0, 2.5, 2.9):
                clock.return_value = 100.0 + elapsed
                label = "Right" if elapsed < 2.5 else "Forward"
                self.controller.process_prediction({"label": label, "confidence": 0.95})
                self.assertEqual(self.controller.last_decision, "FORWARD")
                self.assertEqual(self.controller.movement_deadline_at, 103.0)
            self.assertEqual(len(self.decisions), accepted_count)
            clock.return_value = 103.0
            self.controller._check_prediction_timeout()
            self.assertEqual(self.decisions[-1]["direction"], "STOP")
            self.assertIn("3-second", self.decisions[-1]["reason"])
            self.assertFalse(self.controller.movement_timer.isActive())
            self.assertEqual(len(self.controller.history), 0)
            clock.return_value = 103.1
            self.controller.process_prediction({"label": "Left", "confidence": 0.95})
            self.assertEqual(self.decisions[-1]["direction"], "STOP")
            clock.return_value = 103.2
            self.controller.process_prediction({"label": "Left", "confidence": 0.95})
            self.assertEqual(self.decisions[-1]["direction"], "LEFT")
            self.assertAlmostEqual(self.controller.movement_deadline_at, 106.2)

    def test_prediction_at_deadline_counts_as_a_fresh_vote(self) -> None:
        with patch("core.wheelchair_decision.time.monotonic", return_value=100.0) as clock:
            self._start_direction()
            clock.return_value = 103.0
            self.controller.process_prediction({"label": "Forward", "confidence": 0.95})
            self.assertEqual(self.decisions[-1]["direction"], "STOP")
            self.assertEqual(list(self.controller.history), ["FORWARD"])
            clock.return_value = 103.1
            self.controller.process_prediction({"label": "Forward", "confidence": 0.95})
            self.assertEqual(self.decisions[-1]["direction"], "FORWARD")
            self.assertAlmostEqual(self.controller.movement_deadline_at, 106.1)

    def test_repeated_rest_has_no_hold_or_cooldown_with_any_voting_option(self) -> None:
        with patch("core.wheelchair_decision.time.monotonic", return_value=100.0):
            for votes in (1, 2, 3, 4, 5):
                with self.subTest(votes=votes):
                    self.controller.configure(0.75, votes)
                    self.decisions.clear()
                    for _ in range(2):
                        self.controller.process_prediction({"label": "Rest", "confidence": 0.01})
                        self.assertEqual(self.decisions[-1]["direction"], "STOP")
                        self.assertFalse(self.controller.movement_timer.isActive())
                        self.assertEqual(self.controller.movement_deadline_at, 0.0)
                    self.assertEqual([item["direction"] for item in self.decisions], ["STOP", "STOP"])
                    self._start_direction()
                    self.assertEqual(self.controller.movement_deadline_at, 103.0)
                    self.controller.process_prediction({"label": "Rest", "confidence": 0.01})
                    self.assertTrue(self.controller.movement_timer.isActive())
                    self.assertEqual(self.controller.movement_deadline_at, 103.0)
                    self.assertEqual(self.controller.last_decision, "FORWARD")

    def test_no_voting_accepts_prediction_at_expiry_without_waiting_for_next_result(self) -> None:
        self.controller.configure(0.75, 1)
        with patch("core.wheelchair_decision.time.monotonic", return_value=100.0) as clock:
            self._start_direction()
            self.decisions.clear()
            clock.return_value = 103.0
            self.controller.process_prediction({"label": "Right", "confidence": 0.95})
            self.assertEqual([item["direction"] for item in self.decisions], ["STOP", "RIGHT"])
            self.assertEqual(self.controller.movement_deadline_at, 106.0)

    def test_rest_at_expiry_has_priority_and_does_not_start_a_new_hold(self) -> None:
        self.controller.configure(0.75, 1)
        with patch("core.wheelchair_decision.time.monotonic", return_value=100.0) as clock:
            self._start_direction()
            self.decisions.clear()
            clock.return_value = 103.0
            self.controller.process_prediction({"label": "Rest", "confidence": 0.01})
            self.assertEqual([item["direction"] for item in self.decisions], ["STOP", "STOP"])
            self.assertIn("Safe class detected", self.decisions[-1]["reason"])
            self.assertFalse(self.controller.movement_timer.isActive())
            self.controller.process_prediction({"label": "Forward", "confidence": 0.95})
            self.assertEqual(self.controller.last_decision, "FORWARD")
            self.assertEqual(self.controller.movement_deadline_at, 106.0)

    def test_io_failure_at_expiry_cannot_start_another_hold(self) -> None:
        self.controller.configure(0.75, 1)
        with patch("core.wheelchair_decision.time.monotonic", return_value=100.0) as clock:
            self._start_direction()
            # Simulate a failed expiry STOP disabling EEG control via the panel.
            def disable_on_stop(payload):
                if payload["direction"] == "STOP" and self.controller.enabled:
                    self.controller.set_enabled(False)
            self.controller.decision_ready.connect(disable_on_stop)
            clock.return_value = 103.0
            self.controller.process_prediction({"label": "Right", "confidence": 0.95})
            self.assertFalse(self.controller.enabled)
            self.assertFalse(self.controller.movement_timer.isActive())
            self.assertEqual(self.controller.last_decision, "STOP")

    def test_stop_conditions_cancel_an_active_run(self) -> None:
        self.controller.set_movement_duration(7500)
        for condition in ("Settings", "Disable", "Timeout"):
            with self.subTest(condition=condition):
                with patch("core.wheelchair_decision.time.monotonic", return_value=100.0) as clock:
                    self.controller.set_enabled(True)
                    self.controller.reset()
                    self._start_direction()
                    if condition == "Settings":
                        self.controller.configure(0.75, 2)
                    elif condition == "Disable":
                        self.controller.set_enabled(False)
                    elif condition == "Timeout":
                        clock.return_value = 101.6
                        self.controller._check_prediction_timeout()
                    self.assertEqual(self.decisions[-1]["direction"], "STOP")
                    self.assertFalse(self.controller.movement_timer.isActive())
                    self.assertEqual(self.controller.movement_deadline_at, 0.0)
                    decision_count = len(self.decisions)
                    self.controller._finish_movement()
                    self.assertEqual(len(self.decisions), decision_count)

    def test_prediction_results_cannot_interrupt_or_queue_during_five_second_run(self) -> None:
        self.controller.set_movement_duration(5000)
        for votes in (1, 3):
            with self.subTest(votes=votes):
                self.controller.configure(0.75, votes)
                with patch("core.wheelchair_decision.time.monotonic", return_value=100.0) as clock:
                    self._start_direction()
                    accepted_count = len(self.decisions)
                    for elapsed, label, confidence in (
                        (0.5, "Rest", 0.99), (1.0, "Stop", 0.99),
                        (2.0, "Left", 0.1), (3.0, "Unknown", 0.99),
                        (4.0, "Right", 0.99), (4.9, "Rest", 0.99),
                    ):
                        clock.return_value = 100.0 + elapsed
                        self.controller.process_prediction({"label": label, "confidence": confidence})
                        self.controller._check_prediction_timeout()
                        self.assertEqual(self.controller.last_decision, "FORWARD")
                        self.assertEqual(self.controller.movement_deadline_at, 105.0)
                        self.assertTrue(self.controller.movement_timer.isActive())
                        self.assertEqual(len(self.decisions), accepted_count)
                        self.assertEqual(len(self.controller.history), 0)
                    clock.return_value = 105.0
                    self.controller._check_prediction_timeout()
                    self.assertEqual(self.decisions[-1]["direction"], "STOP")
                    self.assertEqual(len(self.controller.history), 0)
                    for _ in range(votes - 1):
                        self.controller.process_prediction({"label": "Right", "confidence": 0.99})
                        self.assertEqual(self.controller.last_decision, "STOP")
                    self.controller.process_prediction({"label": "Right", "confidence": 0.99})
                    self.assertEqual(self.controller.last_decision, "RIGHT")
                    self.assertEqual(self.controller.movement_deadline_at, 110.0)

    def test_qt_timer_stops_run_without_another_prediction_callback(self) -> None:
        # Isolate duration expiry from the separate 1.5-second prediction watchdog.
        controller = WheelchairDecisionController(prediction_timeout_ms=4000)
        try:
            controller.set_movement_duration(200)
            controller.set_enabled(True)
            for _ in range(2):
                controller.process_prediction({"label": "Right", "confidence": 0.95})
            self.assertEqual(controller.last_decision, "RIGHT")
            QtTest.QTest.qWait(350)
            self.assertEqual(controller.last_decision, "STOP")
            self.assertFalse(controller.movement_timer.isActive())
        finally:
            controller.set_enabled(False)

    def test_low_confidence_stops_immediately(self) -> None:
        self.controller.process_prediction({"label": "Left", "confidence": 0.50})
        self.assertEqual(self.decisions[-1]["direction"], "STOP")
        self.assertIn("below", self.decisions[-1]["reason"])

    def test_rest_stops_immediately(self) -> None:
        self.controller.process_prediction({"label": "Rest", "confidence": 0.99})
        self.assertEqual(self.decisions[-1]["direction"], "STOP")
        self.assertIn("Safe class", self.decisions[-1]["reason"])

    def test_unmapped_class_stops(self) -> None:
        self.controller.process_prediction({"label": "Blink", "confidence": 0.99})
        self.assertEqual(self.decisions[-1]["direction"], "STOP")
        self.assertIn("Unmapped", self.decisions[-1]["reason"])


if __name__ == "__main__":
    unittest.main()

