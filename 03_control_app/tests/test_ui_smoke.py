from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from PyQt5 import QtCore, QtGui, QtTest, QtWidgets  # noqa: E402
from ui.main_window import MainWindow  # noqa: E402


class IntegratedUiSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def _close_window(self, window) -> None:
        window.close()
        # Drain deferred widget cleanup between UI tests.
        window.deleteLater()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
        self.app.processEvents()

    def test_main_window_has_fixed_manual_and_eeg_commands_without_speed_controls(self) -> None:
        window = MainWindow()
        try:
            panel = window.wheelchair_panel
            self.assertFalse(hasattr(window, "plot_widget"))
            self.assertFalse(hasattr(window, "main_splitter"))
            self.assertFalse(window.findChildren(QtWidgets.QScrollArea))
            self.assertEqual(panel.mode_combo.currentText(), "Manual")
            self.assertFalse(panel.service.armed)
            self.assertFalse(hasattr(panel, "speed_slider"))
            self.assertFalse(hasattr(panel, "spin_fwd_min"))
            self.assertEqual(panel.voting_combo.currentText(), "No voting")
            self.assertEqual(panel.decision_controller.votes_required, 1)
            self.assertEqual(panel.movement_duration_edit.text(), "3")
            self.assertEqual(panel.decision_controller.movement_duration_ms, 3000)
            panel.service.connected = True
            panel.service.armed = True
            panel.service.set_output = Mock()
            panel.service.stop_motion = Mock()
            for direction in ("FORWARD", "BACKWARD", "LEFT", "RIGHT"):
                panel._manual_press(direction)
                panel.service.set_output.assert_called_with(direction)
                panel._manual_release()
                panel.service.stop_motion.assert_called()
            panel._command_changed("FORWARD", 2500, 550)
            self.assertIn("X 2.50 V | Y 0.55 V", panel.output_label.text())

            # The same fixed command path is used for accepted EEG decisions.
            panel.mode_combo.setCurrentText("EEG Control")
            panel.service.armed = True
            panel.decision_controller.enabled = True
            panel._handle_decision({"direction": "LEFT"})
            panel.service.set_output.assert_called_with("LEFT")
            panel._handle_decision({"direction": "STOP"})
            panel.service.stop_motion.assert_called()
        finally:
            self._close_window(window)

    def test_connection_type_selects_correct_baud(self) -> None:
        window = MainWindow()
        try:
            panel = window.wheelchair_panel
            panel.port_combo.setCurrentText("TEST")
            panel.service.connect_port = Mock()
            panel._toggle_connection()
            panel.service.connect_port.assert_called_with("TEST", 9600)
            panel.transport_combo.setCurrentIndex(1)
            panel._toggle_connection()
            panel.service.connect_port.assert_called_with("TEST", 115200)
        finally:
            self._close_window(window)

    def test_confidence_threshold_applies_only_on_save_and_resets_decisions(self) -> None:
        window = MainWindow()
        try:
            panel = window.wheelchair_panel
            controller = panel.decision_controller
            decisions = []
            controller.decision_ready.connect(decisions.append)
            controller.set_enabled(True)
            for _ in range(2):
                controller.process_prediction({"label": "Forward", "confidence": 0.85})
            self.assertEqual(decisions[-1]["direction"], "FORWARD")

            window.show()
            editor = panel.minimum_confidence_edit
            editor.setFocus()
            editor.selectAll()
            QtTest.QTest.keyClicks(editor, "90")
            self.assertAlmostEqual(controller.minimum_confidence, 0.75)
            QtTest.QTest.keyClick(editor, QtCore.Qt.Key_Return)
            self.assertAlmostEqual(controller.minimum_confidence, 0.75)
            panel.open_prediction_button.setFocus()
            self.app.processEvents()
            self.assertAlmostEqual(controller.minimum_confidence, 0.75)
            self.assertEqual(decisions[-1]["direction"], "FORWARD")

            QtTest.QTest.mouseClick(panel.save_confidence_button, QtCore.Qt.LeftButton)
            self.assertAlmostEqual(controller.minimum_confidence, 0.90)
            self.assertEqual(decisions[-1]["direction"], "STOP")
            self.assertEqual(len(controller.history), 0)

            for _ in range(2):
                controller.process_prediction({"label": "Forward", "confidence": 0.85})
            self.assertEqual(decisions[-1]["direction"], "STOP")
            for _ in range(2):
                controller.process_prediction({"label": "Forward", "confidence": 0.95})
            self.assertEqual(decisions[-1]["direction"], "FORWARD")

            editor.selectAll()
            QtTest.QTest.keyClicks(editor, "25")
            QtTest.QTest.keyClick(editor, QtCore.Qt.Key_Return)
            self.assertAlmostEqual(controller.minimum_confidence, 0.90)
            QtTest.QTest.mouseClick(panel.save_confidence_button, QtCore.Qt.LeftButton)
            self.assertAlmostEqual(controller.minimum_confidence, 0.25)
            self.assertEqual(decisions[-1]["direction"], "STOP")
            controller.process_prediction({"label": "Forward", "confidence": 0.24})
            self.assertEqual(decisions[-1]["direction"], "STOP")
            for _ in range(2):
                controller.process_prediction({"label": "Forward", "confidence": 0.25})
            self.assertEqual(decisions[-1]["direction"], "FORWARD")
        finally:
            self._close_window(window)

    def test_scrolling_and_voting_do_not_apply_unsaved_threshold(self) -> None:
        window = MainWindow()
        try:
            panel = window.wheelchair_panel
            editor = panel.minimum_confidence_edit
            editor.setText("90")
            wheel = QtGui.QWheelEvent(
                QtCore.QPointF(editor.rect().center()),
                QtCore.QPointF(editor.mapToGlobal(editor.rect().center())),
                QtCore.QPoint(), QtCore.QPoint(0, 120),
                QtCore.Qt.NoButton, QtCore.Qt.NoModifier, QtCore.Qt.NoScrollPhase, False,
            )
            QtWidgets.QApplication.sendEvent(editor, wheel)
            self.assertEqual(editor.text(), "90")
            self.assertAlmostEqual(panel.decision_controller.minimum_confidence, 0.75)
            panel.voting_combo.setCurrentIndex(panel.voting_combo.findData(3))
            self.assertEqual(panel.decision_controller.votes_required, 3)
            self.assertAlmostEqual(panel.decision_controller.minimum_confidence, 0.75)
            panel.save_confidence_button.click()
            self.assertAlmostEqual(panel.decision_controller.minimum_confidence, 0.90)
            self.assertEqual(panel.decision_controller.votes_required, 3)
        finally:
            self._close_window(window)

    def test_invalid_threshold_save_preserves_active_settings(self) -> None:
        window = MainWindow()
        try:
            panel = window.wheelchair_panel
            for text in ("", "24.9", "99.1", "nan", "abc", "90.00"):
                with self.subTest(text=text):
                    panel.minimum_confidence_edit.setText(text)
                    panel.save_confidence_button.click()
                    self.assertAlmostEqual(panel.decision_controller.minimum_confidence, 0.75)
                    self.assertIn("25 to 99%", panel.decision_status_label.text())
            panel.minimum_confidence_edit.setText("99")
            panel.save_confidence_button.click()
            self.assertAlmostEqual(panel.decision_controller.minimum_confidence, 0.99)
            self.assertIn("saved: 99.0%", panel.decision_status_label.text())
        finally:
            self._close_window(window)

    def test_duration_applies_only_on_save_and_preserves_other_settings(self) -> None:
        window = MainWindow()
        try:
            window.show()
            panel = window.wheelchair_panel
            controller = panel.decision_controller
            controller.set_enabled(True)
            controller.process_prediction({"label": "Forward", "confidence": 0.95})
            editor = panel.movement_duration_edit
            editor.setFocus()
            editor.selectAll()
            QtTest.QTest.keyClicks(editor, "2.5")
            QtTest.QTest.keyClick(editor, QtCore.Qt.Key_Return)
            panel.open_prediction_button.setFocus()
            self.app.processEvents()
            self.assertEqual(controller.movement_duration_ms, 3000)
            self.assertEqual(controller.last_decision, "FORWARD")
            wheel = QtGui.QWheelEvent(
                QtCore.QPointF(editor.rect().center()),
                QtCore.QPointF(editor.mapToGlobal(editor.rect().center())),
                QtCore.QPoint(), QtCore.QPoint(0, 120),
                QtCore.Qt.NoButton, QtCore.Qt.NoModifier, QtCore.Qt.NoScrollPhase, False,
            )
            QtWidgets.QApplication.sendEvent(editor, wheel)
            self.assertEqual(editor.text(), "2.5")
            self.assertEqual(controller.movement_duration_ms, 3000)
            panel.minimum_confidence_edit.setText("90")
            QtTest.QTest.mouseClick(panel.save_duration_button, QtCore.Qt.LeftButton)
            self.assertEqual(controller.movement_duration_ms, 2500)
            self.assertEqual(controller.last_decision, "STOP")
            self.assertFalse(controller.movement_timer.isActive())
            self.assertAlmostEqual(controller.minimum_confidence, 0.75)
            panel.movement_duration_edit.setText("5")
            panel.voting_combo.setCurrentIndex(panel.voting_combo.findData(3))
            panel.save_confidence_button.click()
            self.assertEqual(controller.movement_duration_ms, 2500)
            self.assertEqual(controller.votes_required, 3)
            self.assertAlmostEqual(controller.minimum_confidence, 0.90)
        finally:
            self._close_window(window)

    def test_invalid_duration_save_preserves_saved_duration(self) -> None:
        window = MainWindow()
        try:
            panel = window.wheelchair_panel
            for text in ("", "0", "-1", "0.09", "60.1", "nan", "abc", "2.55"):
                with self.subTest(text=text):
                    panel.movement_duration_edit.setText(text)
                    panel.save_duration_button.click()
                    self.assertEqual(panel.decision_controller.movement_duration_ms, 3000)
                    self.assertIn("0.1 to 60 seconds", panel.decision_status_label.text())
        finally:
            self._close_window(window)

    def test_eeg_run_sends_direction_refreshes_then_stop_at_saved_duration(self) -> None:
        window = MainWindow()
        try:
            panel = window.wheelchair_panel
            panel.mode_combo.setCurrentText("EEG Control")
            panel.service.serial_port = Mock(is_open=True)
            panel.service.serial_port.write.side_effect = lambda payload: len(payload)
            panel.service.connected = True
            panel.service.armed = True
            panel.eeg_streaming = True
            panel.prediction_active = True
            panel.model_has_stop_class = True
            serial_port = panel.service.serial_port
            with patch("core.wheelchair_decision.time.monotonic", return_value=100.0) as clock:
                panel.movement_duration_edit.setText("5")
                panel.save_duration_button.click()
                panel._sync_decision_enabled()
                panel.handle_prediction({"label": "Forward", "confidence": 0.95})
                serial_port.write.assert_called_with(b"w")
                self.assertTrue(panel.service.heartbeat_timer.isActive())
                for elapsed, label, confidence in (
                    (1.0, "Rest", 0.99), (2.0, "Unknown", 0.99),
                    (3.0, "Left", 0.1), (4.0, "Stop", 0.99),
                ):
                    clock.return_value = 100.0 + elapsed
                    panel.handle_prediction({"label": label, "confidence": confidence})
                    panel.decision_controller._check_prediction_timeout()
                    panel.service._send_heartbeat()
                    serial_port.write.assert_called_with(b"w")
                    self.assertEqual(panel.service.current_direction, "FORWARD")
                clock.return_value = 104.9
                panel.handle_prediction({"label": "Left", "confidence": 0.95})
                panel.service._send_heartbeat()
                serial_port.write.assert_called_with(b"w")
                clock.return_value = 105.0
                panel.decision_controller._check_prediction_timeout()
                serial_port.write.assert_called_with(b"x")
                self.assertEqual(panel.service.current_direction, "STOP")
                self.assertFalse(panel.service.heartbeat_timer.isActive())
                self.assertIn("5-second movement completed", panel.decision_status_label.text())
        finally:
            self._close_window(window)

    def test_voting_selector_controls_live_predictions_and_stops_when_changed(self) -> None:
        window = MainWindow()
        try:
            panel = window.wheelchair_panel
            panel.mode_combo.setCurrentText("EEG Control")
            panel.service.connected = True
            panel.service.armed = True
            panel.service.set_output = Mock()
            panel.service.stop_motion = Mock()
            panel.eeg_streaming = True
            panel.prediction_active = True
            panel.model_has_stop_class = True
            panel._sync_decision_enabled()
            for votes in (1, 2, 3, 4, 5):
                with self.subTest(votes=votes):
                    panel.voting_combo.setCurrentIndex(panel.voting_combo.findData(votes))
                    self.assertEqual(panel.decision_controller.votes_required, votes)
                    self.assertEqual(panel.decision_controller.last_decision, "STOP")
                    self.assertFalse(panel.decision_controller.movement_timer.isActive())
                    self.assertEqual(len(panel.decision_controller.history), 0)
                    panel.service.set_output.reset_mock()
                    panel.handle_prediction({"label": "Left", "confidence": 0.74})
                    panel.service.set_output.assert_not_called()
                    for _ in range(votes - 1):
                        panel.handle_prediction({"label": "Left", "confidence": 0.95})
                        panel.service.set_output.assert_not_called()
                    panel.handle_prediction({"label": "Left", "confidence": 0.95})
                    panel.service.set_output.assert_called_once_with("LEFT")
                    self.assertIn("moving for 3 seconds", panel.decision_status_label.text())
            panel.voting_combo.setCurrentIndex(0)
            self.assertEqual(panel.decision_controller.last_decision, "STOP")
            panel.service.stop_motion.assert_called()
        finally:
            self._close_window(window)

    def test_rest_forward_rest_sends_commands_without_a_rest_cooldown(self) -> None:
        window = MainWindow()
        try:
            panel = window.wheelchair_panel
            panel.mode_combo.setCurrentText("EEG Control")
            panel.service.serial_port = Mock(is_open=True)
            panel.service.serial_port.write.side_effect = lambda payload: len(payload)
            panel.service.connected = True
            panel.service.armed = True
            panel.eeg_streaming = True
            panel.prediction_active = True
            panel.model_has_stop_class = True
            serial_port = panel.service.serial_port
            with patch("core.wheelchair_decision.time.monotonic", return_value=100.0) as clock:
                panel._sync_decision_enabled()
                for index, (label, confidence, command) in enumerate((
                    ("Rest", 0.01, b"x"),
                    ("Rest", 0.01, b"x"),
                    ("Forward", 0.95, b"w"),
                    ("Rest", 0.01, b"w"),
                    ("Forward", 0.95, b"w"),
                )):
                    clock.return_value = 100.0 + index * 0.01
                    panel.handle_prediction({"label": label, "confidence": confidence})
                    serial_port.write.assert_called_with(command)
                    if index < 2:
                        self.assertFalse(panel.service.heartbeat_timer.isActive())
                        self.assertFalse(panel.decision_controller.movement_timer.isActive())
                        self.assertEqual(panel.decision_controller.movement_deadline_at, 0.0)
                    else:
                        self.assertTrue(panel.service.heartbeat_timer.isActive())
                        self.assertAlmostEqual(
                            panel.decision_controller.movement_deadline_at, 103.02,
                        )
                self.assertEqual(
                    [call.args[0] for call in serial_port.write.call_args_list],
                    [b"x", b"x", b"w"],
                )
        finally:
            self._close_window(window)

    def test_emergency_stop_and_stream_loss_interrupt_latched_movement(self) -> None:
        for condition in ("emergency_stop", "stream_loss", "prediction_stopped"):
            with self.subTest(condition=condition):
                window = MainWindow()
                try:
                    panel = window.wheelchair_panel
                    panel.mode_combo.setCurrentText("EEG Control")
                    panel.service.serial_port = Mock(is_open=True)
                    panel.service.serial_port.write.side_effect = lambda payload: len(payload)
                    panel.service.connected = True
                    panel.service.armed = True
                    panel.eeg_streaming = True
                    panel.prediction_active = True
                    panel.model_has_stop_class = True
                    panel._sync_decision_enabled()
                    panel.handle_prediction({"label": "Forward", "confidence": 0.99})
                    self.assertEqual(panel.service.current_direction, "FORWARD")
                    if condition == "emergency_stop":
                        panel.emergency_stop()
                        self.assertFalse(panel.service.armed)
                    elif condition == "stream_loss":
                        panel.set_eeg_streaming(False)
                    else:
                        panel.set_prediction_active(False)
                    panel.service.serial_port.write.assert_called_with(b"x")
                    self.assertEqual(panel.service.current_direction, "STOP")
                    self.assertFalse(panel.service.heartbeat_timer.isActive())
                    self.assertFalse(panel.decision_controller.movement_timer.isActive())
                finally:
                    self._close_window(window)

    def test_manual_focus_loss_stops_a_held_direction(self) -> None:
        window = MainWindow()
        try:
            panel = window.wheelchair_panel
            panel.service.connected = True
            panel.service.armed = True
            panel.service.set_output = Mock()
            panel.service.stop_motion = Mock()
            panel._manual_press("FORWARD")
            panel.eventFilter(window, QtCore.QEvent(QtCore.QEvent.WindowDeactivate))
            panel.service.stop_motion.assert_called()
            self.assertEqual(panel._pressed_direction, "")
        finally:
            self._close_window(window)

    def test_training_displays_rest_trim_counts_and_keeps_collection_settings(self) -> None:
        window = MainWindow()
        try:
            collection = window._get_ml_window("data_collection")
            training = window._get_ml_window("train_model")
            before = collection._collection_config_payload()
            self.assertTrue(collection.record_rest_check.isChecked())
            training._on_load_model_clicked = Mock()
            summary = "Rest undersampling: removed 60 training windows.\nTraining windows per class (before -> after): Rest: 80 -> 20"
            training._on_train_success({
                "model_path": "synthetic-test.joblib", "run_dir": "synthetic-run",
                "accuracy": 0.8, "weighted_f1": 0.8, "cv_mean_accuracy": 0.75, "cv_std_accuracy": 0.1,
                "train_windows": 80, "test_windows": 40, "classes": ["Forward", "Left", "Rest", "Right"],
                "feature_dim": 24, "train_seconds": 0.1, "report": "Synthetic test report",
                "rest_undersampling_summary": summary,
            })
            self.assertIn("Rest: 80 -> 20", training.metrics_box.toPlainText())
            self.assertEqual(collection._collection_config_payload(), before)
            training._on_load_model_clicked.assert_called_once()
        finally:
            self._close_window(window)


if __name__ == "__main__":
    unittest.main()
