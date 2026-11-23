from __future__ import annotations

import json
import csv
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
CODE_DIR = Path(__file__).resolve().parents[1] / "code"
WORKSPACE = CODE_DIR.parents[1]
sys.path.insert(0, str(CODE_DIR))

import numpy as np
from PyQt5 import QtCore, QtGui, QtTest, QtWidgets
from app_theme import apply_dark_theme
from ui.main_window import MainWindow


class LightweightAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        # Qt's Windows offscreen plugin starts with an empty font database.
        # Load the actual app font so clipping checks use real glyph metrics.
        font_path = Path("C:/Windows/Fonts/segoeui.ttf")
        if not QtGui.QFontDatabase().families() and font_path.is_file():
            QtGui.QFontDatabase.addApplicationFont(str(font_path))
        apply_dark_theme(cls.app, font_size=14)

    def dispose(self, window):
        window.close()
        window.deleteLater()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
        self.app.processEvents()

    def test_recording_window_handles_resize_during_native_window_setup(self):
        from ui.eeg_ml_window import RecordingSessionWindow

        def dispatch_early_resize(window):
            window.resizeEvent(QtGui.QResizeEvent(window.size(), QtCore.QSize()))

        with patch("ui.eeg_ml_window.apply_dark_title_bar", side_effect=dispatch_early_resize) as theme:
            window = RecordingSessionWindow()
        try:
            theme.assert_called_once_with(window)
            window.show()
            window.resize(1100, 700)
            self.app.processEvents()
            self.assertIsNone(window._activity_pixmap)
            self.assertFalse(window.image_label.isVisible())
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "cue.png"
                cue = QtGui.QPixmap(100, 50)
                cue.fill(QtCore.Qt.red)
                self.assertTrue(cue.save(str(path)))
                window.set_activity_image(str(path))
                window.resize(1200, 760)
                self.app.processEvents()
                self.assertFalse(window.image_label.pixmap().isNull())
                window.set_activity_image(None)
                window.resize(1100, 700)
                self.app.processEvents()
                self.assertFalse(window.image_label.isVisible())
        finally:
            self.dispose(window)

    def test_main_controls_fit_laptop_sizes_without_scrolling_or_plot_widgets(self):
        window = MainWindow()
        try:
            window.show()
            for width, height in ((1280, 720), (1366, 700), (1100, 660)):
                with self.subTest(size=(width, height)):
                    window.resize(width, height)
                    self.app.processEvents()
                    self.assertLessEqual(window.width(), width)
                    self.assertLessEqual(window.height(), height)
                    self.assertLessEqual(window.minimumSizeHint().height(), height)
                    self.assertFalse(window.findChildren(QtWidgets.QScrollArea))
                    self.assertFalse(window.findChildren(QtWidgets.QGraphicsView))
                    panel = window.wheelchair_panel
                    controls = [window.connect_button, window.start_button, window.ml_data_button,
                                window.ml_train_button, window.ml_load_button, window.ml_realtime_button,
                                panel.connect_button, panel.arm_button, panel.stop_button,
                                panel.minimum_confidence_edit, panel.save_confidence_button,
                                panel.movement_duration_edit, panel.save_duration_button,
                                panel.voting_combo, panel.open_prediction_button,
                                *panel.direction_buttons.values()]
                    for control in controls:
                        location = control.mapTo(window.centralWidget(), QtCore.QPoint(0, 0))
                        rect = QtCore.QRect(location, control.size())
                        self.assertTrue(control.isVisible(), control.text() if hasattr(control, "text") else type(control).__name__)
                        self.assertTrue(window.centralWidget().rect().contains(rect), msg=f"{type(control).__name__} clipped: {rect}")
                        parent = control.parentWidget()
                        self.assertTrue(parent.rect().contains(QtCore.QRect(control.pos(), control.size())),
                                        msg=f"{type(control).__name__} clipped inside {type(parent).__name__}")
            for removed in ("plot_widget", "ring_buffer", "display_buffer", "display_pipeline",
                            "impedance_button", "filters_toggle_button", "graph_stream_checkbox", "auto_scale_checkbox"):
                self.assertFalse(hasattr(window, removed), removed)
            self.assertFalse(window.status_timer.isActive())
            self.assertEqual(window._ml_windows(), [])
            self.assertNotIn("UI FPS", window.metrics_label.text())
        finally:
            self.dispose(window)

    def test_eeg_chunks_reach_collection_and_inference_without_a_display_buffer(self):
        window = MainWindow()
        worker = Mock()
        worker.isRunning.return_value = False
        try:
            window.board_service.connect = Mock(side_effect=lambda port: setattr(window.board_service, "connected", True))
            window.board_service.get_eeg_channels = Mock(return_value=list(range(16)))
            window.board_service.get_sample_rate = Mock(return_value=125)
            window.board_service.get_connection_name = Mock(return_value="TEST EEG")
            window.board_service.start_stream = Mock(side_effect=lambda: setattr(window.board_service, "streaming", True))
            window.board_service.stop_stream = Mock(side_effect=lambda: setattr(window.board_service, "streaming", False))
            collection = Mock(prediction_enabled=False)
            realtime = Mock(prediction_enabled=True)
            window._ml_windows_by_mode = {"data_collection": collection, "realtime_prediction": realtime}
            window._create_worker = Mock(return_value=worker)
            window._connect_to_port("TEST")
            window.handle_start()
            self.assertTrue(window.status_timer.isActive())
            self.assertTrue(window.wheelchair_panel.eeg_streaming)
            chunk = np.ones((16, 10))
            timestamps = np.arange(10) / 125
            indices = np.arange(10)
            window.handle_data_ready({"eeg": chunk, "timestamps": timestamps, "sample_index": indices})
            for workflow in (collection, realtime):
                workflow.handle_incoming_chunk.assert_called_once_with(chunk, timestamps=timestamps, sample_index=indices)
            self.assertEqual(window._samples_received, 10)
            window.wheelchair_panel.mode_combo.setCurrentText("EEG Control")
            service = window.wheelchair_panel.service
            service.connected = True
            service.armed = True
            service.stop_motion = Mock()
            service.set_output = Mock()
            window._handle_prediction_activity_changed(True)
            window.wheelchair_panel.handle_prediction({"label": "Forward", "confidence": 0.99})
            service.set_output.assert_called_with("FORWARD")
            window.handle_worker_error("test acquisition failure")
            service.stop_motion.assert_called()
            self.assertFalse(window.wheelchair_panel.decision_controller.enabled)
            self.assertFalse(window.status_timer.isActive())
            self.assertIn("test acquisition failure", window.status_label.text())
            window.handle_data_ready({"eeg": chunk})
            self.assertEqual(window._samples_received, 10)
        finally:
            window.board_service.connected = False
            self.dispose(window)

    def test_latest_loaded_model_is_used_and_reopening_prediction_preserves_its_state(self):
        window = MainWindow()
        try:
            older = window._get_ml_window("load_model")
            newer = window._get_ml_window("train_model")
            def bundle(name):
                return SimpleNamespace(class_names=["Forward", "Left", "Rest", "Right"],
                                       sample_rate=125, channel_count=16, window_samples=250, stride_samples=62,
                                       name=name)
            older.model_bundle = bundle("older")
            older.loaded_model_path = "older.joblib"
            newer.model_bundle = bundle("newer")
            newer.loaded_model_path = "newer.joblib"
            window._handle_ml_model_loaded(older)
            window._handle_ml_model_loaded(newer)
            realtime = window._get_ml_window("realtime_prediction")
            self.assertIs(realtime.model_bundle, newer.model_bundle)
            self.assertEqual(realtime.loaded_model_path, "newer.joblib")
            self.assertFalse(hasattr(realtime, "realtime_bar_plot"))
            self.assertFalse(realtime.findChildren(QtWidgets.QGraphicsView))
            realtime._samples_since_submit = 30
            window._open_realtime_prediction_window()
            self.assertEqual(realtime._samples_since_submit, 30)
            realtime._on_prediction_ready({"label": "Left", "confidence": 0.92, "latency_ms": 2.4})
            self.assertEqual(window.wheelchair_panel.prediction_label.text(), "Left")
            self.assertIn("92.0%", realtime.realtime_conf_label.text())
        finally:
            self.dispose(window)

    def test_real_csv_replay_stream_reaches_workflow_and_stops_cleanly(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            source = Path(directory) / "replay.csv"
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["board_timestamp", "sample_index"] + [f"ch{index + 1}" for index in range(16)])
                for index in range(100):
                    writer.writerow([index / 125, index] + [float(index + channel) for channel in range(16)])
            window = MainWindow()
            workflow = Mock()
            window._ml_windows_by_mode["data_collection"] = workflow
            try:
                window._connect_to_port(f"csv://{source}")
                window.handle_start()
                QtTest.QTest.qWait(250)
                self.assertGreater(window._samples_received, 0)
                payload = workflow.handle_incoming_chunk.call_args
                self.assertIsNotNone(payload)
                self.assertEqual(payload.args[0].shape[0], 16)
                self.assertEqual(payload.args[0].shape[1], len(payload.kwargs["timestamps"]))
                self.assertEqual(payload.args[0].shape[1], len(payload.kwargs["sample_index"]))
                window.handle_stop()
                self.assertIsNone(window.stream_worker)
                self.assertFalse(window.board_service.streaming)
                self.assertFalse(window.status_timer.isActive())
            finally:
                self.dispose(window)

    def test_background_inference_emits_latest_window_without_audit_metadata(self):
        from core.eeg_inference_worker import EEGInferenceWorker
        worker = EEGInferenceWorker()
        worker.set_model_bundle(SimpleNamespace())
        spy = QtTest.QSignalSpy(worker.prediction_ready)
        worker.submit_window(np.zeros((16, 32)))
        worker.submit_window(np.ones((16, 32)))
        result = {"label": "Forward", "confidence": 0.95, "latency_ms": 1.0}
        try:
            with patch("core.eeg_inference_worker.predict_from_window", return_value=result) as predict:
                worker.start()
                self.assertTrue(spy.wait(2000))
                worker.stop()
                np.testing.assert_array_equal(predict.call_args.args[1], np.ones((16, 32)))
                self.assertEqual(spy[0][0], result)
                self.assertNotIn("prediction_id", spy[0][0])
        finally:
            worker.stop()

    def test_fresh_main_and_helper_launches_create_no_logs_or_plotting_imports(self):
        log_dir = WORKSPACE / "Logs"
        def snapshot():
            return {str(path): (path.stat().st_size, path.stat().st_mtime_ns)
                    for path in log_dir.rglob("*") if path.is_file()} if log_dir.exists() else {}
        before = snapshot()
        script = textwrap.dedent("""
            import json, sys, threading
            from pathlib import Path
            entrypoint = Path(sys.argv[1])
            sys.path.insert(0, str(entrypoint.parent))
            from PyQt5 import QtCore, QtGui, QtWidgets
            original = QtWidgets.QApplication
            class TimedApplication(original):
                def __init__(self, argv):
                    super().__init__(argv)
                    from pathlib import Path
                    font_path = Path('C:/Windows/Fonts/segoeui.ttf')
                    if not QtGui.QFontDatabase().families() and font_path.is_file():
                        QtGui.QFontDatabase.addApplicationFont(str(font_path))
                    QtCore.QTimer.singleShot(150, self.closeAllWindows)
            QtWidgets.QApplication = TimedApplication
            module = __import__(entrypoint.stem)
            code = module.main()
            print(json.dumps({"exit_code": code,
                "plot_imported": any(name.startswith('pyqtgraph') for name in sys.modules),
                "ml_imported": 'core.eeg_ml' in sys.modules,
                "audit_imported": 'core.session_logging' in sys.modules,
                "threads": [thread.name for thread in threading.enumerate()]}))
        """)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            for entrypoint in (CODE_DIR.parent / "run_control.py", CODE_DIR / "main.py", CODE_DIR / "simulator_main.py"):
                with self.subTest(entrypoint=entrypoint.name):
                    run = subprocess.run([sys.executable, "-c", script, str(entrypoint)],
                                         cwd=directory, capture_output=True, text=True, timeout=30)
                    self.assertEqual(run.returncode, 0, run.stderr)
                    result = json.loads(run.stdout.strip().splitlines()[-1])
                    self.assertEqual(result["exit_code"], 0)
                    self.assertFalse(result["plot_imported"])
                    self.assertFalse(result["ml_imported"])
                    self.assertFalse(result["audit_imported"])
                    self.assertEqual(result["threads"], ["MainThread"])
                    self.assertFalse(list(Path(directory).rglob("*.log")))
                    self.assertFalse((Path(directory) / "Logs").exists())
        self.assertEqual(snapshot(), before)


if __name__ == "__main__":
    unittest.main()
