from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from PyQt5 import QtCore, QtGui, QtWidgets

from app_theme import apply_dark_title_bar, themed_button_style, themed_label_style
from config import APP_TITLE, DEFAULT_CHANNEL_COUNT, DEFAULT_POLL_INTERVAL_MS, DEFAULT_SAMPLE_RATE
from core.board_service import BoardService
from core.serial_ports import list_serial_ports
from core.stream_worker import StreamWorker
from ui.wheelchair_panel import WheelchairControlPanel

if TYPE_CHECKING:
    from ui.eeg_ml_window import EegMLWindow


class MainWindow(QtWidgets.QMainWindow):
    """Single-page EEG acquisition and wheelchair controls, without plotting."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        super().__init__()
        self.logger = logger or logging.getLogger(__name__)
        self.board_service = BoardService(logger=self.logger)
        self.stream_worker: StreamWorker | None = None
        self.channel_count = DEFAULT_CHANNEL_COUNT
        self.sample_rate = DEFAULT_SAMPLE_RATE
        self.available_ports: list[str] = []
        self.selected_port_text = ""
        self.last_error_message = None
        self._ml_windows_by_mode: dict[str, EegMLWindow] = {}
        self._loaded_model_source = None
        self._samples_received = 0
        self._rate_samples = 0
        self._rate_started = time.monotonic()
        self._closing = False
        self.setWindowTitle(APP_TITLE)
        self.setMinimumSize(1100, 660)
        self.resize(1280, 700)
        self._build_ui()
        self.status_timer = QtCore.QTimer(self)
        self.status_timer.setInterval(1000)
        self.status_timer.timeout.connect(self._update_data_rate)
        self.refresh_ports(update_status=False)
        self._set_button_state(False, False)
        self._set_status("Ready — connect an EEG board or replay source.")
        apply_dark_title_bar(self)
        screen = self.screen() or QtWidgets.QApplication.primaryScreen()
        if screen is not None:
            frame = self.frameGeometry()
            frame.moveCenter(screen.availableGeometry().center())
            self.move(frame.topLeft())

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(12, 8, 12, 8)
        root.setSpacing(8)
        connection = QtWidgets.QFrame()
        connection.setProperty("card", True)
        row = QtWidgets.QHBoxLayout(connection)
        row.setContentsMargins(12, 8, 12, 8)
        row.setSpacing(10)
        title = QtWidgets.QLabel("EEG CONNECTION")
        title.setStyleSheet("font-weight: 800;")
        self.port_combo = QtWidgets.QComboBox()
        self.port_combo.setEditable(True)
        self.port_combo.setMinimumContentsLength(14)
        self.port_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.port_combo.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.refresh_button = QtWidgets.QPushButton("Refresh")
        self.connect_button = QtWidgets.QPushButton("Connect")
        self.start_button = QtWidgets.QPushButton("Start")
        self.board_detail_label = QtWidgets.QLabel("Disconnected")
        self.board_detail_label.setStyleSheet(themed_label_style("muted"))
        self.board_detail_label.setMinimumWidth(150)
        self.board_detail_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        row.addWidget(title)
        row.addWidget(self.port_combo, 2)
        row.addWidget(self.refresh_button)
        row.addWidget(self.connect_button)
        row.addWidget(self.start_button)
        row.addWidget(self.board_detail_label, 1)
        root.addWidget(connection)
        workflow = QtWidgets.QFrame()
        workflow.setProperty("card", True)
        self.workflow_layout = QtWidgets.QHBoxLayout(workflow)
        self.workflow_layout.setContentsMargins(12, 8, 12, 8)
        self.workflow_layout.setSpacing(10)
        self.workflow_layout.addWidget(QtWidgets.QLabel("EEG WORKFLOW"))
        self.ml_data_button = QtWidgets.QPushButton("Data Collection")
        self.ml_train_button = QtWidgets.QPushButton("Train Model")
        self.ml_load_button = QtWidgets.QPushButton("Load Model")
        self.ml_realtime_button = QtWidgets.QPushButton("Realtime Prediction")
        for button in (self.ml_data_button, self.ml_train_button, self.ml_load_button, self.ml_realtime_button):
            self.workflow_layout.addWidget(button, 1)
        root.addWidget(workflow)
        self.wheelchair_panel = WheelchairControlPanel(logger=self.logger, parent=self)
        root.addWidget(self.wheelchair_panel, 1)
        footer = QtWidgets.QFrame()
        footer.setProperty("card", True)
        status = QtWidgets.QHBoxLayout(footer)
        status.setContentsMargins(12, 6, 12, 6)
        status.setSpacing(16)
        self.status_label = QtWidgets.QLabel()
        self.status_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        self.prediction_status_label = QtWidgets.QLabel("Prediction: off")
        self.metrics_label = QtWidgets.QLabel("Samples: 0 | Input: 0 Hz")
        self.metrics_label.setStyleSheet(themed_label_style("muted"))
        status.addWidget(self.status_label, 1)
        status.addWidget(self.prediction_status_label)
        status.addWidget(self.metrics_label)
        root.addWidget(footer)
        for button in (self.connect_button, self.start_button, self.ml_data_button,
                       self.ml_train_button, self.ml_load_button, self.ml_realtime_button):
            button.setStyleSheet(themed_button_style("accent"))
        self.refresh_button.setStyleSheet(themed_button_style("muted"))
        self.refresh_button.clicked.connect(self.refresh_ports)
        self.connect_button.clicked.connect(self.handle_connect_toggle)
        self.start_button.clicked.connect(self.handle_start_stop_toggle)
        self.ml_data_button.clicked.connect(self._open_data_collection_window)
        self.ml_train_button.clicked.connect(self._open_train_model_window)
        self.ml_load_button.clicked.connect(self._open_load_model_window)
        self.ml_realtime_button.clicked.connect(self._open_realtime_prediction_window)
        self.wheelchair_panel.request_realtime_window.connect(self._open_realtime_prediction_window)

    def _get_ml_window(self, mode: str) -> EegMLWindow:
        window = self._ml_windows_by_mode.get(mode)
        if window is not None:
            return window
        # Keep sklearn, training widgets and inference buffers out of startup.
        from ui.eeg_ml_window import EegMLWindow
        window = EegMLWindow(logger=self.logger, parent=self, mode=mode)
        window.set_stream_context(self.sample_rate, self.channel_count, self.board_service.connected)
        window.set_streaming(self.board_service.streaming)
        window.recording_activity_changed.connect(self._handle_recording_activity_changed)
        window.prediction_activity_changed.connect(self._handle_prediction_activity_changed)
        window.model_loaded.connect(self._handle_ml_model_loaded)
        if mode == "realtime_prediction":
            window.prediction_ready.connect(self.wheelchair_panel.handle_prediction)
            if self._loaded_model_source is not None:
                window.adopt_model_from(self._loaded_model_source)
        self._ml_windows_by_mode[mode] = window
        return window

    def _ml_windows(self) -> list[EegMLWindow]:
        return list(self._ml_windows_by_mode.values())

    def _handle_ml_model_loaded(self, source_window) -> None:
        self._loaded_model_source = source_window
        self.wheelchair_panel.set_loaded_model(source_window.model_bundle, source_window.loaded_model_path)
        realtime = self._ml_windows_by_mode.get("realtime_prediction")
        if realtime is not None and source_window is not realtime:
            realtime.adopt_model_from(source_window)

    def _open_ml_window(self, window) -> None:
        frame = window.frameGeometry()
        frame.moveCenter(self.frameGeometry().center())
        window.move(frame.topLeft())
        window.show()
        window.raise_()
        window.activateWindow()

    def _open_data_collection_window(self) -> None:
        self._open_ml_window(self._get_ml_window("data_collection"))

    def _open_train_model_window(self) -> None:
        self._open_ml_window(self._get_ml_window("train_model"))

    def _open_load_model_window(self) -> None:
        self._open_ml_window(self._get_ml_window("load_model"))

    def _open_realtime_prediction_window(self) -> None:
        self._open_ml_window(self._get_ml_window("realtime_prediction"))

    def _handle_recording_activity_changed(self, active: bool) -> None:
        self._set_status("EEG recording in progress." if active else
                         ("EEG streaming." if self.board_service.streaming else "Ready."))

    def _handle_prediction_activity_changed(self, _active: bool) -> None:
        realtime = self._ml_windows_by_mode.get("realtime_prediction")
        active = bool(realtime is not None and realtime.prediction_enabled and self.board_service.streaming)
        self.wheelchair_panel.set_prediction_active(active)
        self.prediction_status_label.setText("Prediction: live" if active else "Prediction: off")

    def refresh_ports(self, update_status: bool = True) -> None:
        current = self.port_combo.currentText().strip()
        try:
            self.available_ports = list_serial_ports()
        except Exception as exc:
            self._show_error(str(exc))
            return
        self.port_combo.clear()
        self.port_combo.addItems(self.available_ports)
        if current:
            self.port_combo.setCurrentText(current)
        if update_status and not self.board_service.connected:
            self._set_status(f"Ready — {len(self.available_ports)} EEG source(s) available.")

    def handle_connect_toggle(self) -> None:
        if self.board_service.connected:
            self.handle_disconnect()
        else:
            self.handle_connect()

    def handle_connect(self) -> None:
        try:
            self._connect_to_port(self.port_combo.currentText().strip())
        except Exception as exc:
            self.logger.exception("EEG connection failed.")
            self._show_error(str(exc))
            self._set_button_state(self.board_service.connected, False)

    def _connect_to_port(self, port: str) -> None:
        self.board_service.connect(port)
        self.channel_count = len(self.board_service.get_eeg_channels())
        self.sample_rate = self.board_service.get_sample_rate()
        self.selected_port_text = port
        name = self.board_service.get_connection_name()
        self.board_detail_label.setText(f"{self.channel_count} channels · {self.sample_rate} Hz")
        self.board_detail_label.setToolTip(name)
        for window in self._ml_windows():
            window.set_stream_context(self.sample_rate, self.channel_count, True)
            window.set_streaming(False)
        self._set_button_state(True, False)
        self._set_status(f"EEG connected — {name}")

    def handle_start_stop_toggle(self) -> None:
        if self.board_service.streaming:
            self.handle_stop()
        else:
            self.handle_start()

    def _create_worker(self) -> StreamWorker:
        worker = StreamWorker(self.board_service, DEFAULT_POLL_INTERVAL_MS,
                              max(1, self.sample_rate // 4), logger=self.logger)
        worker.data_ready.connect(self.handle_data_ready)
        worker.error.connect(self.handle_worker_error)
        return worker

    def handle_start(self) -> None:
        if self.board_service.streaming:
            return
        try:
            self.board_service.start_stream()
            self._samples_received = 0
            self._rate_samples = 0
            self._rate_started = time.monotonic()
            for window in self._ml_windows():
                window.set_stream_context(self.sample_rate, self.channel_count, True)
                window.set_streaming(True)
            self.wheelchair_panel.set_eeg_streaming(True)
            self.stream_worker = self._create_worker()
            self.stream_worker.start()
            self.status_timer.start()
            self._set_button_state(True, True)
            self._set_status("EEG streaming — ready for collection or live prediction.")
        except Exception as exc:
            self.logger.exception("EEG stream start failed.")
            self.handle_stop()
            self._show_error(str(exc))

    def handle_stop(self) -> None:
        # Revoke movement permission before waiting for acquisition cleanup.
        self.wheelchair_panel.set_eeg_streaming(False)
        self.status_timer.stop()
        for window in self._ml_windows():
            window.set_streaming(False)
        try:
            self._stop_worker()
            self.board_service.stop_stream()
        except Exception as exc:
            self.logger.exception("EEG stream stop failed.")
            self._show_error(str(exc))
        else:
            self._set_status("EEG connected — stream stopped." if self.board_service.connected else "Ready.")
        finally:
            self._set_button_state(self.board_service.connected, False)
            self._handle_prediction_activity_changed(False)
            self.metrics_label.setText(f"Samples: {self._samples_received:,} | Input: stopped")

    def handle_disconnect(self) -> None:
        self.handle_stop()
        try:
            self.board_service.disconnect()
            for window in self._ml_windows():
                window.set_stream_context(self.sample_rate, self.channel_count, False)
        except Exception as exc:
            self.logger.exception("EEG disconnect failed.")
            self._show_error(str(exc))
        else:
            self.board_detail_label.setText("Disconnected")
            self._set_status("EEG disconnected.")
        finally:
            self._set_button_state(self.board_service.connected, False)

    def _stop_worker(self) -> None:
        worker = self.stream_worker
        if worker is None:
            return
        worker.stop()
        if worker.isRunning() and not worker.wait(1500):
            raise RuntimeError("EEG acquisition is still stopping; please retry.")
        worker.deleteLater()
        self.stream_worker = None

    def handle_data_ready(self, payload) -> None:
        if self._closing or not self.board_service.streaming:
            return
        try:
            chunk = payload.get("eeg") if isinstance(payload, dict) else payload
            if chunk is None or chunk.size == 0:
                return
            timestamps = payload.get("timestamps") if isinstance(payload, dict) else None
            sample_index = payload.get("sample_index") if isinstance(payload, dict) else None
            self._samples_received += int(chunk.shape[1])
            self._rate_samples += int(chunk.shape[1])
            for window in self._ml_windows():
                window.handle_incoming_chunk(chunk, timestamps=timestamps, sample_index=sample_index)
        except Exception as exc:
            self.logger.exception("Incoming EEG processing failed.")
            self.handle_worker_error(str(exc))

    def handle_worker_error(self, message: str) -> None:
        self.handle_stop()
        self._show_error(message)

    def _update_data_rate(self) -> None:
        now = time.monotonic()
        rate = self._rate_samples / max(0.001, now - self._rate_started)
        self.metrics_label.setText(f"Samples: {self._samples_received:,} | Input: {rate:.0f} Hz")
        self._rate_samples = 0
        self._rate_started = now

    def _set_button_state(self, connected: bool, streaming: bool) -> None:
        self.connect_button.setText("Disconnect" if connected else "Connect")
        self.connect_button.setStyleSheet(themed_button_style("danger" if connected else "accent"))
        self.start_button.setEnabled(connected)
        self.start_button.setText("Stop" if streaming else "Start")
        self.start_button.setStyleSheet(themed_button_style("danger" if streaming else "accent"))
        self.port_combo.setEnabled(not connected)
        self.refresh_button.setEnabled(not connected)

    def _set_status(self, message: str) -> None:
        self.last_error_message = None
        self.status_label.setText(message)
        self.status_label.setToolTip(message)
        self.status_label.setStyleSheet(themed_label_style("muted"))

    def _show_error(self, message: str) -> None:
        self.last_error_message = str(message)
        self.status_label.setText(f"Error: {message}")
        self.status_label.setToolTip(str(message))
        self.status_label.setStyleSheet(themed_label_style("danger"))

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        self._closing = True
        self.status_timer.stop()
        self.wheelchair_panel.shutdown()
        try:
            self._stop_worker()
        except Exception as exc:
            self._closing = False
            self._show_error(str(exc))
            event.ignore()
            return
        for window in self._ml_windows():
            window.shutdown()
            window.hide()
        try:
            self.board_service.disconnect()
        except Exception:
            self.logger.exception("EEG shutdown failed.")
        event.accept()
