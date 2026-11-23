from __future__ import annotations

import logging

from PyQt5 import QtCore, QtGui, QtWidgets

from app_theme import THEME_COLORS, themed_button_style, themed_label_style
from core.wheelchair_outputs import format_output
from core.wheelchair_decision import (
    MAX_EEG_MOVEMENT_DURATION_MS,
    MIN_EEG_MOVEMENT_DURATION_MS,
    WheelchairDecisionController,
)
from core.wheelchair_protocol import (
    BLUETOOTH_BAUD_RATE,
    USB_BAUD_RATE,
)
from core.wheelchair_service import WheelchairService


class WheelchairControlPanel(QtWidgets.QWidget):
    """Manual and confidence-gated EEG control surface for the wheelchair."""

    request_realtime_window = QtCore.pyqtSignal()

    def __init__(self, logger: logging.Logger | None = None, parent=None) -> None:
        super().__init__(parent)
        self.logger = logger or logging.getLogger(__name__)
        self.service = WheelchairService(logger=self.logger, parent=self)
        self.decision_controller = WheelchairDecisionController(parent=self)
        self.eeg_streaming = False
        self.prediction_active = False
        self.loaded_model_name = "No model loaded"
        self.model_has_stop_class = False
        self._pressed_direction = ""
        self._event_filter_installed = False
        self._prediction_history: list[str] = []

        self._build_ui()
        self._connect_signals()
        self.refresh_ports()
        self._update_controls()

    def _build_ui(self) -> None:
        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.setColumnStretch(0, 3)
        layout.setColumnStretch(1, 3)
        layout.setColumnStretch(2, 4)
        self.setStyleSheet("QGroupBox { margin-top: 10px; padding: 4px; }")

        title = QtWidgets.QLabel("Wheelchair Control")
        title.setStyleSheet("font-size: 22px; font-weight: 800;")
        layout.addWidget(title, 0, 0, 1, 3)

        connection_group = QtWidgets.QGroupBox("Wheelchair Connection")
        connection_layout = QtWidgets.QGridLayout(connection_group)
        self.port_combo = QtWidgets.QComboBox()
        self.port_combo.setEditable(True)
        self.refresh_button = QtWidgets.QPushButton("Refresh")
        self.connect_button = QtWidgets.QPushButton("Connect")
        self.transport_combo = QtWidgets.QComboBox()
        self.transport_combo.addItem("Bluetooth (HC-06)", BLUETOOTH_BAUD_RATE)
        self.transport_combo.addItem("USB (Arduino)", USB_BAUD_RATE)
        connection_layout.addWidget(QtWidgets.QLabel("COM port"), 0, 0)
        connection_layout.addWidget(self.port_combo, 0, 1, 1, 2)
        connection_layout.addWidget(self.refresh_button, 1, 1)
        connection_layout.addWidget(self.connect_button, 1, 2)
        connection_layout.addWidget(QtWidgets.QLabel("Connection type"), 2, 0)
        connection_layout.addWidget(self.transport_combo, 2, 1, 1, 2)
        layout.addWidget(connection_group, 1, 0)

        safety_group = QtWidgets.QGroupBox("Mode and Safety")
        safety_layout = QtWidgets.QGridLayout(safety_group)
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Manual", "EEG Control"])
        self.arm_button = QtWidgets.QPushButton("ARM OUTPUT")
        self.arm_button.setCheckable(True)
        self.stop_button = QtWidgets.QPushButton("EMERGENCY STOP")
        self.stop_button.setMinimumHeight(48)
        self.stop_button.setStyleSheet(themed_button_style("danger"))
        safety_layout.addWidget(QtWidgets.QLabel("Control mode"), 0, 0)
        safety_layout.addWidget(self.mode_combo, 0, 1)
        safety_layout.addWidget(self.arm_button, 1, 0, 1, 2)
        safety_layout.addWidget(self.stop_button, 2, 0, 1, 2)
        layout.addWidget(safety_group, 2, 0)

        control_group = QtWidgets.QGroupBox("Manual Dead-Man Controls")
        control_layout = QtWidgets.QGridLayout(control_group)
        self.direction_buttons: dict[str, QtWidgets.QPushButton] = {}
        button_specs = [
            ("FORWARD", "FORWARD (W)", 0, 1),
            ("LEFT", "LEFT (A)", 1, 0),
            ("STOP", "STOP", 1, 1),
            ("RIGHT", "RIGHT (D)", 1, 2),
            ("BACKWARD", "BACKWARD (S)", 2, 1),
        ]
        for direction, text, row, col in button_specs:
            button = QtWidgets.QPushButton(text)
            button.setMinimumHeight(40)
            button.setToolTip(format_output(direction))
            if direction == "STOP":
                button.setStyleSheet(themed_button_style("danger"))
            self.direction_buttons[direction] = button
            control_layout.addWidget(button, row, col)
        help_label = QtWidgets.QLabel("Hold a direction button or key to move; release to stop.")
        help_label.setWordWrap(True)
        help_label.setStyleSheet(themed_label_style("muted"))
        control_layout.addWidget(help_label, 3, 0, 1, 3)
        fixed_hint = QtWidgets.QLabel("Fixed outputs: neutral 2.50 V.\nDirection targets: 0.55 / 4.45 V.")
        fixed_hint.setWordWrap(True)
        fixed_hint.setStyleSheet(themed_label_style("muted"))
        control_layout.addWidget(fixed_hint, 4, 0, 1, 3)
        layout.addWidget(control_group, 1, 1, 2, 1)

        prediction_group = QtWidgets.QGroupBox("Realtime EEG Decision")
        prediction_layout = QtWidgets.QGridLayout(prediction_group)
        self.open_prediction_button = QtWidgets.QPushButton("Open Realtime Prediction")
        self.model_label = QtWidgets.QLabel(self.loaded_model_name)
        self.model_label.setWordWrap(True)
        self.prediction_label = QtWidgets.QLabel("N/A")
        self.prediction_label.setStyleSheet("font-size: 24px; font-weight: 800;")
        self.confidence_label = QtWidgets.QLabel("0.0%")
        self.history_label = QtWidgets.QLabel("—")
        self.history_label.setWordWrap(True)
        self.minimum_confidence_edit = QtWidgets.QLineEdit(
            f"{self.decision_controller.minimum_confidence * 100.0:.1f}"
        )
        threshold_validator = QtGui.QDoubleValidator(25.0, 99.0, 1, self.minimum_confidence_edit)
        threshold_validator.setNotation(QtGui.QDoubleValidator.StandardNotation)
        threshold_validator.setLocale(QtCore.QLocale.c())
        self.minimum_confidence_edit.setValidator(threshold_validator)
        self.minimum_confidence_edit.setMinimumWidth(0)
        self.minimum_confidence_edit.setAccessibleName("Confidence threshold percentage")
        self.minimum_confidence_edit.setToolTip(
            "Enter 25 to 99% (up to one decimal place), then click Save to apply."
        )
        self.save_confidence_button = QtWidgets.QPushButton("Save")
        self.save_confidence_button.setAutoDefault(False)
        self.save_confidence_button.setAccessibleName("Save confidence threshold")
        self.save_confidence_button.setToolTip("Apply the entered confidence threshold.")
        threshold_row = QtWidgets.QHBoxLayout()
        threshold_row.setSpacing(6)
        threshold_row.addWidget(self.minimum_confidence_edit, 1)
        threshold_row.addWidget(QtWidgets.QLabel("%"))
        threshold_row.addWidget(self.save_confidence_button)
        self.voting_combo = QtWidgets.QComboBox()
        self.voting_combo.addItem("No voting", 1)
        for votes in (2, 3, 4, 5):
            self.voting_combo.addItem(f"{votes} votes", votes)
        self.voting_combo.setCurrentIndex(
            self.voting_combo.findData(self.decision_controller.votes_required)
        )
        self.voting_combo.setAccessibleName("Prediction voting")
        self.voting_combo.setToolTip(
            "No voting uses one prediction above the confidence threshold. "
            "Voting requires the selected number of matches among the latest five qualifying predictions."
        )
        self.movement_duration_edit = QtWidgets.QLineEdit(
            f"{self.decision_controller.movement_duration_ms / 1000:g}"
        )
        duration_validator = QtGui.QDoubleValidator(
            MIN_EEG_MOVEMENT_DURATION_MS / 1000,
            MAX_EEG_MOVEMENT_DURATION_MS / 1000,
            1,
            self.movement_duration_edit,
        )
        duration_validator.setNotation(QtGui.QDoubleValidator.StandardNotation)
        duration_validator.setLocale(QtCore.QLocale.c())
        self.movement_duration_edit.setValidator(duration_validator)
        self.movement_duration_edit.setMinimumWidth(0)
        self.movement_duration_edit.setAccessibleName("EEG movement duration in seconds")
        self.movement_duration_edit.setToolTip(
            "Enter 0.1 to 60 seconds (up to one decimal place), then click Save. "
            "Predictions, including Rest, cannot interrupt an active run. "
            "Emergency Stop and stream/connection failures still stop movement."
        )
        self.save_duration_button = QtWidgets.QPushButton("Save")
        self.save_duration_button.setAutoDefault(False)
        self.save_duration_button.setAccessibleName("Save movement duration")
        self.save_duration_button.setToolTip("Apply the entered EEG movement duration.")
        duration_row = QtWidgets.QHBoxLayout()
        duration_row.setSpacing(6)
        duration_row.addWidget(self.movement_duration_edit, 1)
        duration_row.addWidget(QtWidgets.QLabel("s"))
        duration_row.addWidget(self.save_duration_button)
        self.model_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        self.model_label.setMinimumHeight(36)
        self.model_label.setMaximumHeight(40)
        prediction_layout.addWidget(QtWidgets.QLabel("Model"), 0, 0)
        prediction_layout.addWidget(self.model_label, 0, 1)
        prediction_layout.addWidget(self.prediction_label, 1, 0)
        prediction_layout.addWidget(self.confidence_label, 1, 1)
        confidence_threshold_label = QtWidgets.QLabel("Confidence threshold")
        confidence_threshold_label.setBuddy(self.minimum_confidence_edit)
        prediction_layout.addWidget(confidence_threshold_label, 2, 0)
        prediction_layout.addLayout(threshold_row, 2, 1)
        prediction_layout.addWidget(QtWidgets.QLabel("Prediction voting"), 3, 0)
        prediction_layout.addWidget(self.voting_combo, 3, 1)
        movement_duration_label = QtWidgets.QLabel("Movement duration")
        movement_duration_label.setBuddy(self.movement_duration_edit)
        prediction_layout.addWidget(movement_duration_label, 4, 0)
        prediction_layout.addLayout(duration_row, 4, 1)
        prediction_layout.addWidget(QtWidgets.QLabel("Recent"), 5, 0)
        prediction_layout.addWidget(self.history_label, 5, 1)
        self.history_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        self.history_label.setWordWrap(False)
        prediction_layout.addWidget(self.open_prediction_button, 6, 0, 1, 2)
        layout.addWidget(prediction_group, 1, 2, 2, 1)

        output_group = QtWidgets.QGroupBox("Command Status")
        output_layout = QtWidgets.QGridLayout(output_group)
        output_layout.setColumnStretch(0, 1)
        output_layout.setColumnStretch(1, 2)
        self.output_label = QtWidgets.QLabel(
            f"STOP — {format_output('STOP')}"
        )
        self.output_label.setStyleSheet("font-size: 17px; font-weight: 800;")
        self.status_label = QtWidgets.QLabel("Disconnected and disarmed")
        self.status_label.setWordWrap(True)
        self.decision_status_label = QtWidgets.QLabel("EEG control is inactive.")
        self.decision_status_label.setWordWrap(True)
        self.decision_status_label.setStyleSheet(themed_label_style("muted"))
        self.output_label.setToolTip("Configured joystick targets; the controller provides no measured voltage feedback.")
        output_layout.addWidget(self.output_label, 0, 0)
        output_layout.addWidget(self.status_label, 1, 0)
        output_layout.addWidget(self.decision_status_label, 0, 1, 2, 1)
        layout.addWidget(output_group, 3, 0, 1, 3)
        layout.setRowStretch(1, 1)
        layout.setRowStretch(2, 1)

        for button in [
            self.refresh_button,
            self.connect_button,
            self.arm_button,
            self.open_prediction_button,
            self.save_confidence_button,
            self.save_duration_button,
            *[button for name, button in self.direction_buttons.items() if name != "STOP"],
        ]:
            button.setStyleSheet(themed_button_style("accent"))

    def _connect_signals(self) -> None:
        self.refresh_button.clicked.connect(self.refresh_ports)
        self.connect_button.clicked.connect(self._toggle_connection)
        self.arm_button.toggled.connect(self._toggle_arm)
        self.stop_button.clicked.connect(self.emergency_stop)
        self.direction_buttons["STOP"].pressed.connect(self._manual_stop)
        for direction in ("FORWARD", "BACKWARD", "LEFT", "RIGHT"):
            button = self.direction_buttons[direction]
            button.pressed.connect(lambda value=direction: self._manual_press(value))
            button.released.connect(self._manual_release)
        self.mode_combo.currentTextChanged.connect(self._mode_changed)
        self.open_prediction_button.clicked.connect(self.request_realtime_window)
        self.save_confidence_button.clicked.connect(self._save_confidence_threshold)
        self.save_duration_button.clicked.connect(self._save_movement_duration)
        self.voting_combo.currentIndexChanged.connect(self._decision_settings_changed)

        self.service.connection_changed.connect(self._connection_changed)
        self.service.armed_changed.connect(self._armed_changed)
        self.service.command_changed.connect(self._command_changed)
        self.service.error.connect(self._show_error)
        self.decision_controller.decision_ready.connect(self._handle_decision)
        self.decision_controller.status_changed.connect(self.decision_status_label.setText)

    def showEvent(self, event: QtGui.QShowEvent) -> None:  # noqa: N802
        super().showEvent(event)
        if not self._event_filter_installed and QtWidgets.QApplication.instance() is not None:
            QtWidgets.QApplication.instance().installEventFilter(self)
            self._event_filter_installed = True

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        if event.type() in (QtCore.QEvent.WindowDeactivate, QtCore.QEvent.ApplicationDeactivate):
            if self._pressed_direction:
                self._manual_release()
            return super().eventFilter(watched, event)
        if self.mode_combo.currentText() != "Manual" or not self.service.armed:
            return super().eventFilter(watched, event)
        key_map = {
            QtCore.Qt.Key_W: "FORWARD",
            QtCore.Qt.Key_S: "BACKWARD",
            QtCore.Qt.Key_A: "LEFT",
            QtCore.Qt.Key_D: "RIGHT",
        }
        # Handle release even if focus moved to an input while a key was held.
        if event.type() == QtCore.QEvent.KeyRelease and not event.isAutoRepeat():
            if event.key() in key_map and self._pressed_direction == key_map[event.key()]:
                self._manual_release()
                return True
        if not self.window().isActiveWindow():
            return super().eventFilter(watched, event)
        if event.type() == QtCore.QEvent.KeyPress and not event.isAutoRepeat():
            if event.key() in (QtCore.Qt.Key_X, QtCore.Qt.Key_Space):
                self._manual_stop()
                return True
        focus = QtWidgets.QApplication.focusWidget()
        if isinstance(focus, (QtWidgets.QLineEdit, QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox, QtWidgets.QComboBox)):
            return super().eventFilter(watched, event)
        if event.type() == QtCore.QEvent.KeyPress and not event.isAutoRepeat():
            if event.key() in key_map:
                self._manual_press(key_map[event.key()])
                return True
        return super().eventFilter(watched, event)

    def refresh_ports(self) -> None:
        current = self.port_combo.currentText().strip()
        ports = self.service.available_ports()
        self.port_combo.clear()
        self.port_combo.addItems(ports)
        if current:
            index = self.port_combo.findText(current)
            if index >= 0:
                self.port_combo.setCurrentIndex(index)
            else:
                self.port_combo.setEditText(current)

    def set_eeg_streaming(self, streaming: bool) -> None:
        self.eeg_streaming = bool(streaming)
        self._sync_decision_enabled()
        self._update_controls()

    def set_prediction_active(self, active: bool) -> None:
        self.prediction_active = bool(active)
        self._sync_decision_enabled()
        self._update_controls()

    def set_loaded_model(self, bundle, path: str = "") -> None:
        if bundle is None:
            self.loaded_model_name = "No model loaded"
            self.model_has_stop_class = False
        else:
            name = QtCore.QFileInfo(str(path)).fileName() or "Loaded model"
            classes = ", ".join(bundle.class_names)
            self.loaded_model_name = f"{name} — {classes}"
            normalized_classes = {
                str(label).strip().casefold().replace("-", "_") for label in bundle.class_names
            }
            self.model_has_stop_class = bool(
                normalized_classes.intersection(self.decision_controller.STOP_LABELS)
            )
        self.model_label.setText(self.loaded_model_name)
        self.model_label.setToolTip(self.loaded_model_name)
        self._update_controls()

    def handle_prediction(self, result: dict) -> None:
        label = str(result.get("label", "N/A"))
        confidence = float(result.get("confidence", 0.0) or 0.0)
        self.prediction_label.setText(label)
        self.confidence_label.setText(f"{confidence * 100.0:.1f}%")
        self._prediction_history.append(label)
        self._prediction_history = self._prediction_history[-5:]
        self.history_label.setText(" → ".join(self._prediction_history))
        self.history_label.setToolTip(" → ".join(self._prediction_history))
        self.decision_controller.process_prediction(result)

    def emergency_stop(self) -> None:
        self.decision_controller.set_enabled(False)
        try:
            self.service.emergency_stop()
        except Exception as exc:
            self._show_error(str(exc))
        self.status_label.setText("EMERGENCY STOP — output disarmed")
        self.status_label.setStyleSheet(themed_label_style("danger"))

    def shutdown(self) -> None:
        self.decision_controller.set_enabled(False)
        if self._event_filter_installed and QtWidgets.QApplication.instance() is not None:
            QtWidgets.QApplication.instance().removeEventFilter(self)
            self._event_filter_installed = False
        self.service.shutdown()

    def _toggle_connection(self) -> None:
        if self.service.connected:
            self.service.disconnect()
            return
        try:
            self.service.connect_port(self.port_combo.currentText(), self.transport_combo.currentData())
        except Exception as exc:
            self._show_error(str(exc))

    def _toggle_arm(self, checked: bool) -> None:
        try:
            if checked:
                if self.mode_combo.currentText() == "EEG Control" and not self.model_has_stop_class:
                    raise RuntimeError(
                        "EEG control requires a loaded model with a Rest, Stop, Idle, or Neutral class."
                    )
                self.service.arm()
            else:
                self.service.disarm()
        except Exception as exc:
            self.arm_button.blockSignals(True)
            self.arm_button.setChecked(False)
            self.arm_button.blockSignals(False)
            self._show_error(str(exc))
        self._sync_decision_enabled()
        self._update_controls()

    def _connection_changed(self, connected: bool, port: str) -> None:
        self.connect_button.setText("Disconnect" if connected else "Connect")
        self.connect_button.setStyleSheet(themed_button_style("danger" if connected else "accent"))
        self.port_combo.setEnabled(not connected)
        self.refresh_button.setEnabled(not connected)
        self.transport_combo.setEnabled(not connected)
        self._pressed_direction = ""
        if connected:
            self.status_label.setText(f"Connected to {port}; output disarmed")
            self.status_label.setStyleSheet(themed_label_style("success"))
        else:
            self.status_label.setText("Disconnected and disarmed")
            self.status_label.setStyleSheet(themed_label_style("muted"))
        self._sync_decision_enabled()
        self._update_controls()

    def _armed_changed(self, armed: bool) -> None:
        if not armed:
            self._pressed_direction = ""
        self.arm_button.blockSignals(True)
        self.arm_button.setChecked(bool(armed))
        self.arm_button.setText("DISARM OUTPUT" if armed else "ARM OUTPUT")
        self.arm_button.setStyleSheet(themed_button_style("danger" if armed else "accent"))
        self.arm_button.blockSignals(False)
        if armed:
            self.status_label.setText(f"ARMED — {self.mode_combo.currentText()} mode")
            self.status_label.setStyleSheet(themed_label_style("warning"))
        elif self.service.connected:
            self.status_label.setText(f"Connected to {self.service.port_name}; output disarmed")
            self.status_label.setStyleSheet(themed_label_style("success"))
        self._sync_decision_enabled()
        self._update_controls()

    def _command_changed(self, direction: str, x_value: int, y_value: int) -> None:
        self.output_label.setText(
            f"{direction} — X {x_value / 1000:.2f} V | Y {y_value / 1000:.2f} V"
        )
        if direction == "STOP":
            self.output_label.setStyleSheet("font-size: 17px; font-weight: 800;")
        else:
            self.output_label.setStyleSheet(
                f"font-size: 17px; font-weight: 800; color: {THEME_COLORS['warning']};"
            )

    def _manual_press(self, direction: str) -> None:
        if self.mode_combo.currentText() != "Manual" or not self.service.armed:
            return
        self._pressed_direction = direction
        try:
            self.service.set_output(direction)
        except Exception as exc:
            self._show_error(str(exc))

    def _manual_release(self) -> None:
        self._pressed_direction = ""
        self._manual_stop()

    def _manual_stop(self) -> None:
        self._pressed_direction = ""
        try:
            if self.service.connected:
                self.service.stop_motion()
        except Exception as exc:
            self._show_error(str(exc))

    def _mode_changed(self, _mode: str) -> None:
        if self.service.armed:
            try:
                self.service.disarm()
            except Exception as exc:
                self._show_error(str(exc))
        self._prediction_history.clear()
        self.history_label.setText("—")
        self._sync_decision_enabled()
        self._update_controls()

    def _save_confidence_threshold(self) -> None:
        if not self.minimum_confidence_edit.hasAcceptableInput():
            self.decision_status_label.setText(
                "Enter a confidence threshold from 25 to 99% (up to one decimal place), then click Save."
            )
            self.minimum_confidence_edit.setFocus()
            self.minimum_confidence_edit.selectAll()
            return
        percent = float(self.minimum_confidence_edit.text())
        self.decision_controller.configure(percent / 100.0, self.voting_combo.currentData())
        self.minimum_confidence_edit.setText(f"{percent:.1f}")
        self.decision_status_label.setText(f"Confidence threshold saved: {percent:.1f}%.")

    def _save_movement_duration(self) -> None:
        if not self.movement_duration_edit.hasAcceptableInput():
            self.decision_status_label.setText(
                "Enter a movement duration from 0.1 to 60 seconds (up to one decimal place), then click Save."
            )
            self.movement_duration_edit.setFocus()
            self.movement_duration_edit.selectAll()
            return
        seconds = float(self.movement_duration_edit.text())
        self.decision_controller.set_movement_duration(int(round(seconds * 1000)))
        self.movement_duration_edit.setText(f"{seconds:g}")
        self.decision_status_label.setText(f"Movement duration saved: {seconds:g} seconds.")

    def _decision_settings_changed(self, _value) -> None:
        try:
            self.decision_controller.configure(
                self.decision_controller.minimum_confidence,
                self.voting_combo.currentData(),
            )
        except ValueError as exc:
            self._show_error(str(exc))

    def _sync_decision_enabled(self) -> None:
        should_enable = (
            self.mode_combo.currentText() == "EEG Control"
            and self.service.connected
            and self.service.armed
            and self.eeg_streaming
            and self.prediction_active
        )
        self.decision_controller.set_enabled(should_enable)

    def _handle_decision(self, payload: dict) -> None:
        direction = str(payload.get("direction", "STOP")).upper()
        try:
            if direction == "STOP":
                if self.service.connected:
                    self.service.stop_motion()
                return
            if not self.decision_controller.enabled or not self.service.armed:
                return
            self.service.set_output(direction)
        except Exception as exc:
            self._show_error(str(exc))

    def _update_controls(self) -> None:
        connected = self.service.connected
        armed = self.service.armed
        manual = self.mode_combo.currentText() == "Manual"
        eeg_ready = self.eeg_streaming and self.prediction_active and self.model_has_stop_class
        self.arm_button.setEnabled(connected and (manual or eeg_ready))
        for direction, button in self.direction_buttons.items():
            button.setEnabled(connected if direction == "STOP" else connected and armed and manual)
        if self.mode_combo.currentText() == "EEG Control" and not eeg_ready:
            self.decision_status_label.setText(
                "EEG mode requires a running stream, live prediction, and a model containing Rest/Stop."
            )

    def _show_error(self, message: str) -> None:
        self.status_label.setText(f"Error: {message}")
        self.status_label.setStyleSheet(themed_label_style("danger"))
        self.logger.error("Wheelchair panel: %s", message)
