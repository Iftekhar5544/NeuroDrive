from __future__ import annotations

import logging

from PyQt5 import QtCore

try:
    import serial
    from serial.tools import list_ports
except ImportError as exc:  # pragma: no cover
    serial = None
    list_ports = None
    SERIAL_IMPORT_ERROR = exc
else:
    SERIAL_IMPORT_ERROR = None

from core.wheelchair_outputs import CENTER_MV, direction_output
from core.wheelchair_protocol import BLUETOOTH_BAUD_RATE, SEND_INTERVAL_MS, build_command


class WheelchairService(QtCore.QObject):
    """Send fixed direction commands; arming is an application-side gate.

    command_changed reports configured millivolt targets, not measured feedback.
    The firmware independently stops after 800 ms without movement refreshes.
    """

    connection_changed = QtCore.pyqtSignal(bool, str)
    armed_changed = QtCore.pyqtSignal(bool)
    command_changed = QtCore.pyqtSignal(str, int, int)
    heartbeat_changed = QtCore.pyqtSignal(bool)
    error = QtCore.pyqtSignal(str)

    def __init__(self, logger: logging.Logger | None = None, parent=None) -> None:
        super().__init__(parent)
        self.logger = logger or logging.getLogger(__name__)
        self.serial_port = None
        self.connected = False
        self.armed = False
        self.port_name = ""
        self.current_direction = "STOP"
        self.current_x_mv = CENTER_MV
        self.current_y_mv = CENTER_MV
        self.heartbeat_timer = QtCore.QTimer(self)
        self.heartbeat_timer.setInterval(SEND_INTERVAL_MS)
        self.heartbeat_timer.timeout.connect(self._send_heartbeat)

    @staticmethod
    def available_ports() -> list[str]:
        if SERIAL_IMPORT_ERROR is not None or list_ports is None:
            return []
        return [item.device for item in sorted(list_ports.comports(), key=lambda item: item.device)]

    def connect_port(self, port: str, baudrate: int = BLUETOOTH_BAUD_RATE) -> None:
        if SERIAL_IMPORT_ERROR is not None or serial is None:
            raise RuntimeError("PySerial is not installed.") from SERIAL_IMPORT_ERROR
        if self.connected:
            raise RuntimeError("Wheelchair is already connected.")
        selected = str(port).strip()
        if not selected:
            raise ValueError("Select a wheelchair COM port first.")
        try:
            self.serial_port = serial.Serial(
                selected, int(baudrate), timeout=0, write_timeout=0.2,
            )
            self.port_name = selected
            self.connected = True
            self.armed = False
            self.stop_motion()
        except Exception:
            if self.serial_port is not None:
                try:
                    self.serial_port.close()
                except Exception:
                    pass
                self.serial_port = None
            self._set_disconnected()
            raise
        self.logger.info("Wheelchair controller connected on %s at %s baud", selected, baudrate)
        self.connection_changed.emit(True, selected)
        self.armed_changed.emit(False)

    def disconnect(self) -> None:
        try:
            if self.connected:
                self.stop_motion()
        except Exception as exc:
            self.logger.warning("Could not send final wheelchair STOP: %s", exc)
        finally:
            if self.serial_port is not None:
                try:
                    self.serial_port.close()
                except Exception:
                    pass
            self.serial_port = None
            self._set_disconnected()

    def arm(self) -> None:
        if not self.connected:
            raise RuntimeError("Connect the wheelchair controller before arming.")
        self.stop_motion()
        self.armed = True
        self.armed_changed.emit(True)
        self.logger.info("Wheelchair output enabled in app.")

    def disarm(self) -> None:
        try:
            self.stop_motion()
        finally:
            self.armed = False
            self.armed_changed.emit(False)

    def set_output(self, direction: str) -> None:
        if not self.connected:
            raise RuntimeError("Wheelchair controller is not connected.")
        if not self.armed:
            raise RuntimeError("Wheelchair output is not armed.")
        name = str(direction).strip().upper()
        try:
            x_mv, y_mv = direction_output(name)
        except ValueError:
            self.emergency_stop()
            raise
        if name == "STOP":
            self.stop_motion()
            return
        self._write_command(name)
        self.current_direction = name
        self.current_x_mv = x_mv
        self.current_y_mv = y_mv
        self.heartbeat_timer.start()
        self.heartbeat_changed.emit(True)
        self.command_changed.emit(name, x_mv, y_mv)

    def stop_motion(self) -> None:
        # Cancel refreshes before serial I/O, including when the STOP write fails.
        self.heartbeat_timer.stop()
        self.current_direction = "STOP"
        self.current_x_mv = CENTER_MV
        self.current_y_mv = CENTER_MV
        self.heartbeat_changed.emit(False)
        if self.connected:
            self._write_command("STOP")
        self.command_changed.emit("STOP", CENTER_MV, CENTER_MV)

    def emergency_stop(self) -> None:
        self.disarm()
        self.logger.warning("Wheelchair emergency stop requested.")

    def shutdown(self) -> None:
        self.disconnect()

    def _send_heartbeat(self) -> None:
        if not self.connected or not self.armed or self.current_direction == "STOP":
            self.heartbeat_timer.stop()
            self.heartbeat_changed.emit(False)
            return
        try:
            self._write_command(self.current_direction)
        except Exception:
            # _write_command already cancels refreshes and drops the connection.
            pass

    def _write_command(self, direction: str) -> None:
        payload = build_command(direction)
        try:
            if self.serial_port is None or not self.serial_port.is_open:
                raise RuntimeError("Wheelchair serial port is not open.")
            written = int(self.serial_port.write(payload))
            if written != len(payload):
                raise RuntimeError("Wheelchair serial write was incomplete.")
        except Exception as exc:
            self.logger.exception("Wheelchair command write failed")
            self._handle_io_error(exc)
            raise

    def _handle_io_error(self, exc: Exception) -> None:
        message = f"Wheelchair communication failed: {exc}"
        self.logger.error(message)
        self.heartbeat_timer.stop()
        if self.serial_port is not None:
            try:
                written = int(self.serial_port.write(build_command("STOP")))
                if written != 1:
                    raise RuntimeError("Recovery STOP write was incomplete.")
            except Exception:
                self.logger.exception("Recovery STOP failed")
            try:
                self.serial_port.close()
            except Exception:
                pass
        self.serial_port = None
        self._set_disconnected()
        self.error.emit(message)

    def _set_disconnected(self) -> None:
        self.heartbeat_timer.stop()
        self.connected = False
        self.armed = False
        self.port_name = ""
        self.current_direction = "STOP"
        self.current_x_mv = CENTER_MV
        self.current_y_mv = CENTER_MV
        self.connection_changed.emit(False, "")
        self.armed_changed.emit(False)
        self.heartbeat_changed.emit(False)
        self.command_changed.emit("STOP", CENTER_MV, CENTER_MV)
