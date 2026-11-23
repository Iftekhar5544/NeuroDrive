from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from PyQt5 import QtWidgets  # noqa: E402
from core.wheelchair_service import WheelchairService  # noqa: E402


class FakeSerial:
    def __init__(self) -> None:
        self.is_open = True
        self.writes = []
        self.fail = False
        self.short_write = False

    def write(self, payload: bytes) -> int:
        if self.fail:
            raise OSError("connection lost")
        self.writes.append(payload)
        return 0 if self.short_write else len(payload)

    def close(self) -> None:
        self.is_open = False


class WheelchairServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self) -> None:
        self.serial = FakeSerial()
        self.service = WheelchairService()
        self.outputs = []
        self.service.command_changed.connect(lambda *args: self.outputs.append(args))
        self.errors = []
        self.service.error.connect(self.errors.append)
        self.serial_patch = patch("core.wheelchair_service.serial.Serial", return_value=self.serial)
        self.open_serial = self.serial_patch.start()

    def tearDown(self) -> None:
        self.service.shutdown()
        self.serial_patch.stop()

    def test_connect_and_arm_send_only_stop_then_movement_refreshes(self) -> None:
        self.service.connect_port("TEST")
        self.open_serial.assert_called_once_with("TEST", 9600, timeout=0, write_timeout=0.2)
        self.assertEqual(self.serial.writes, [b"x"])
        self.assertFalse(self.service.armed)
        with self.assertRaises(RuntimeError):
            self.service.set_output("FORWARD")
        self.service.arm()
        self.assertEqual(self.serial.writes, [b"x", b"x"])
        self.assertFalse(self.service.heartbeat_timer.isActive())
        self.service.set_output("FORWARD")
        self.assertEqual(self.outputs[-1], ("FORWARD", 2500, 550))
        self.assertEqual(self.serial.writes[-1], b"w")
        self.assertEqual(self.service.heartbeat_timer.interval(), 200)
        self.assertTrue(self.service.heartbeat_timer.isActive())
        self.service._send_heartbeat()
        self.assertEqual(self.serial.writes[-2:], [b"w", b"w"])
        self.service.stop_motion()
        self.assertEqual(self.serial.writes[-1], b"x")
        self.assertEqual(self.outputs[-1], ("STOP", 2500, 2500))
        self.assertFalse(self.service.heartbeat_timer.isActive())
        writes_after_stop = list(self.serial.writes)
        self.service._send_heartbeat()
        self.assertEqual(self.serial.writes, writes_after_stop)

    def test_each_direction_sends_one_fixed_character(self) -> None:
        self.service.connect_port("TEST", 115200)
        self.open_serial.assert_called_once_with("TEST", 115200, timeout=0, write_timeout=0.2)
        self.service.arm()
        for name, command, target in (
            ("FORWARD", b"w", (2500, 550)),
            ("BACKWARD", b"s", (2500, 4450)),
            ("LEFT", b"a", (550, 2500)),
            ("RIGHT", b"d", (4450, 2500)),
        ):
            with self.subTest(direction=name):
                self.service.set_output(name)
                self.assertEqual(self.serial.writes[-1], command)
                self.assertEqual(self.outputs[-1], (name, *target))

    def test_disarm_emergency_stop_and_disconnect_cancel_movement(self) -> None:
        self.service.connect_port("TEST")
        for stop in (self.service.disarm, self.service.emergency_stop):
            self.service.arm()
            self.service.set_output("RIGHT")
            stop()
            self.assertEqual(self.serial.writes[-1], b"x")
            self.assertFalse(self.service.armed)
            self.assertFalse(self.service.heartbeat_timer.isActive())
        self.service.arm()
        self.service.set_output("LEFT")
        self.service.disconnect()
        self.assertEqual(self.serial.writes[-1], b"x")
        self.assertFalse(self.serial.is_open)
        self.assertFalse(self.service.connected)
        self.assertFalse(self.service.armed)

    def test_invalid_direction_stops_existing_motion(self) -> None:
        self.service.connect_port("TEST")
        self.service.arm()
        self.service.set_output("FORWARD")
        with self.assertRaises(ValueError):
            self.service.set_output("DIAGONAL")
        self.assertEqual(self.serial.writes[-1], b"x")
        self.assertFalse(self.service.armed)
        self.assertFalse(self.service.heartbeat_timer.isActive())

    def test_movement_stop_and_heartbeat_write_failures_drop_connection(self) -> None:
        for operation in ("movement", "stop", "heartbeat", "partial"):
            with self.subTest(operation=operation):
                self.serial = FakeSerial()
                self.open_serial.return_value = self.serial
                self.service.connect_port("TEST")
                self.service.arm()
                self.service.set_output("FORWARD")
                if operation == "partial":
                    self.serial.short_write = True
                else:
                    self.serial.fail = True
                if operation == "heartbeat":
                    self.service._send_heartbeat()
                else:
                    with self.assertRaises((OSError, RuntimeError)):
                        if operation == "stop":
                            self.service.stop_motion()
                        else:
                            self.service.set_output("LEFT")
                self.assertFalse(self.service.connected)
                self.assertFalse(self.service.armed)
                self.assertFalse(self.service.heartbeat_timer.isActive())
                self.assertFalse(self.serial.is_open)
                self.assertEqual(self.service.current_direction, "STOP")
                self.assertEqual(self.outputs[-1], ("STOP", 2500, 2500))
        self.assertEqual(len(self.errors), 4)

    def test_reconnecting_after_failure_requires_fresh_arming(self) -> None:
        self.service.connect_port("TEST")
        self.service.arm()
        self.service.set_output("FORWARD")
        self.serial.fail = True
        self.service._send_heartbeat()
        self.serial = FakeSerial()
        self.open_serial.return_value = self.serial
        self.service.connect_port("NEW")
        self.assertFalse(self.service.armed)
        self.assertEqual(self.serial.writes, [b"x"])
        with self.assertRaises(RuntimeError):
            self.service.set_output("FORWARD")


if __name__ == "__main__":
    unittest.main()
