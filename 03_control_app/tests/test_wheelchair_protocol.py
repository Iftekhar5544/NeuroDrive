from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from core import wheelchair_outputs, wheelchair_protocol  # noqa: E402
from core.wheelchair_protocol import build_command  # noqa: E402


class WheelchairProtocolTests(unittest.TestCase):
    def test_wire_commands_match_new_controller(self) -> None:
        for name, expected in (
            ("FORWARD", b"w"), ("BACKWARD", b"s"),
            ("LEFT", b"a"), ("RIGHT", b"d"), ("STOP", b"x"),
        ):
            with self.subTest(direction=name):
                self.assertEqual(build_command(name), expected)
        self.assertEqual(build_command(" forward "), b"w")

    def test_legacy_and_unknown_commands_are_rejected(self) -> None:
        for command in ("CMD", "ARM", "DISARM", "MOVE", ""):
            with self.subTest(command=command), self.assertRaises(ValueError):
                build_command(command)

    def test_app_targets_and_timing_match_firmware(self) -> None:
        firmware = (ROOT / "01_arduino_code" / "arduino_code.ino").read_text(encoding="utf-8")
        for name in ("DAC_VDD_MV", "CENTER_MV", "LOW_MV", "HIGH_MV"):
            match = re.search(rf"const uint16_t {name}\s*=\s*(\d+);", firmware)
            self.assertIsNotNone(match, name)
            self.assertEqual(int(match.group(1)), getattr(wheelchair_outputs, name))
        match = re.search(r"COMMAND_TIMEOUT_MS\s*=\s*(\d+);", firmware)
        self.assertEqual(int(match.group(1)), wheelchair_protocol.COMMAND_TIMEOUT_MS)
        self.assertLess(wheelchair_protocol.SEND_INTERVAL_MS, wheelchair_protocol.COMMAND_TIMEOUT_MS)
        self.assertIn(f"HC06.begin({wheelchair_protocol.BLUETOOTH_BAUD_RATE})", firmware)
        self.assertIn(f"Serial.begin({wheelchair_protocol.USB_BAUD_RATE})", firmware)
        for command in ("w", "s", "a", "d", "x"):
            self.assertIn(f"case '{command}':", firmware)


if __name__ == "__main__":
    unittest.main()
