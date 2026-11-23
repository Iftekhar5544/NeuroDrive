"""Single-character protocol used by the HC-06 wheelchair firmware."""
from __future__ import annotations

BLUETOOTH_BAUD_RATE = 9600
USB_BAUD_RATE = 115200
SEND_INTERVAL_MS = 200
COMMAND_TIMEOUT_MS = 800
DIRECTION_COMMANDS = {
    "FORWARD": b"w",
    "BACKWARD": b"s",
    "LEFT": b"a",
    "RIGHT": b"d",
    "STOP": b"x",
}


def build_command(direction: str) -> bytes:
    name = str(direction).strip().upper()
    if name not in DIRECTION_COMMANDS:
        raise ValueError(f"Unsupported wheelchair direction: {direction}")
    return DIRECTION_COMMANDS[name]
