"""Fixed joystick voltages from the new wheelchair firmware.

The Arduino owns the DAC conversion. Python displays configured targets; it
never sends raw codes or custom speed settings.
"""
from __future__ import annotations

DAC_VDD_MV = 5040
CENTER_MV = 2500
LOW_MV = 550
HIGH_MV = 4450
DIRECTION_VOLTAGES_MV = {
    "STOP": (CENTER_MV, CENTER_MV),
    "FORWARD": (CENTER_MV, LOW_MV),
    "BACKWARD": (CENTER_MV, HIGH_MV),
    "LEFT": (LOW_MV, CENTER_MV),
    "RIGHT": (HIGH_MV, CENTER_MV),
}


def direction_output(direction: str) -> tuple[int, int]:
    """Return the fixed X/Y target in millivolts, including neutral STOP."""
    name = str(direction).strip().upper()
    if name not in DIRECTION_VOLTAGES_MV:
        raise ValueError(f"Unknown wheelchair direction: {direction}")
    return DIRECTION_VOLTAGES_MV[name]


def format_output(direction: str) -> str:
    x_mv, y_mv = direction_output(direction)
    return f"X {x_mv / 1000:.2f} V | Y {y_mv / 1000:.2f} V"
