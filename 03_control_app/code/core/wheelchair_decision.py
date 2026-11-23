from __future__ import annotations

import time
from collections import Counter, deque

from PyQt5 import QtCore


DEFAULT_EEG_MOVEMENT_DURATION_MS = 3000
MIN_EEG_MOVEMENT_DURATION_MS = 100
MAX_EEG_MOVEMENT_DURATION_MS = 60000


class WheelchairDecisionController(QtCore.QObject):
    """Optionally vote on EEG predictions and hold directions for the saved duration."""

    decision_ready = QtCore.pyqtSignal(object)
    status_changed = QtCore.pyqtSignal(str)

    MOVEMENT_LABELS = {"forward": "FORWARD", "left": "LEFT", "right": "RIGHT"}
    STOP_LABELS = {"rest", "stop", "idle", "neutral", "no_command", "no command"}

    def __init__(
        self,
        minimum_confidence: float = 0.75,
        history_size: int = 5,
        votes_required: int = 1,
        prediction_timeout_ms: int = 1500,
        parent=None,
    ) -> None:
        super().__init__(parent)
        if history_size < 1:
            raise ValueError("history_size must be positive")
        if votes_required < 1 or votes_required > history_size:
            raise ValueError("votes_required must be between 1 and history_size")
        self.minimum_confidence = float(minimum_confidence)
        self.history_size = int(history_size)
        self.votes_required = int(votes_required)
        self.prediction_timeout_ms = int(prediction_timeout_ms)
        self.movement_duration_ms = DEFAULT_EEG_MOVEMENT_DURATION_MS
        self.enabled = False
        self.history: deque[str] = deque(maxlen=self.history_size)
        self.last_prediction_at = 0.0
        self.last_decision = "STOP"
        self.movement_deadline_at = 0.0

        self.movement_timer = QtCore.QTimer(self)
        self.movement_timer.setSingleShot(True)
        self.movement_timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.movement_timer.setInterval(self.movement_duration_ms)
        self.movement_timer.timeout.connect(self._finish_movement)

        self.watchdog_timer = QtCore.QTimer(self)
        self.watchdog_timer.setInterval(200)
        self.watchdog_timer.timeout.connect(self._check_prediction_timeout)

    def configure(self, minimum_confidence: float, votes_required: int) -> None:
        confidence = float(minimum_confidence)
        votes = int(votes_required)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("minimum confidence must be between 0 and 1")
        if not 1 <= votes <= self.history_size:
            raise ValueError(f"votes required must be between 1 and {self.history_size}")
        self.minimum_confidence = confidence
        self.votes_required = votes
        self.reset("Decision settings updated.")

    def set_movement_duration(self, duration_ms: int) -> None:
        if (
            isinstance(duration_ms, bool)
            or not isinstance(duration_ms, int)
            or not MIN_EEG_MOVEMENT_DURATION_MS <= duration_ms <= MAX_EEG_MOVEMENT_DURATION_MS
        ):
            raise ValueError("Movement duration must be between 0.1 and 60 seconds.")
        self.reset("Movement duration updated.")
        self.movement_duration_ms = duration_ms
        self.movement_timer.setInterval(duration_ms)

    def set_enabled(self, enabled: bool) -> None:
        requested = bool(enabled)
        if requested == self.enabled:
            return
        self.enabled = requested
        self.history.clear()
        self.last_prediction_at = time.monotonic() if requested else 0.0
        if requested:
            self.watchdog_timer.start()
            waiting = "a prediction" if self.votes_required == 1 else "agreeing predictions"
            self.status_changed.emit(f"EEG decision controller armed; waiting for {waiting}.")
        else:
            self.watchdog_timer.stop()
            self._emit_stop("EEG decision controller disabled.")

    def reset(self, reason: str = "Decision history reset.") -> None:
        self.history.clear()
        self.last_prediction_at = time.monotonic() if self.enabled else 0.0
        self._emit_stop(reason)

    def process_prediction(self, result: dict) -> None:
        if not self.enabled:
            return
        self.last_prediction_at = time.monotonic()

        # An accepted direction is latched for its full saved duration. Keep
        # receiving results for the watchdog, but do not act on or queue them.
        if self.movement_deadline_at:
            if self.last_prediction_at < self.movement_deadline_at:
                return
            self._finish_movement()
            # Sending the expiry STOP can synchronously disable us on I/O failure.
            if not self.enabled:
                return

        raw_label = str(result.get("label", "")).strip()
        normalized = raw_label.casefold().replace("-", "_")
        confidence = float(result.get("confidence", 0.0) or 0.0)

        if normalized in self.STOP_LABELS:
            self.history.clear()
            self._emit_stop(f"Safe class detected: {raw_label or 'Rest'}.", confidence)
            return
        if normalized not in self.MOVEMENT_LABELS:
            self.history.clear()
            self._emit_stop(f"Unmapped class '{raw_label or 'Unknown'}'; stopped.", confidence)
            return
        if confidence < self.minimum_confidence:
            self.history.clear()
            self._emit_stop(
                f"Confidence {confidence:.0%} is below {self.minimum_confidence:.0%}; stopped.",
                confidence,
            )
            return

        direction = self.MOVEMENT_LABELS[normalized]
        vote_history = []
        voting_summary = "no voting"
        if self.votes_required > 1:
            self.history.append(direction)
            counts = Counter(self.history)
            votes = int(counts.get(direction, 0))
            if votes < self.votes_required:
                self._emit_stop(
                    f"Waiting for {self.votes_required} agreeing predictions ({votes}/{self.votes_required}).",
                    confidence,
                )
                return
            vote_history = list(self.history)
            voting_summary = f"{votes} votes"

        self.last_decision = direction
        self.history.clear()
        self.movement_deadline_at = time.monotonic() + self.movement_duration_ms / 1000.0
        self.movement_timer.start()
        payload = {
            "direction": direction,
            "confidence": confidence,
            "reason": (
                f"Accepted {direction}: {voting_summary} at {confidence:.0%}; "
                f"moving for {self.movement_duration_ms / 1000:g} seconds."
            ),
            "history": vote_history,
            "duration_ms": self.movement_duration_ms,
        }
        self.status_changed.emit(payload["reason"])
        self.decision_ready.emit(payload)

    def _finish_movement(self) -> None:
        if not self.movement_deadline_at:
            return
        self.history.clear()
        waiting = "a new prediction" if self.votes_required == 1 else "new votes"
        self._emit_stop(
            f"{self.movement_duration_ms / 1000:g}-second movement completed; waiting for {waiting}."
        )

    def _emit_stop(self, reason: str, confidence: float = 0.0) -> None:
        self.movement_timer.stop()
        self.movement_deadline_at = 0.0
        self.last_decision = "STOP"
        payload = {
            "direction": "STOP",
            "confidence": float(confidence),
            "reason": str(reason),
            "history": list(self.history),
        }
        self.decision_ready.emit(payload)
        self.status_changed.emit(str(reason))

    def _check_prediction_timeout(self) -> None:
        if not self.enabled or self.last_prediction_at <= 0.0:
            return
        now = time.monotonic()
        if self.movement_deadline_at and now >= self.movement_deadline_at:
            self._finish_movement()
            return
        elapsed_ms = (now - self.last_prediction_at) * 1000.0
        if elapsed_ms > float(self.prediction_timeout_ms):
            self.history.clear()
            self.last_prediction_at = time.monotonic()
            self._emit_stop("Prediction stream timed out; stopped.")

