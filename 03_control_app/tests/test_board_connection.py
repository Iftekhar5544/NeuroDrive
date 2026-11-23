from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from brainflow.exit_codes import BrainFlowError, BrainFlowExitCodes
from core.board_service import BoardService


class BoardConnectionTests(unittest.TestCase):
    def test_not_ready_releases_session_and_allows_retry(self):
        error = BrainFlowError("unable to prepare streaming session", BrainFlowExitCodes.BOARD_NOT_READY_ERROR)
        failed = Mock()
        failed.prepare_session.side_effect = error
        # Cleanup must not mask the connection error if preparation failed.
        failed.release_session.side_effect = error
        ready = Mock()
        ready.get_board_id.return_value = 2
        service = BoardService(logger=Mock())
        with patch("core.board_service.BoardShim") as shim:
            shim.side_effect = [failed, ready]
            shim.get_eeg_channels.return_value = list(range(1, 17))
            shim.get_sampling_rate.return_value = 125
            with self.assertRaises(RuntimeError) as caught:
                service.connect("COM3")
            self.assertIs(caught.exception.__cause__, error)
            for hint in ("COM3", "PC", "GPIO6", "battery"):
                self.assertIn(hint, str(caught.exception))
            failed.release_session.assert_called_once()
            self.assertIsNone(service.board)
            self.assertIsNone(service.board_id)
            self.assertFalse(service.connected)
            self.assertFalse(service.streaming)
            self.assertEqual(service._eeg_channels, [])
            service.connect("COM3")
            self.assertTrue(service.connected)
            self.assertEqual(service.get_sample_rate(), 125)
            self.assertEqual(len(service.get_eeg_channels()), 16)
            service.disconnect()
            ready.release_session.assert_called_once()

    def test_simulator_failure_preserves_endpoint_without_cyton_guidance(self):
        error = BrainFlowError("unable to prepare streaming session", BrainFlowExitCodes.BOARD_NOT_READY_ERROR)
        with patch("core.board_service.BoardShim") as shim:
            shim.return_value.prepare_session.side_effect = error
            service = BoardService(logger=Mock())
            with self.assertRaises(RuntimeError) as caught:
                service.connect("sim://127.0.0.1:6677")
            self.assertIn("127.0.0.1:6677", str(caught.exception))
            self.assertNotIn("GPIO6", str(caught.exception))
            shim.return_value.release_session.assert_called_once()
            self.assertFalse(service.connected)


if __name__ == "__main__":
    unittest.main()
