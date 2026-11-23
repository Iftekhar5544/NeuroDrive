from __future__ import annotations

import logging
import os
import sys
import warnings
import ctypes
from pathlib import Path



os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts=false")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*", category=UserWarning)


def configure_logging() -> logging.Logger:
    """Console diagnostics only; never create files, folders or logging threads."""
    logger = logging.getLogger("neurowave")
    logger.setLevel(logging.WARNING)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
        logger.addHandler(handler)
    return logger


def main() -> int:
    logger = configure_logging()
    try:
        return _run_app(logger)
    except Exception:
        logger.exception("Application startup or event loop failed.")
        raise


def _run_app(logger) -> int:

    if sys.platform.startswith("win"):
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("NeuroWave.EEGMonitor")
        except Exception:
            pass

    try:
        from PyQt5 import QtGui, QtWidgets
    except ImportError as exc:
        print("PyQt5 is not installed. Run `pip install -r requirements.txt` first.", file=sys.stderr)
        logger.exception("PyQt5 import failed.")
        return 1

    try:
        import numpy  # noqa: F401
    except ImportError as exc:
        print("Missing runtime dependency. Run `pip install -r requirements.txt` first.", file=sys.stderr)
        logger.exception("Runtime dependency import failed.")
        return 1

    from app_theme import apply_dark_theme
    from ui.main_window import MainWindow

    app = QtWidgets.QApplication(sys.argv)
    apply_dark_theme(app, font_size=14)
    icon_path = Path(__file__).resolve().parent / "images" / "app_icon.png"
    if icon_path.exists():
        app_icon = QtGui.QIcon(str(icon_path))
        app.setWindowIcon(app_icon)
    else:
        app_icon = QtGui.QIcon()
    window = MainWindow(logger=logger)
    if not app_icon.isNull():
        window.setWindowIcon(app_icon)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
