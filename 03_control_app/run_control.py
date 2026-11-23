"""Launch the integrated NeuroWave EEG wheelchair application."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
CODE_DIR = PROJECT_DIR / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from main import main  


if __name__ == "__main__":
    sys.exit(main())

