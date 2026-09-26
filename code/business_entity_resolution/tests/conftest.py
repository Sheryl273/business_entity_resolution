"""
conftest.py — pytest fixtures shared across Person C test modules.
"""
import sys
from pathlib import Path

# Ensure src/ is importable from any test file
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
