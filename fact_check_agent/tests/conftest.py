"""Pytest configuration — ensures memory_agent is importable as `src.*`."""
import sys
from pathlib import Path

_MEMORY_AGENT = str(Path(__file__).resolve().parent.parent.parent / "memory_agent")
if _MEMORY_AGENT not in sys.path:
    sys.path.insert(0, _MEMORY_AGENT)
