"""Bootstrap: add memory_agent to sys.path so its internal `src.*` imports resolve.

Import this module first in any file that needs to import from memory_agent.
Idempotent — safe to import multiple times.
"""
import sys
from pathlib import Path

_MEMORY_AGENT_ROOT = Path(__file__).resolve().parent.parent.parent / "memory_agent"

if str(_MEMORY_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_MEMORY_AGENT_ROOT))
