import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_MEMORY_AGENT = str(_ROOT / "memory_agent")
if _MEMORY_AGENT not in sys.path:
    sys.path.insert(0, _MEMORY_AGENT)

os.environ.setdefault("NEO4J_URI",      "bolt://localhost:7687")
os.environ.setdefault("NEO4J_PASSWORD", "fakenews123")
os.environ.setdefault("OPENAI_API_KEY", "unused")
