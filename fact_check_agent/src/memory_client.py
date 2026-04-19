"""MemoryAgent singleton for the Fact-Check Agent process.

Creates one MemoryAgent instance per process and reuses it across all
graph invocations. Neo4j driver connections are expensive — never
instantiate MemoryAgent inside a node or per-request.

Usage:
    from fact_check_agent.src.memory_client import get_memory, close_memory

    memory = get_memory()   # returns singleton; creates it on first call
    ...
    close_memory()          # call at process shutdown to close Neo4j driver
"""
import logging

from fact_check_agent.src._bootstrap import *  # noqa: F401,F403 — sets memory_agent on sys.path
from src.memory.agent import MemoryAgent  # memory_agent
from src.config import settings as _memory_settings  # memory_agent settings

logger = logging.getLogger(__name__)

_memory: MemoryAgent | None = None


def get_memory() -> MemoryAgent:
    """Return (or create) the process-level MemoryAgent singleton."""
    global _memory
    if _memory is None:
        logger.info("Initialising MemoryAgent singleton")
        _memory = MemoryAgent(_memory_settings)
    return _memory


def close_memory() -> None:
    """Close the MemoryAgent and its Neo4j driver. Call at process shutdown."""
    global _memory
    if _memory is not None:
        _memory.close()
        _memory = None
        logger.info("MemoryAgent closed")
