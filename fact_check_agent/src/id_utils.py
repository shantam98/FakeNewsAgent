"""Re-export make_id from memory_agent.

Bootstrap must run before this import, which is guaranteed by importing
config (or _bootstrap) first in any entry point.
"""
from src._bootstrap import *  # noqa: F401,F403

from src.id_utils import make_id  # memory_agent's id_utils

__all__ = ["make_id"]
