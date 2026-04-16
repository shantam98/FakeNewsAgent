"""Root conftest.py — sets up sys.path for all tests.

1. Blocks ROS testing plugins that are incompatible with pytest 9.
2. Adds memory_agent/ to sys.path so nodes that do lazy `from src.models.*`
   imports (e.g. write_memory) can find the memory_agent models.
"""
import sys
from pathlib import Path

# memory_agent lives next to fact_check_agent in the same repo
_MEMORY_AGENT_ROOT = Path(__file__).resolve().parent / "memory_agent"
if _MEMORY_AGENT_ROOT.exists() and str(_MEMORY_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_MEMORY_AGENT_ROOT))


def pytest_configure(config):
    config.pluginmanager.set_blocked("launch_testing_ros_pytest_entrypoint")
