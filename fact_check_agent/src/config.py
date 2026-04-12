"""Settings for the Fact-Check Agent.

Extends memory_agent's Settings with LangSmith tracing fields.
The memory_agent path is bootstrapped here so all downstream imports work.
"""
from src._bootstrap import *  # noqa: F401,F403 — sets up sys.path for memory_agent

from pydantic_settings import SettingsConfigDict
from src.config import Settings as _MemorySettings  # memory_agent's Settings


class Settings(_MemorySettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LangSmith / Langfuse tracing
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "fakenews-factcheck"


settings = Settings()
