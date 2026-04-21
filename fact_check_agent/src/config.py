"""Settings for the Fact-Check Agent.

Standalone pydantic-settings config — all fields declared here directly.
Memory-agent's Settings is no longer inherited to avoid the bare-`src`
namespace collision between fact_check_agent/src and memory_agent/src.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Neo4j Aura (passed through to MemoryAgent at runtime)
    neo4j_uri: str = ""
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    # ChromaDB — cloud (leave blank to use local Docker instance)
    chroma_api_key: str = ""
    chroma_tenant: str = ""
    chroma_database: str = ""
    # ChromaDB — local Docker (used when chroma_api_key is blank)
    chroma_host: str = ""
    chroma_port: int = 8000

    # OpenAI
    openai_api_key: str = ""

    # Tavily Search API
    tavily_api_key: str = ""

    # Telegram scraper (unused by fact_check_agent — kept for MemoryAgent parity)
    telegram_scraper_api_url: str = ""
    telegram_scraper_api_key: str = ""

    # Model settings
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o"

    # LLM provider toggle: "openai" | "ollama"
    llm_provider: str = "openai"
    embedding_provider: str = "openai"  # changing to "ollama" requires re-seeding ChromaDB

    # Ollama settings (only used when *_provider = "ollama")
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_llm_model: str = "gemma4:e2b"
    ollama_embedding_model: str = "nomic-embed-text"
    ollama_vlm_model: str = ""             # vision-language model for caption generation; leave blank to skip

    # Retrieval enhancements
    use_graph_rag: bool = False       # enable Neo4j entity-claim traversal in query_memory
    use_cross_encoder: bool = False   # rerank merged results with cross-encoder (requires sentence-transformers)
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 5           # final number of chunks passed to synthesizer

    # Cross-modal consistency — SigLIP
    use_siglip: bool = False          # use SigLIP embedding similarity instead of VLM for image check
    siglip_model: str = "google/siglip-base-patch16-224"
    siglip_threshold: float = 0.10    # sigmoid probability below this → conflict flagged

    # S2 — Adaptive Retrieval Gate
    use_retrieval_gate: bool = False  # ask LLM if live search is needed before calling Tavily

    # S3 — Claim Decomposition
    use_claim_decomposition: bool = False  # split compound claims before retrieval

    # S4 — Multi-Agent Debate
    use_debate: bool = False               # enable advocate/arbiter debate for low-confidence verdicts
    debate_confidence_threshold: int = 70  # trigger debate when confidence_score < this value

    # S6 — Freshness ReAct Agent
    use_freshness_react: bool = False      # replace single LLM call with tool-calling ReAct loop

    # Benchmark / evaluation
    dry_run: bool = False                  # skip all DB writes (ChromaDB + Neo4j) during benchmark runs
    offline_mode: bool = False             # skip all DB reads+writes — implies dry_run; no Docker needed

    # LangSmith tracing
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "fakenews-factcheck"


settings = Settings()
