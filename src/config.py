"""Central configuration for Oracle.

Every tunable lives here so nothing else hardcodes a model name, path, or
chunk size. Override anything with an `ORACLE_`-prefixed environment variable
or a local `.env` file::

    ORACLE_PROVIDER=groq
    ORACLE_TOP_K=8
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime configuration, populated from env vars / `.env` / defaults."""

    model_config = SettingsConfigDict(
        env_prefix="ORACLE_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------------------------------------------------------------- paths
    data_dir: Path = PROJECT_ROOT / "data"
    chroma_dir: Path = PROJECT_ROOT / "chroma"
    collection_name: str = "oracle_docs"

    # ------------------------------------------------------------ provider
    # "ollama" = local (dev, demos). "groq" = hosted, same Llama 3.1 weights,
    # used for the publicly deployed app where Ollama cannot run.
    provider: Literal["ollama", "groq"] = "ollama"

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"
    groq_model: str = "llama-3.1-8b-instant"
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")

    # Router only has to emit a one-word decision, so it can run on something
    # smaller/faster. Empty = reuse the main model (no extra `ollama pull`).
    router_model: str = ""

    temperature: float = 0.2      # low: this is factual Q&A over documents
    num_ctx: int = 8192
    request_timeout: int = 120

    # ---------------------------------------------------------- embeddings
    embed_model: str = "all-MiniLM-L6-v2"   # 384-dim, ~80MB, fine on CPU

    # ------------------------------------------------- chunking / retrieval
    chunk_size: int = 800          # tokens
    chunk_overlap: int = 120       # tokens
    top_k: int = 5
    fetch_k: int = 20              # candidates pulled before filtering
    score_threshold: float = 0.25  # cosine floor; below this -> no evidence

    # --------------------------------------------------------------- agent
    max_iterations: int = 2        # hard cap on the synthesis -> research loop
    max_history_turns: int = 10

    @property
    def model_name(self) -> str:
        """The generation model for the active provider."""
        return self.ollama_model if self.provider == "ollama" else self.groq_model

    @property
    def router_model_name(self) -> str:
        """The router's model, falling back to the main one."""
        return self.router_model or self.model_name


settings = Settings()
