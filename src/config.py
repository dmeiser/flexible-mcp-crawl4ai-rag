"""
Application settings and enums for the Crawl4AI MCP server.
"""

from enum import Enum
from typing import Optional

from pydantic import PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class ChunkStrategy(str, Enum):
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"
    FIXED = "fixed"
    SEMANTIC = "semantic"


class ContentClass(str, Enum):
    TEXT = "text"
    CODE = "code"
    STRUCTURED = "structured"
    MARKDOWN_RAW = "markdown_raw"
    MARKDOWN_FIT = "markdown_fit"


class MarkdownIndexPolicy(str, Enum):
    RAW_ONLY = "raw-only"
    FIT_ONLY = "fit-only"
    BOTH_BY_DEFAULT = "both-by-default"


# ---------------------------------------------------------------------------
# Application settings
# ---------------------------------------------------------------------------
class Settings(BaseSettings):
    POSTGRES_URL: PostgresDsn

    EMBEDDING_DIM: int = 768

    # OpenAI-compatible embedding settings
    EMBEDDING_BASE_URL: Optional[str] = None
    EMBEDDING_API_KEY: Optional[str] = None
    EMBEDDING_MODEL_NAME: str = "nomic-embed-text"
    EMBEDDING_MAX_RETRIES: int = 3
    EMBEDDING_RETRY_DELAY_SECONDS: float = 1.0

    BATCH_SIZE: int = 50

    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    CHUNK_STRATEGY: ChunkStrategy = ChunkStrategy.PARAGRAPH

    # RAG feature flags
    USE_CONTEXTUAL_EMBEDDINGS: bool = False
    USE_HYBRID_SEARCH: bool = False
    USE_AGENTIC_RAG: bool = False
    USE_RERANKING: bool = False

    # Markdown indexing policy
    MARKDOWN_INDEX_POLICY: MarkdownIndexPolicy = MarkdownIndexPolicy.BOTH_BY_DEFAULT
    MARKDOWN_FALLBACK_ENABLED: bool = True

    # ---------------------------------------------------------------------------
    # Default LLM — shared fallback for all LLM-powered features below.
    # If a feature-specific override is not set it falls back to these values.
    # ---------------------------------------------------------------------------
    DEFAULT_LLM_BASE_URL: Optional[str] = None
    DEFAULT_LLM_API_KEY: Optional[str] = None
    DEFAULT_LLM_MODEL_NAME: Optional[str] = None
    DEFAULT_LLM_MAX_RETRIES: int = 3
    DEFAULT_LLM_RETRY_DELAY_SECONDS: float = 1.0

    # Per-feature LLM overrides — all fall back to DEFAULT_LLM_* when unset.

    # 1. Contextual embeddings (requires USE_CONTEXTUAL_EMBEDDINGS=true)
    CONTEXTUAL_LLM_BASE_URL: Optional[str] = None
    CONTEXTUAL_LLM_API_KEY: Optional[str] = None
    CONTEXTUAL_LLM_MODEL_NAME: Optional[str] = None
    CONTEXTUAL_LLM_MAX_RETRIES: Optional[int] = None
    CONTEXTUAL_LLM_RETRY_DELAY_SECONDS: Optional[float] = None

    # 2. Agentic RAG (requires USE_AGENTIC_RAG=true)
    AGENTIC_LLM_BASE_URL: Optional[str] = None
    AGENTIC_LLM_API_KEY: Optional[str] = None
    AGENTIC_LLM_MODEL_NAME: Optional[str] = None
    AGENTIC_LLM_MAX_RETRIES: Optional[int] = None
    AGENTIC_LLM_RETRY_DELAY_SECONDS: Optional[float] = None

    # 3. Reranking (requires USE_RERANKING=true)
    RERANK_LLM_BASE_URL: Optional[str] = None
    RERANK_LLM_API_KEY: Optional[str] = None
    RERANK_LLM_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_LLM_MAX_RETRIES: Optional[int] = None
    RERANK_LLM_RETRY_DELAY_SECONDS: Optional[float] = None

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ---------------------------------------------------------------------------
    # Computed effective configs (feature override → DEFAULT_LLM_* fallback)
    # ---------------------------------------------------------------------------
    @property
    def effective_embedding_base_url(self) -> Optional[str]:
        return self.EMBEDDING_BASE_URL

    @property
    def effective_embedding_model_name(self) -> str:
        return self.EMBEDDING_MODEL_NAME

    @property
    def effective_embedding_max_retries(self) -> int:
        return int(self.EMBEDDING_MAX_RETRIES)

    @property
    def effective_embedding_retry_delay_seconds(self) -> float:
        return float(self.EMBEDDING_RETRY_DELAY_SECONDS)

    @property
    def effective_contextual_base_url(self) -> Optional[str]:
        return self.CONTEXTUAL_LLM_BASE_URL or self.DEFAULT_LLM_BASE_URL

    @property
    def effective_contextual_api_key(self) -> Optional[str]:
        return self.CONTEXTUAL_LLM_API_KEY or self.DEFAULT_LLM_API_KEY

    @property
    def effective_contextual_model_name(self) -> Optional[str]:
        return self.CONTEXTUAL_LLM_MODEL_NAME or self.DEFAULT_LLM_MODEL_NAME

    @property
    def effective_contextual_max_retries(self) -> int:
        return int(self.CONTEXTUAL_LLM_MAX_RETRIES or self.DEFAULT_LLM_MAX_RETRIES)

    @property
    def effective_contextual_retry_delay_seconds(self) -> float:
        return float(self.CONTEXTUAL_LLM_RETRY_DELAY_SECONDS or self.DEFAULT_LLM_RETRY_DELAY_SECONDS)

    @property
    def effective_agentic_base_url(self) -> Optional[str]:
        return self.AGENTIC_LLM_BASE_URL or self.DEFAULT_LLM_BASE_URL

    @property
    def effective_agentic_api_key(self) -> Optional[str]:
        return self.AGENTIC_LLM_API_KEY or self.DEFAULT_LLM_API_KEY

    @property
    def effective_agentic_model_name(self) -> Optional[str]:
        return self.AGENTIC_LLM_MODEL_NAME or self.DEFAULT_LLM_MODEL_NAME

    @property
    def effective_agentic_max_retries(self) -> int:
        return int(self.AGENTIC_LLM_MAX_RETRIES or self.DEFAULT_LLM_MAX_RETRIES)

    @property
    def effective_agentic_retry_delay_seconds(self) -> float:
        return float(self.AGENTIC_LLM_RETRY_DELAY_SECONDS or self.DEFAULT_LLM_RETRY_DELAY_SECONDS)

    @property
    def effective_rerank_base_url(self) -> Optional[str]:
        return self.RERANK_LLM_BASE_URL or self.DEFAULT_LLM_BASE_URL

    @property
    def effective_rerank_api_key(self) -> Optional[str]:
        return self.RERANK_LLM_API_KEY or self.DEFAULT_LLM_API_KEY

    @property
    def effective_rerank_model_name(self) -> str:
        return self.RERANK_LLM_MODEL_NAME or self.DEFAULT_LLM_MODEL_NAME or "cross-encoder/ms-marco-MiniLM-L-6-v2"

    @property
    def effective_rerank_max_retries(self) -> int:
        return int(self.RERANK_LLM_MAX_RETRIES or self.DEFAULT_LLM_MAX_RETRIES)

    @property
    def effective_rerank_retry_delay_seconds(self) -> float:
        return float(self.RERANK_LLM_RETRY_DELAY_SECONDS or self.DEFAULT_LLM_RETRY_DELAY_SECONDS)


settings = Settings()  # type: ignore[call-arg]
