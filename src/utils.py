"""
Utility functions for the Crawl4AI MCP server.
PostgreSQL/pgvector backend with dual embedding provider (OpenAI or Ollama).
"""

import asyncio
import hashlib
import json
import logging
import math
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple, cast

import httpx
import numpy as np
from openai import AsyncOpenAI, OpenAI
from pgvector.sqlalchemy import Vector
from pydantic import PostgresDsn, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Column, Field, Session, SQLModel, create_engine, select

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------
class EmbeddingError(Exception):
    """Raised when embedding creation fails."""


# Keep legacy alias so old imports don't break during transition
OllamaError = EmbeddingError


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class ChunkStrategy(str, Enum):
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"
    FIXED = "fixed"
    SEMANTIC = "semantic"


class EmbeddingProvider(str, Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"


class LLMProvider(str, Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"


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

    # Embedding provider: "openai" or "ollama"
    EMBEDDING_PROVIDER: EmbeddingProvider = EmbeddingProvider.OLLAMA
    EMBEDDING_DIM: int = 768

    # Provider-agnostic embedding settings (preferred)
    EMBEDDING_BASE_URL: Optional[str] = None
    EMBEDDING_API_KEY: Optional[str] = None
    EMBEDDING_MODEL_NAME: Optional[str] = None
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
    DEFAULT_LLM_PROVIDER: LLMProvider = LLMProvider.OPENAI
    DEFAULT_LLM_BASE_URL: Optional[str] = None
    DEFAULT_LLM_API_KEY: Optional[str] = None
    DEFAULT_LLM_MODEL_NAME: Optional[str] = None

    # Per-feature LLM overrides — all fall back to DEFAULT_LLM_* when unset.
    # A feature's LLM is considered active when its effective model name is set.

    # 1. Contextual embeddings (requires USE_CONTEXTUAL_EMBEDDINGS=true)
    CONTEXTUAL_LLM_PROVIDER: Optional[LLMProvider] = None
    CONTEXTUAL_LLM_BASE_URL: Optional[str] = None
    CONTEXTUAL_LLM_API_KEY: Optional[str] = None
    CONTEXTUAL_LLM_MODEL_NAME: Optional[str] = None

    # 2. Hybrid search
    HYBRID_LLM_PROVIDER: Optional[LLMProvider] = None
    HYBRID_LLM_BASE_URL: Optional[str] = None
    HYBRID_LLM_API_KEY: Optional[str] = None
    HYBRID_LLM_MODEL_NAME: Optional[str] = None

    # 3. Agentic RAG (requires USE_AGENTIC_RAG=true)
    AGENTIC_LLM_PROVIDER: Optional[LLMProvider] = None
    AGENTIC_LLM_BASE_URL: Optional[str] = None
    AGENTIC_LLM_API_KEY: Optional[str] = None
    AGENTIC_LLM_MODEL_NAME: Optional[str] = None

    # 4. Reranking (requires USE_RERANKING=true)
    #    RERANK_LLM_MODEL_NAME also doubles as the local CrossEncoder model name
    #    when no BASE_URL is configured.
    RERANK_LLM_PROVIDER: Optional[LLMProvider] = None
    RERANK_LLM_BASE_URL: Optional[str] = None
    RERANK_LLM_API_KEY: Optional[str] = None
    RERANK_LLM_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ---------------------------------------------------------------------------
    # Computed effective configs (feature override → DEFAULT_LLM_* fallback)
    # ---------------------------------------------------------------------------
    @property
    def effective_embedding_base_url(self) -> Optional[str]:
        return self.EMBEDDING_BASE_URL

    @property
    def effective_embedding_ollama_url(self) -> str:
        return self.EMBEDDING_BASE_URL or "http://localhost:11434/api/embeddings"

    @property
    def effective_embedding_model_name(self) -> str:
        if self.EMBEDDING_MODEL_NAME:
            return self.EMBEDDING_MODEL_NAME
        if self.EMBEDDING_PROVIDER == EmbeddingProvider.OPENAI:
            return "text-embedding-3-small"
        return "nomic-embed-text"

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
    def effective_contextual_provider(self) -> LLMProvider:
        return self.CONTEXTUAL_LLM_PROVIDER or self.DEFAULT_LLM_PROVIDER

    @property
    def effective_contextual_api_key(self) -> Optional[str]:
        return self.CONTEXTUAL_LLM_API_KEY or self.DEFAULT_LLM_API_KEY

    @property
    def effective_contextual_model_name(self) -> Optional[str]:
        return self.CONTEXTUAL_LLM_MODEL_NAME or self.DEFAULT_LLM_MODEL_NAME

    @property
    def effective_hybrid_base_url(self) -> Optional[str]:
        return self.HYBRID_LLM_BASE_URL or self.DEFAULT_LLM_BASE_URL

    @property
    def effective_hybrid_provider(self) -> LLMProvider:
        return self.HYBRID_LLM_PROVIDER or self.DEFAULT_LLM_PROVIDER

    @property
    def effective_hybrid_api_key(self) -> Optional[str]:
        return self.HYBRID_LLM_API_KEY or self.DEFAULT_LLM_API_KEY

    @property
    def effective_hybrid_model_name(self) -> Optional[str]:
        return self.HYBRID_LLM_MODEL_NAME or self.DEFAULT_LLM_MODEL_NAME

    @property
    def effective_agentic_base_url(self) -> Optional[str]:
        return self.AGENTIC_LLM_BASE_URL or self.DEFAULT_LLM_BASE_URL

    @property
    def effective_agentic_provider(self) -> LLMProvider:
        return self.AGENTIC_LLM_PROVIDER or self.DEFAULT_LLM_PROVIDER

    @property
    def effective_agentic_api_key(self) -> Optional[str]:
        return self.AGENTIC_LLM_API_KEY or self.DEFAULT_LLM_API_KEY

    @property
    def effective_agentic_model_name(self) -> Optional[str]:
        return self.AGENTIC_LLM_MODEL_NAME or self.DEFAULT_LLM_MODEL_NAME

    @property
    def effective_rerank_base_url(self) -> Optional[str]:
        return self.RERANK_LLM_BASE_URL or self.DEFAULT_LLM_BASE_URL

    @property
    def effective_rerank_provider(self) -> LLMProvider:
        return self.RERANK_LLM_PROVIDER or self.DEFAULT_LLM_PROVIDER

    @property
    def effective_rerank_api_key(self) -> Optional[str]:
        return self.RERANK_LLM_API_KEY or self.DEFAULT_LLM_API_KEY

    @property
    def effective_rerank_model_name(self) -> str:
        return self.RERANK_LLM_MODEL_NAME or self.DEFAULT_LLM_MODEL_NAME or "cross-encoder/ms-marco-MiniLM-L-6-v2"

    @model_validator(mode="after")
    def check_openai_config_if_selected(self) -> "Settings":
        if self.EMBEDDING_PROVIDER == EmbeddingProvider.OPENAI and not self.EMBEDDING_API_KEY:
            raise ValueError("EMBEDDING_API_KEY must be set when EMBEDDING_PROVIDER=openai.")
        return self


settings = Settings()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------
engine = create_engine(str(settings.POSTGRES_URL), echo=False)


class CrawledPage(SQLModel, table=True):
    __tablename__ = "crawled_pages"
    id: Optional[int] = Field(default=None, primary_key=True)
    source_id: Optional[int] = Field(default=None, foreign_key="sources.id")
    url: str = Field(index=True)
    chunk_number: int
    content: str
    content_class: str = Field(default=ContentClass.TEXT.value, index=True)
    is_active: bool = Field(default=True, index=True)
    crawl_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), index=True)
    content_hash: str = Field(index=True)
    source_change_id: Optional[str] = Field(default=None, index=True)
    # Phase 9.5 - freshness lifecycle
    first_seen_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_crawled_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = Field(default=None)
    is_pinned: bool = Field(default=False, index=True)
    tombstoned_at: Optional[datetime] = Field(default=None, index=True)
    staleness_score: float = Field(default=0.0)
    # Phase 9.6 - value scoring and hit tracking
    value_score: float = Field(default=0.0)
    last_hit_at: Optional[datetime] = Field(default=None)
    hit_count: int = Field(default=0)
    page_metadata: Dict[str, Any] = Field(default={}, sa_column=Column("metadata", JSONB))
    retrieval_metadata: Dict[str, Any] = Field(default={}, sa_column=Column("retrieval_metadata", JSONB))
    embedding: List[float] = Field(sa_column=Column(Vector(settings.EMBEDDING_DIM)))


class CodeExample(SQLModel, table=True):
    __tablename__ = "code_examples"
    id: Optional[int] = Field(default=None, primary_key=True)
    source_id: Optional[int] = Field(default=None, foreign_key="sources.id")
    url: str = Field(index=True)
    chunk_number: int
    language: Optional[str] = None
    content: str
    summary: Optional[str] = None
    content_class: str = Field(default=ContentClass.CODE.value, index=True)
    is_active: bool = Field(default=True, index=True)
    crawl_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), index=True)
    content_hash: str = Field(index=True)
    source_change_id: Optional[str] = Field(default=None, index=True)
    # Phase 9.5 - freshness lifecycle
    first_seen_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_crawled_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = Field(default=None)
    is_pinned: bool = Field(default=False, index=True)
    tombstoned_at: Optional[datetime] = Field(default=None, index=True)
    staleness_score: float = Field(default=0.0)
    # Phase 9.6 - value scoring and hit tracking
    value_score: float = Field(default=0.0)
    last_hit_at: Optional[datetime] = Field(default=None)
    hit_count: int = Field(default=0)
    ex_metadata: Dict[str, Any] = Field(default={}, sa_column=Column("metadata", JSONB))
    embedding: List[float] = Field(sa_column=Column(Vector(settings.EMBEDDING_DIM)))


class SourcePolicy(SQLModel, table=True):
    """Per-source freshness and eviction configuration."""

    __tablename__ = "source_policies"
    source: str = Field(primary_key=True)
    ttl_days: int = Field(default=90)
    recrawl_interval_hours: int = Field(default=168)
    priority_weight: float = Field(default=1.0)
    min_active_docs: int = Field(default=20)
    max_source_size_mb: Optional[int] = Field(default=None)
    retry_backoff_base_hours: int = Field(default=2)
    max_retry_backoff_hours: int = Field(default=168)
    consecutive_failures: int = Field(default=0)
    next_retry_at: Optional[datetime] = Field(default=None)
    dead_page_failures_threshold: int = Field(default=3)


class StoragePolicy(SQLModel, table=True):
    """Global storage budget configuration (single-row table)."""

    __tablename__ = "storage_policies"
    id: Optional[int] = Field(default=None, primary_key=True)
    is_singleton: bool = Field(default=True)
    max_db_size_gb: float = Field(default=10.0)
    warn_threshold: float = Field(default=0.80)
    high_threshold: float = Field(default=0.90)
    hard_threshold: float = Field(default=1.00)
    target_post_evict_ratio: float = Field(default=0.75)
    tombstone_grace_hours: int = Field(default=24)
    max_crawled_pages_mb: Optional[int] = Field(default=None)
    max_code_examples_mb: Optional[int] = Field(default=None)


class EvictionAuditLog(SQLModel, table=True):
    """Audit log of tombstoning and eviction actions."""

    __tablename__ = "eviction_audit_log"
    id: Optional[int] = Field(default=None, primary_key=True)
    table_name: str
    record_id: int
    source: Optional[str] = None
    evicted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    reason: str
    value_score: float = Field(default=0.0)
    staleness_score: float = Field(default=0.0)
    was_pinned: bool = Field(default=False)


class Source(SQLModel, table=True):
    __tablename__ = "sources"
    id: Optional[int] = Field(default=None, primary_key=True)
    source: str = Field(unique=True, index=True)
    summary: Optional[str] = None


def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session


# ---------------------------------------------------------------------------
# Phase 9.5/9.6 — scoring, staleness, tombstoning
# ---------------------------------------------------------------------------


def compute_staleness_score(age_days: float, half_life_days: float = 90.0) -> float:
    """Exponential staleness decay.  Returns 0.0 (fresh) → ~1.0 (stale)."""
    return 1.0 - math.exp(-age_days / max(half_life_days, 1e-9))


def compute_value_score(
    hit_count: int = 0,
    content_density: float = 0.5,
    age_days: float = 0.0,
    near_dup_sim: float = 0.0,
    source_priority: float = 1.0,
    half_life_days: float = 90.0,
    max_hit_count: float = 100.0,
) -> float:
    """Composite value score (0–1 * source_priority).  Higher = more valuable.

    v = 0.35*utility + 0.20*quality + 0.25*freshness + 0.20*uniqueness
    """
    utility = min(1.0, math.log1p(hit_count) / math.log1p(max(max_hit_count, 1.0)))
    quality = max(0.0, content_density * (1.0 - near_dup_sim))
    freshness = math.exp(-age_days / max(half_life_days, 1e-9))
    uniqueness = max(0.0, 1.0 - near_dup_sim)
    raw = 0.35 * utility + 0.20 * quality + 0.25 * freshness + 0.20 * uniqueness
    return max(0.0, min(1.0, raw * source_priority))


def tombstone_records(
    session: Session,
    record_ids: List[int],
    table_name: str = "crawled_pages",
    reason: str = "manual",
) -> int:
    """Soft-delete records by setting tombstoned_at and is_active=False.

    Logs each eviction to eviction_audit_log.  Returns count tombstoned.
    """
    if not record_ids:
        return 0

    model_cls = _tombstone_model(table_name)
    if model_cls is None:
        logger.warning(f"Unknown table for tombstoning: {table_name}")
        return 0

    now = datetime.now(timezone.utc)
    count = 0
    try:
        records = session.exec(select(model_cls).where(cast(Any, model_cls.id).in_(record_ids))).all()
        for record in records:
            record_any = cast(Any, record)
            record_any.tombstoned_at = now
            record_any.is_active = False
            source_str = _extract_record_source(record_any)
            log_entry = EvictionAuditLog(
                table_name=table_name,
                record_id=int(record_any.id),
                source=source_str,
                evicted_at=now,
                reason=reason,
                value_score=record_any.value_score,
                staleness_score=record_any.staleness_score,
                was_pinned=record_any.is_pinned,
            )
            session.add(log_entry)
            count += 1
        session.commit()
    except SQLAlchemyError as exc:
        logger.error(f"Failed to tombstone records: {exc}")
        session.rollback()
        return 0

    return count


def _tombstone_model(table_name: str) -> type[CrawledPage] | type[CodeExample] | None:
    if table_name == "crawled_pages":
        return CrawledPage
    if table_name == "code_examples":
        return CodeExample
    return None


def _extract_record_source(record_any: Any) -> Optional[str]:
    if hasattr(record_any, "page_metadata") and isinstance(record_any.page_metadata, dict):
        return record_any.page_metadata.get("source")
    if hasattr(record_any, "ex_metadata") and isinstance(record_any.ex_metadata, dict):
        return record_any.ex_metadata.get("source")
    return None


def _get_db_size_bytes(session: Session) -> int:
    """Return current DB size in bytes via pg_database_size."""
    try:
        row = session.exec(text("SELECT pg_database_size(current_database())")).first()  # type: ignore[call-overload]
        return int(row[0]) if row else 0
    except Exception:
        return 0


def _parse_iso_datetime(value: Any) -> datetime:
    """Parse timestamp-like metadata values into timezone-aware UTC datetimes."""
    if isinstance(value, datetime):
        return _ensure_utc_datetime(value)
    if isinstance(value, str) and value.strip():
        parsed = _parse_iso_string(value.strip())
        if parsed is not None:
            return parsed
    return datetime.now(timezone.utc)


def _ensure_utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_iso_string(value: str) -> Optional[datetime]:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return _ensure_utc_datetime(parsed)


# ---------------------------------------------------------------------------
# Embedding creation — async, dual provider
# ---------------------------------------------------------------------------
def _normalize(vec: List[float]) -> List[float]:
    arr = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return vec
    return (arr / norm).tolist()


async def create_embedding(text: str) -> List[float]:
    """
    Create a normalized embedding for a single text string.
    Uses the provider configured in settings.EMBEDDING_PROVIDER.
    """
    if not text or not text.strip():
        raise EmbeddingError("Attempted to create embedding for empty string.")

    if settings.EMBEDDING_PROVIDER == EmbeddingProvider.OPENAI:
        return await _create_openai_embedding(text)
    return await _create_ollama_embedding(text)


async def _create_openai_embedding(text: str) -> List[float]:
    """Create embedding via OpenAI (or OpenAI-compatible) API."""
    client_kwargs: Dict[str, Any] = {
        "api_key": _resolved_openai_api_key(settings.EMBEDDING_API_KEY, LLMProvider.OPENAI),
    }
    if settings.effective_embedding_base_url:
        client_kwargs["base_url"] = settings.effective_embedding_base_url

    client = AsyncOpenAI(**client_kwargs)
    try:
        resp = await client.embeddings.create(
            model=settings.effective_embedding_model_name,
            input=text,
        )
        return _normalize(resp.data[0].embedding)
    except Exception as exc:
        raise EmbeddingError(f"OpenAI embedding failed: {exc}") from exc
    finally:
        await client.close()


async def _create_ollama_embedding(text: str, attempt: int = 1) -> List[float]:
    """Create embedding via Ollama REST API with retry logic."""
    async with httpx.AsyncClient(timeout=60) as client:
        for attempt in range(1, settings.effective_embedding_max_retries + 1):
            try:
                return await _request_ollama_embedding(client, text)
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                await _handle_ollama_retry_or_raise(attempt, exc)
            except httpx.HTTPStatusError as exc:
                raise EmbeddingError(f"Ollama HTTP error {exc.response.status_code}: {exc}") from exc
            except Exception as exc:
                raise EmbeddingError(f"Unexpected embedding error: {exc}") from exc
    raise EmbeddingError("Ollama embedding failed: exhausted retries")  # pragma: no cover


async def _request_ollama_embedding(client: httpx.AsyncClient, text: str) -> List[float]:
    resp = await client.post(
        settings.effective_embedding_ollama_url,
        json={"model": settings.effective_embedding_model_name, "prompt": text},
    )
    resp.raise_for_status()
    embedding = resp.json().get("embedding")
    if not embedding or not isinstance(embedding, list):
        raise EmbeddingError("Ollama returned invalid embedding format.")
    return _normalize(embedding)


async def _handle_ollama_retry_or_raise(attempt: int, exc: Exception) -> None:
    if attempt >= settings.effective_embedding_max_retries:
        raise EmbeddingError(
            f"Ollama request failed after {settings.effective_embedding_max_retries} attempts: {exc}"
        ) from exc
    delay = settings.effective_embedding_retry_delay_seconds * (2 ** (attempt - 1))
    await asyncio.sleep(delay)


async def create_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Create embeddings for multiple texts in parallel."""
    if not texts:
        return []
    return await asyncio.gather(*[create_embedding(t) for t in texts])


# ---------------------------------------------------------------------------
# Contextual embedding (LLM enrichment)
# ---------------------------------------------------------------------------
async def generate_contextual_text(full_document: str, chunk: str) -> Tuple[str, bool]:
    """
    Generate a contextual summary prefix for a chunk using an LLM.
    Returns (enriched_text, was_enriched).
    """
    if not settings.USE_CONTEXTUAL_EMBEDDINGS or not settings.effective_contextual_model_name:
        return chunk, False

    try:
        context_text = await _request_contextual_summary(full_document, chunk)
        enriched = _combine_context_and_chunk(context_text, chunk)
        if enriched is not None:
            return enriched, True
        return chunk, False
    except Exception as exc:
        logger.warning(f"Contextual embedding LLM call failed: {exc}")
        return chunk, False


def _context_prompt(full_document: str, chunk: str) -> str:
    return (
        f"<document>\n{full_document[:25000]}\n</document>\n"
        f"<chunk>\n{chunk}\n</chunk>\n"
        "Given the document and the specific chunk, write a 1-2 sentence "
        "summary capturing the chunk's topic and its relationship to the "
        "wider document. Use keywords that help semantic search.\n"
        "Summary:"
    )


async def _request_contextual_summary(full_document: str, chunk: str) -> str:
    if not settings.effective_contextual_model_name:
        logger.warning(
            "CONTEXTUAL_LLM_MODEL_NAME (or DEFAULT_LLM_MODEL_NAME) is not configured; skipping contextual enrichment."
        )
        return ""

    client = AsyncOpenAI(
        api_key=_resolved_openai_api_key(settings.effective_contextual_api_key, settings.effective_contextual_provider),
        base_url=settings.effective_contextual_base_url,
    )
    try:
        resp = await client.chat.completions.create(
            model=settings.effective_contextual_model_name,
            messages=[{"role": "user", "content": _context_prompt(full_document, chunk)}],
            max_tokens=150,
        )
    finally:
        await client.close()

    content = resp.choices[0].message.content
    return content.strip() if isinstance(content, str) else ""


def _combine_context_and_chunk(context_text: str, chunk: str) -> Optional[str]:
    if not context_text:
        return None
    combined = f"Context: {context_text}\n\n{chunk}"
    return combined[: settings.CHUNK_SIZE * 2]


# ---------------------------------------------------------------------------
# Source upsert
# ---------------------------------------------------------------------------
def upsert_source(session: Session, source: str) -> int:
    """Insert or return id of a source record."""
    existing = session.exec(select(Source).where(Source.source == source)).first()
    if existing:
        if existing.id is None:
            raise ValueError("Existing source row missing id")
        return existing.id
    obj = Source(source=source)
    session.add(obj)
    session.commit()
    session.refresh(obj)
    if obj.id is None:
        raise ValueError("Source insert did not return id")
    return obj.id


def _prepare_page_metadata(
    meta: Dict[str, Any],
    content: str,
) -> tuple[Dict[str, Any], Dict[str, Any], datetime, str, Optional[str]]:
    """Normalize page metadata and derive retrieval metadata + persisted fields."""
    meta_with_size = {"chunk_size": len(content), **meta}
    crawl_timestamp = _parse_iso_datetime(meta_with_size.get("crawl_timestamp") or meta_with_size.get("crawl_time"))

    content_hash = _resolve_content_hash(meta_with_size, content)
    source_change_id = _normalize_source_change_id(meta_with_size.get("source_change_id"))
    references_markdown = _normalize_references_markdown(meta_with_size.get("references_markdown"))
    link_references = _resolve_link_references(meta_with_size.get("link_references"), references_markdown)

    has_citations = bool(meta_with_size.get("has_citations", False))
    meta_with_size["references_markdown"] = references_markdown
    meta_with_size["link_references"] = link_references
    meta_with_size["has_citations"] = has_citations

    retrieval_metadata = {
        "source": meta_with_size.get("source"),
        "url": meta_with_size.get("url"),
        "crawl_type": meta_with_size.get("crawl_type"),
        "run_id": meta_with_size.get("run_id"),
        "markdown_variant": meta_with_size.get("markdown_variant"),
        "extraction_strategy": meta_with_size.get("extraction_strategy"),
        "session_id": meta_with_size.get("session_id"),
        "source_type": meta_with_size.get("source_type"),
        "crawl_timestamp": meta_with_size.get("crawl_timestamp") or meta_with_size.get("crawl_time"),
        "references_markdown": references_markdown,
        "link_references": link_references,
        "has_link_references": bool(references_markdown or link_references),
        "has_citations": has_citations,
    }

    return meta_with_size, retrieval_metadata, crawl_timestamp, content_hash, source_change_id


def _resolve_content_hash(meta: Dict[str, Any], content: str) -> str:
    content_hash = meta.get("content_hash")
    if isinstance(content_hash, str) and content_hash:
        return content_hash
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _normalize_source_change_id(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _normalize_references_markdown(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _resolve_link_references(value: Any, references_markdown: str) -> List[Dict[str, str]]:
    if isinstance(value, list):
        return value
    return extract_link_references(references_markdown)


def _build_crawled_page_row(
    session: Session,
    *,
    url: str,
    chunk_number: int,
    content: str,
    metadata: Dict[str, Any],
    embedding: List[float],
    first_seen_map: Dict[Tuple[str, int], datetime],
    now: datetime,
) -> CrawledPage:
    """Build one CrawledPage ORM row from normalized inputs."""
    source_str = metadata.get("source", "")
    source_id = upsert_source(session, source_str) if source_str else None
    meta_with_size, retrieval_metadata, crawl_timestamp, content_hash, source_change_id = _prepare_page_metadata(
        metadata,
        content,
    )

    return CrawledPage(
        source_id=source_id,
        url=url,
        chunk_number=chunk_number,
        content=content,
        content_class=str(meta_with_size.get("content_class") or ContentClass.TEXT.value),
        is_active=bool(meta_with_size.get("is_active", True)),
        crawl_timestamp=crawl_timestamp,
        content_hash=content_hash,
        source_change_id=source_change_id,
        page_metadata=meta_with_size,
        retrieval_metadata=retrieval_metadata,
        embedding=embedding,
        first_seen_at=first_seen_map.get((url, chunk_number), now),
        last_seen_at=now,
        last_crawled_at=crawl_timestamp,
    )


def _build_code_example_row(
    session: Session,
    *,
    url: str,
    chunk_number: int,
    content: str,
    language: Optional[str],
    summary: Optional[str],
    metadata: Dict[str, Any],
    embedding: List[float],
    first_seen_map: Dict[Tuple[str, int], datetime],
    now: datetime,
) -> CodeExample:
    """Build one CodeExample ORM row from normalized inputs."""
    source_str = metadata.get("source", "")
    source_id = upsert_source(session, source_str) if source_str else None

    content_hash = _resolve_content_hash(metadata, content)
    source_change_id = _normalize_source_change_id(metadata.get("source_change_id"))

    crawl_ts = _parse_iso_datetime(metadata.get("crawl_timestamp") or metadata.get("crawl_time"))
    return CodeExample(
        source_id=source_id,
        url=url,
        chunk_number=chunk_number,
        language=language,
        content=content,
        summary=summary,
        content_class=str(metadata.get("content_class") or ContentClass.CODE.value),
        is_active=bool(metadata.get("is_active", True)),
        crawl_timestamp=crawl_ts,
        content_hash=content_hash,
        source_change_id=source_change_id,
        ex_metadata=metadata,
        embedding=embedding,
        first_seen_at=first_seen_map.get((url, chunk_number), now),
        last_seen_at=now,
        last_crawled_at=crawl_ts,
    )


def _validate_document_batch_lengths(
    urls: List[str],
    contents: List[str],
    metadatas: List[Dict[str, Any]],
    chunk_numbers: List[int],
) -> bool:
    """Return True when document batch inputs are length-aligned."""
    return len(urls) == len(contents) == len(metadatas) == len(chunk_numbers)


def _collect_valid_document_entries(
    urls: List[str],
    contents: List[str],
    metadatas: List[Dict[str, Any]],
    chunk_numbers: List[int],
    full_documents: Optional[List[str]],
) -> List[tuple[str, str, Dict[str, Any], int, Optional[str]]]:
    """Filter out empty-content entries while preserving index alignment."""
    return [
        (url, content, metadata, chunk_number, (full_documents[index] if full_documents else None))
        for index, (url, content, metadata, chunk_number) in enumerate(zip(urls, contents, metadatas, chunk_numbers))
        if content and content.strip()
    ]


def _delete_existing_crawled_pages(
    session: Session,
    urls: List[str],
) -> Dict[Tuple[str, int], datetime]:
    """Delete existing pages for URLs and return preserved first_seen map."""
    first_seen_map: Dict[Tuple[str, int], datetime] = {}
    unique_urls = list(set(urls))
    try:
        to_delete = session.exec(select(CrawledPage).where(cast(Any, CrawledPage.url).in_(unique_urls))).all()
        for row in to_delete:
            first_seen_map[(row.url, row.chunk_number)] = row.first_seen_at
            session.delete(row)
        if to_delete:
            session.commit()
    except SQLAlchemyError as exc:
        logger.warning(f"Delete before insert failed: {exc}")
        session.rollback()
        return {}
    return first_seen_map


# ---------------------------------------------------------------------------
# Document storage
# ---------------------------------------------------------------------------
async def add_documents_to_db(
    session: Session,
    urls: List[str],
    contents: List[str],
    metadatas: List[Dict[str, Any]],
    chunk_numbers: List[int],
    full_documents: Optional[List[str]] = None,
) -> int:
    """
    Embed and store a batch of text chunks into crawled_pages.
    Deletes existing rows for those URLs first (upsert semantics).
    Returns total rows inserted.
    """
    valid = _prepare_valid_document_entries(urls, contents, metadatas, chunk_numbers, full_documents)
    if valid is None:
        return 0

    v_urls, v_contents, v_metas, v_chunks, v_fulldocs = valid

    first_seen_map = _delete_existing_crawled_pages(session, list(v_urls))

    embed_texts = await _embedding_texts_for_documents(v_contents, v_fulldocs, full_documents)
    return await _insert_document_batches(
        session=session,
        embed_texts=embed_texts,
        urls=v_urls,
        metadatas=v_metas,
        chunk_numbers=v_chunks,
        first_seen_map=first_seen_map,
    )


def _prepare_valid_document_entries(
    urls: List[str],
    contents: List[str],
    metadatas: List[Dict[str, Any]],
    chunk_numbers: List[int],
    full_documents: Optional[List[str]],
) -> Optional[tuple[Any, Any, Any, Any, Any]]:
    if not urls:
        return None
    if not _validate_document_batch_lengths(urls, contents, metadatas, chunk_numbers):
        logger.error("Input list lengths mismatch in add_documents_to_db.")
        return None
    valid = _collect_valid_document_entries(urls, contents, metadatas, chunk_numbers, full_documents)
    if not valid:
        return None
    v_urls, v_contents, v_metas, v_chunks, v_fulldocs = zip(*valid)
    return v_urls, v_contents, v_metas, v_chunks, v_fulldocs


async def _embedding_texts_for_documents(
    contents: Sequence[str],
    full_docs: Sequence[Optional[str]],
    full_documents: Optional[List[str]],
) -> List[str]:
    if not _should_enrich_with_context(full_documents):
        return list(contents)
    enriched = await asyncio.gather(*[generate_contextual_text(fd or "", c) for c, fd in zip(contents, full_docs)])
    return [entry[0] for entry in enriched]


def _should_enrich_with_context(full_documents: Optional[List[str]]) -> bool:
    return bool(settings.USE_CONTEXTUAL_EMBEDDINGS and settings.effective_contextual_model_name and full_documents)


async def _insert_document_batches(
    session: Session,
    embed_texts: List[str],
    urls: Sequence[str],
    metadatas: Sequence[Dict[str, Any]],
    chunk_numbers: Sequence[int],
    first_seen_map: Dict[Tuple[str, int], datetime],
) -> int:
    now_utc = datetime.now(timezone.utc)
    total_added = 0
    batch_size = settings.BATCH_SIZE
    for batch_start in range(0, len(embed_texts), batch_size):
        total_added += await _insert_single_document_batch(
            session,
            embed_texts,
            urls,
            metadatas,
            chunk_numbers,
            first_seen_map,
            now_utc,
            batch_start,
            batch_size,
        )
    return total_added


async def _insert_single_document_batch(
    session: Session,
    embed_texts: Sequence[str],
    urls: Sequence[str],
    metadatas: Sequence[Dict[str, Any]],
    chunk_numbers: Sequence[int],
    first_seen_map: Dict[Tuple[str, int], datetime],
    now_utc: datetime,
    batch_start: int,
    batch_size: int,
) -> int:
    batch_end = batch_start + batch_size
    b_texts = list(embed_texts[batch_start:batch_end])
    b_urls = list(urls[batch_start:batch_end])
    b_metas = list(metadatas[batch_start:batch_end])
    b_chunks = list(chunk_numbers[batch_start:batch_end])
    embeddings = await _safe_batch_embeddings(b_texts)
    if embeddings is None:
        return 0
    rows = _build_crawled_rows_batch(session, b_urls, b_chunks, b_texts, b_metas, embeddings, first_seen_map, now_utc)
    return _commit_rows(session, rows)


async def _safe_batch_embeddings(texts: List[str]) -> Optional[List[List[float]]]:
    try:
        return await create_embeddings_batch(texts)
    except EmbeddingError as exc:
        logger.error(f"Skipping batch due to embedding error: {exc}")
        return None


def _build_crawled_rows_batch(
    session: Session,
    urls: List[str],
    chunk_numbers: List[int],
    texts: List[str],
    metadatas: List[Dict[str, Any]],
    embeddings: List[List[float]],
    first_seen_map: Dict[Tuple[str, int], datetime],
    now_utc: datetime,
) -> List[CrawledPage]:
    return [
        _build_crawled_page_row(
            session,
            url=urls[i],
            chunk_number=chunk_numbers[i],
            content=texts[i],
            metadata=metadatas[i],
            embedding=embedding,
            first_seen_map=first_seen_map,
            now=now_utc,
        )
        for i, embedding in enumerate(embeddings)
    ]


def _commit_rows(session: Session, rows: List[CrawledPage]) -> int:
    try:
        session.add_all(rows)
        session.commit()
        return len(rows)
    except SQLAlchemyError as exc:
        logger.error(f"DB insert failed for batch: {exc}")
        session.rollback()
        return 0


async def add_code_examples_to_db(
    session: Session,
    urls: List[str],
    contents: List[str],
    languages: List[Optional[str]],
    summaries: List[Optional[str]],
    metadatas: List[Dict[str, Any]],
    chunk_numbers: List[int],
) -> int:
    """Embed and store code examples into code_examples table."""
    if not urls:
        return 0

    first_seen_code_map = _delete_existing_code_examples(session, urls)

    _now_code = datetime.now(timezone.utc)

    try:
        embeddings = await create_embeddings_batch(contents)
    except EmbeddingError as exc:
        logger.error(f"Code example embedding failed: {exc}")
        return 0

    rows = _build_code_example_rows(
        session,
        urls,
        contents,
        languages,
        summaries,
        metadatas,
        chunk_numbers,
        embeddings,
        first_seen_code_map,
        _now_code,
    )
    try:
        session.add_all(rows)
        session.commit()
        return len(rows)
    except SQLAlchemyError as exc:
        logger.error(f"Code example DB insert failed: {exc}")
        session.rollback()
        return 0


def _delete_existing_code_examples(session: Session, urls: List[str]) -> Dict[Tuple[str, int], datetime]:
    first_seen_code_map: Dict[Tuple[str, int], datetime] = {}
    unique_urls = list(set(urls))
    try:
        to_delete = session.exec(select(CodeExample).where(cast(Any, CodeExample.url).in_(unique_urls))).all()
        for row in to_delete:
            first_seen_code_map[(row.url, row.chunk_number)] = row.first_seen_at
            session.delete(row)
        if to_delete:
            session.commit()
        return first_seen_code_map
    except SQLAlchemyError as exc:
        logger.warning(f"Delete before code insert failed: {exc}")
        session.rollback()
        return {}


def _build_code_example_rows(
    session: Session,
    urls: List[str],
    contents: List[str],
    languages: List[Optional[str]],
    summaries: List[Optional[str]],
    metadatas: List[Dict[str, Any]],
    chunk_numbers: List[int],
    embeddings: List[List[float]],
    first_seen_code_map: Dict[Tuple[str, int], datetime],
    now_code: datetime,
) -> List[CodeExample]:
    return [
        _build_code_example_row(
            session,
            url=urls[i],
            chunk_number=chunk_numbers[i],
            content=contents[i],
            language=languages[i],
            summary=summaries[i],
            metadata=metadatas[i] if isinstance(metadatas[i], dict) else {},
            embedding=emb,
            first_seen_map=first_seen_code_map,
            now=now_code,
        )
        for i, emb in enumerate(embeddings)
    ]


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
async def search_documents(
    session: Session,
    query: str,
    match_count: int = 10,
    filter_metadata: Optional[Dict[str, Any]] = None,
    use_hybrid: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """
    Search crawled_pages using vector similarity, optionally fused with FTS.
    """
    if not query or not query.strip():
        logger.warning("Empty search query.")
        return []

    hybrid = _resolve_hybrid_mode(use_hybrid)
    query_embedding = await _safe_query_embedding(query)
    if query_embedding is None:
        return []

    filter_json = json.dumps(filter_metadata or {})
    return _search_documents_with_embedding(
        session=session,
        query=query,
        query_embedding=query_embedding,
        match_count=match_count,
        filter_metadata=filter_metadata,
        filter_json=filter_json,
        hybrid=hybrid,
    )


def _search_documents_with_embedding(
    session: Session,
    query: str,
    query_embedding: List[float],
    match_count: int,
    filter_metadata: Optional[Dict[str, Any]],
    filter_json: str,
    hybrid: bool,
) -> List[Dict[str, Any]]:
    vector_results = _vector_search_rows_to_results(
        _run_vector_search(
            session=session,
            query_embedding=query_embedding,
            filter_json=filter_json,
            match_count=match_count,
            hybrid=hybrid,
            filter_metadata=filter_metadata,
        )
    )

    if not hybrid:
        return vector_results[:match_count]

    fts_raw = _run_fts_search(
        session=session,
        query=query,
        filter_json=filter_json,
        match_count=match_count,
    )
    return _merge_hybrid_results(vector_results=vector_results, fts_raw=fts_raw, match_count=match_count)


def _resolve_hybrid_mode(use_hybrid: Optional[bool]) -> bool:
    if use_hybrid is None:
        return settings.USE_HYBRID_SEARCH
    return use_hybrid


async def _safe_query_embedding(query: str) -> Optional[List[float]]:
    try:
        return await create_embedding(query)
    except EmbeddingError as exc:
        logger.error(f"Failed to embed query: {exc}")
        return None


def _run_vector_search(
    session: Session,
    query_embedding: List[float],
    filter_json: str,
    match_count: int,
    hybrid: bool,
    filter_metadata: Optional[Dict[str, Any]],
) -> Sequence[Any]:
    """Run DB vector search with fallback to Python-side similarity."""
    try:
        return session.exec(  # type: ignore[call-overload]
            text(
                "SELECT id, url, chunk_number, content, metadata, similarity "
                "FROM match_crawled_pages(CAST(:emb AS vector), :cnt, CAST(:filt AS jsonb))"
            ).bindparams(
                emb=str(query_embedding),
                cnt=match_count * 2 if hybrid else match_count,
                filt=filter_json,
            )
        ).all()
    except Exception:
        return _python_side_vector_search(session, query_embedding, match_count * 2, filter_metadata)


def _vector_search_rows_to_results(raw_rows: Sequence[Any]) -> List[Dict[str, Any]]:
    """Normalize vector-search rows to API result dicts."""
    return [
        {
            "id": row[0],
            "url": row[1],
            "chunk_number": row[2],
            "content": row[3],
            "page_metadata": row[4] if isinstance(row[4], dict) else {},
            "similarity_score": float(row[5]),
            "fts_score": 0.0,
        }
        for row in raw_rows
    ]


def _run_fts_search(
    session: Session,
    query: str,
    filter_json: str,
    match_count: int,
) -> Sequence[Any]:
    """Run FTS query used for hybrid RRF search; return empty on failure."""
    try:
        return session.exec(  # type: ignore[call-overload]
            text(
                "SELECT id, url, chunk_number, content, metadata, "
                "ts_rank(fts, plainto_tsquery('english', :q)) AS rank "
                "FROM crawled_pages "
                "WHERE fts @@ plainto_tsquery('english', :q) "
                "AND is_active = TRUE "
                "AND tombstoned_at IS NULL "
                "AND CAST(:filt AS jsonb) <@ metadata "
                "ORDER BY rank DESC LIMIT :cnt"
            ).bindparams(q=query, filt=filter_json, cnt=match_count * 2)
        ).all()
    except Exception:
        return []


def _merge_hybrid_results(
    vector_results: List[Dict[str, Any]],
    fts_raw: Sequence[Any],
    match_count: int,
) -> List[Dict[str, Any]]:
    """Merge vector and FTS rows using reciprocal rank fusion."""
    fts_map = _fts_score_map(fts_raw)
    vector_rank = _rank_map_from_vector_results(vector_results)
    fts_rank = _rank_map_from_fts_rows(fts_raw)
    rrf_scores = _rrf_scores(vector_rank, fts_rank)
    id_to_result = _merge_vector_and_fts_rows(vector_results, fts_raw, fts_map)
    merged = sorted(id_to_result.values(), key=lambda result: rrf_scores.get(result["id"], 0), reverse=True)
    return merged[:match_count]


def _fts_score_map(fts_raw: Sequence[Any]) -> Dict[int, float]:
    return {int(row[0]): float(row[5]) for row in fts_raw}


def _rank_map_from_vector_results(vector_results: List[Dict[str, Any]]) -> Dict[int, int]:
    return {int(result["id"]): index + 1 for index, result in enumerate(vector_results)}


def _rank_map_from_fts_rows(fts_raw: Sequence[Any]) -> Dict[int, int]:
    return {int(row[0]): index + 1 for index, row in enumerate(fts_raw)}


def _rrf_scores(vector_rank: Dict[int, int], fts_rank: Dict[int, int]) -> Dict[int, float]:
    k = 60
    all_ids = set(vector_rank) | set(fts_rank)
    return {
        record_id: 1 / (k + vector_rank.get(record_id, len(all_ids) + k))
        + 1 / (k + fts_rank.get(record_id, len(all_ids) + k))
        for record_id in all_ids
    }


def _merge_vector_and_fts_rows(
    vector_results: List[Dict[str, Any]],
    fts_raw: Sequence[Any],
    fts_map: Dict[int, float],
) -> Dict[int, Dict[str, Any]]:
    id_to_result: Dict[int, Dict[str, Any]] = {int(result["id"]): result for result in vector_results}
    for row in fts_raw:
        row_id = int(row[0])
        if row_id in id_to_result:
            continue
        id_to_result[row_id] = {
            "id": row_id,
            "url": row[1],
            "chunk_number": row[2],
            "content": row[3],
            "page_metadata": row[4] if isinstance(row[4], dict) else {},
            "similarity_score": 0.0,
            "fts_score": fts_map.get(row_id, 0.0),
        }
    return id_to_result


def _python_side_vector_search(
    session: Session,
    query_embedding: List[float],
    limit: int,
    filter_metadata: Optional[Dict[str, Any]],
) -> List[Any]:
    """Fallback: compute cosine similarity in Python."""
    q = np.array(query_embedding, dtype=np.float32)
    pages = _active_pages(session.exec(select(CrawledPage)).all())
    pages = _metadata_filtered_pages(pages, filter_metadata)

    scored: List[Any] = []
    for p in pages:
        scored_row = _page_similarity_row(p, q)
        if scored_row is not None:
            scored.append(scored_row)

    scored.sort(key=lambda x: x[5], reverse=True)
    return scored[:limit]


def _active_pages(pages: Sequence[CrawledPage]) -> List[CrawledPage]:
    return [page for page in pages if page.is_active and page.tombstoned_at is None]


def _metadata_filtered_pages(
    pages: List[CrawledPage],
    filter_metadata: Optional[Dict[str, Any]],
) -> List[CrawledPage]:
    if not filter_metadata:
        return pages
    return [page for page in pages if _page_matches_metadata(page, filter_metadata)]


def _page_matches_metadata(page: CrawledPage, filter_metadata: Dict[str, Any]) -> bool:
    metadata = page.page_metadata if isinstance(page.page_metadata, dict) else {}
    return all(metadata.get(key) == value for key, value in filter_metadata.items())


def _page_similarity_row(page: CrawledPage, query_vec: np.ndarray[Any, Any]) -> Optional[tuple[Any, ...]]:
    if not page.embedding:
        return None
    similarity = _cosine_similarity(query_vec, np.array(page.embedding, dtype=np.float32))
    return (page.id, page.url, page.chunk_number, page.content, page.page_metadata, similarity)


def _cosine_similarity(vec_a: np.ndarray[Any, Any], vec_b: np.ndarray[Any, Any]) -> float:
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


async def search_code_examples(
    session: Session,
    query: str,
    match_count: int = 5,
    language: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Search code_examples by vector similarity."""
    if not query or not query.strip():
        return []

    try:
        query_embedding = await create_embedding(query)
    except EmbeddingError as exc:
        logger.error(f"Failed to embed code query: {exc}")
        return []

    filter_json = _code_examples_filter_json(language)
    raw = _query_code_example_rows(session, query_embedding, match_count, filter_json)
    return [_code_example_row_to_result(row) for row in raw]


def _code_examples_filter_json(language: Optional[str]) -> str:
    if not language:
        return json.dumps({})
    return json.dumps({"language": language})


def _query_code_example_rows(
    session: Session,
    query_embedding: List[float],
    match_count: int,
    filter_json: str,
) -> Sequence[Any]:
    try:
        return session.execute(
            text(
                "SELECT id, url, chunk_number, language, content, summary, metadata, similarity "
                "FROM match_code_examples(CAST(:emb AS vector), :cnt, CAST(:filt AS jsonb))"
            ).bindparams(emb=str(query_embedding), cnt=match_count, filt=filter_json)
        ).all()
    except Exception:
        return []


def _code_example_row_to_result(row: Any) -> Dict[str, Any]:
    return {
        "id": row[0],
        "url": row[1],
        "chunk_number": row[2],
        "language": row[3],
        "content": row[4],
        "summary": row[5],
        "metadata": row[6] if isinstance(row[6], dict) else {},
        "similarity_score": float(row[7]),
    }


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------
def rerank_results(
    query: str,
    results: List[Dict[str, Any]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Rerank results using a cross-encoder model.
    Falls back to original order if sentence_transformers unavailable.
    """
    if not settings.USE_RERANKING or not results:
        return results[:top_k]

    try:
        if settings.effective_rerank_base_url:
            return _rerank_with_openai_compatible_api(query, results, top_k)
        return _rerank_with_cross_encoder(query, results, top_k)
    except Exception as exc:
        logger.warning(f"Reranking failed, returning original order: {exc}")
        return results[:top_k]


def _resolved_openai_api_key(api_key: Optional[str], provider: LLMProvider) -> Optional[str]:
    if isinstance(api_key, str) and api_key.strip():
        return api_key
    if provider == LLMProvider.OLLAMA:
        return "ollama"
    return api_key


def _rerank_with_openai_compatible_api(
    query: str,
    results: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    client = OpenAI(
        api_key=_resolved_openai_api_key(settings.effective_rerank_api_key, settings.effective_rerank_provider),
        base_url=settings.effective_rerank_base_url,
    )
    try:
        response = client.chat.completions.create(
            model=settings.effective_rerank_model_name,
            messages=cast(Any, _rerank_messages(query, results)),
            temperature=0,
        )
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()

    scores = _parse_rerank_scores(response, expected_count=len(results))
    if scores is None:
        raise ValueError("Invalid rerank API response format")
    return _apply_rerank_scores(results, scores, top_k)


def _rerank_messages(query: str, results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    candidates = [
        {
            "index": idx,
            "content": str(result.get("content", ""))[:3000],
        }
        for idx, result in enumerate(results)
    ]
    return [
        {
            "role": "system",
            "content": (
                "You are a relevance ranker. Return strict JSON only in the format "
                '{"scores": [float, ...]} with one score per candidate index in order. '
                "Higher score means more relevant to the query."
            ),
        },
        {
            "role": "user",
            "content": json.dumps({"query": query, "candidates": candidates}, ensure_ascii=False),
        },
    ]


def _parse_rerank_scores(response: Any, expected_count: int) -> Optional[List[float]]:
    content = _rerank_response_content(response)
    if content is None:
        return None
    payload = _rerank_json_payload(content)
    scores = _rerank_score_values(payload, expected_count)
    if scores is None:
        return None
    return [float(score) for score in scores]


def _rerank_response_content(response: Any) -> Optional[str]:
    content = response.choices[0].message.content
    return content if isinstance(content, str) else None


def _rerank_json_payload(content: str) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(content)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _rerank_score_values(payload: Optional[Dict[str, Any]], expected_count: int) -> Optional[List[Any]]:
    if payload is None:
        return None
    scores = payload.get("scores")
    if not isinstance(scores, list) or len(scores) != expected_count:
        return None
    return scores


def _apply_rerank_scores(results: List[Dict[str, Any]], scores: List[float], top_k: int) -> List[Dict[str, Any]]:
    for index, result in enumerate(results):
        result["rerank_score"] = float(scores[index])
    return sorted(results, key=lambda item: item.get("rerank_score", 0), reverse=True)[:top_k]


def _rerank_with_cross_encoder(
    query: str,
    results: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    from sentence_transformers import CrossEncoder  # noqa: PLC0415

    model = CrossEncoder(settings.effective_rerank_model_name)
    pairs = [(query, result["content"]) for result in results]
    scores = model.predict(pairs)
    for index, result in enumerate(results):
        result["rerank_score"] = float(scores[index])
    return sorted(results, key=lambda item: item.get("rerank_score", 0), reverse=True)[:top_k]


# ---------------------------------------------------------------------------
# Code example extraction helper
# ---------------------------------------------------------------------------
def extract_code_blocks(markdown: str) -> List[Dict[str, Any]]:
    """
    Extract fenced code blocks from markdown text.
    Returns list of {language, content} dicts.
    """
    import re

    pattern = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
    blocks = []
    for match in pattern.finditer(markdown):
        lang = match.group(1).strip() or None
        code = match.group(2).strip()
        if code:
            blocks.append({"language": lang, "content": code})
    return blocks


def extract_link_references(references_markdown: str) -> List[Dict[str, str]]:
    """Extract structured link references from references markdown text."""
    if not references_markdown or not references_markdown.strip():
        return []

    extracted: List[Dict[str, str]] = []
    for line in references_markdown.splitlines():
        extracted_ref = _extract_reference_from_line(line, len(extracted))
        if extracted_ref is not None:
            extracted.append(extracted_ref)

    return extracted


def _extract_reference_from_line(line: str, existing_count: int) -> Optional[Dict[str, str]]:
    candidate = line.strip()
    if not candidate:
        return None

    for parser in (_parse_markdown_reference, _parse_markdown_link_reference, _parse_plain_url_reference):
        parsed = parser(candidate, existing_count)
        if parsed is not None:
            return parsed
    return None


def _parse_markdown_reference(candidate: str, existing_count: int) -> Optional[Dict[str, str]]:
    _ = existing_count
    markdown_ref = re.match(r"^\[(?P<label>[^\]]+)\]\s*:\s*(?P<url>https?://\S+)(?:\s+(?P<text>.*))?$", candidate)
    if not markdown_ref:
        return None
    return {
        "label": markdown_ref.group("label"),
        "url": markdown_ref.group("url"),
        "text": (markdown_ref.group("text") or "").strip(),
    }


def _parse_markdown_link_reference(candidate: str, existing_count: int) -> Optional[Dict[str, str]]:
    markdown_link = re.search(r"\[(?P<text>[^\]]+)\]\((?P<url>https?://[^)]+)\)", candidate)
    if not markdown_link:
        return None
    return {
        "label": str(existing_count + 1),
        "url": markdown_link.group("url"),
        "text": markdown_link.group("text").strip(),
    }


def _parse_plain_url_reference(candidate: str, existing_count: int) -> Optional[Dict[str, str]]:
    plain_url = re.search(r"(?P<url>https?://\S+)", candidate)
    if not plain_url:
        return None
    label_match = re.match(r"^(?P<label>\d+)[\.:\)]", candidate)
    return {
        "label": label_match.group("label") if label_match else str(existing_count + 1),
        "url": plain_url.group("url"),
        "text": candidate,
    }
