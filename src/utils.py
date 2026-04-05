"""
Utility functions for the Crawl4AI MCP server.
PostgreSQL/pgvector backend with dual embedding provider (OpenAI or Ollama).
"""
import asyncio
import logging
import os
import json
import hashlib
import math
from enum import Enum
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Generator, Tuple

import httpx
import numpy as np
from openai import AsyncOpenAI
from pydantic import HttpUrl, PostgresDsn, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from sqlmodel import Field, SQLModel, create_engine, Session, select, Column
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector
from sqlalchemy.exc import SQLAlchemyError

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


class ContentClass(str, Enum):
    TEXT = "text"
    CODE = "code"
    STRUCTURED = "structured"
    MARKDOWN_RAW = "markdown_raw"
    MARKDOWN_FIT = "markdown_fit"


# ---------------------------------------------------------------------------
# Application settings
# ---------------------------------------------------------------------------
class Settings(BaseSettings):
    POSTGRES_URL: PostgresDsn

    # Embedding provider: "openai" or "ollama"
    EMBEDDING_PROVIDER: EmbeddingProvider = EmbeddingProvider.OLLAMA
    EMBEDDING_DIM: int = 768

    # Ollama settings (used when EMBEDDING_PROVIDER=ollama)
    OLLAMA_API_URL: str = "http://localhost:11434/api/embeddings"
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text"
    OLLAMA_MAX_RETRIES: int = 3
    OLLAMA_RETRY_DELAY_SECONDS: float = 1.0

    # OpenAI settings (used when EMBEDDING_PROVIDER=openai)
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_EMBED_MODEL: str = "text-embedding-3-small"
    OPENAI_BASE_URL: Optional[str] = None  # Override for Ollama OpenAI-compat endpoint

    BATCH_SIZE: int = 50

    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    CHUNK_STRATEGY: ChunkStrategy = ChunkStrategy.PARAGRAPH

    # RAG feature flags
    USE_CONTEXTUAL_EMBEDDINGS: bool = False
    USE_HYBRID_SEARCH: bool = False
    USE_AGENTIC_RAG: bool = False
    USE_RERANKING: bool = False

    # LLM for contextual embeddings
    LLM_ENABLED: bool = False
    LLM_API_KEY: Optional[str] = None
    LLM_BASE_URL: Optional[str] = None
    LLM_MODEL_NAME: Optional[str] = None

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @model_validator(mode="after")
    def check_llm_config_if_enabled(self) -> "Settings":
        if self.LLM_ENABLED:
            if not self.LLM_API_KEY:
                raise ValueError("LLM_API_KEY must be set when LLM_ENABLED is true.")
            if not self.LLM_BASE_URL:
                raise ValueError("LLM_BASE_URL must be set when LLM_ENABLED is true.")
            if not self.LLM_MODEL_NAME:
                raise ValueError("LLM_MODEL_NAME must be set when LLM_ENABLED is true.")
        return self

    @model_validator(mode="after")
    def check_openai_config_if_selected(self) -> "Settings":
        if self.EMBEDDING_PROVIDER == EmbeddingProvider.OPENAI and not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY must be set when EMBEDDING_PROVIDER=openai.")
        return self


settings = Settings()


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
    content_hash: Optional[str] = Field(default=None, index=True)
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
    content_hash: Optional[str] = Field(default=None, index=True)
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

    if table_name == "crawled_pages":
        model_cls = CrawledPage
    elif table_name == "code_examples":
        model_cls = CodeExample
    else:
        logger.warning(f"Unknown table for tombstoning: {table_name}")
        return 0

    now = datetime.now(timezone.utc)
    count = 0
    try:
        records = session.exec(
            select(model_cls).where(model_cls.id.in_(record_ids))  # type: ignore[attr-defined]
        ).all()
        for record in records:
            record.tombstoned_at = now
            record.is_active = False
            source_str: Optional[str] = None
            if hasattr(record, "page_metadata") and isinstance(record.page_metadata, dict):
                source_str = record.page_metadata.get("source")
            elif hasattr(record, "ex_metadata") and isinstance(record.ex_metadata, dict):
                source_str = record.ex_metadata.get("source")
            log_entry = EvictionAuditLog(
                table_name=table_name,
                record_id=record.id,
                source=source_str,
                evicted_at=now,
                reason=reason,
                value_score=record.value_score,
                staleness_score=record.staleness_score,
                was_pinned=record.is_pinned,
            )
            session.add(log_entry)
            count += 1
        session.commit()
    except SQLAlchemyError as exc:
        logger.error(f"Failed to tombstone records: {exc}")
        session.rollback()
        return 0

    return count


def _get_db_size_bytes(session: Session) -> int:
    """Return current DB size in bytes via pg_database_size."""
    try:
        row = session.exec(text("SELECT pg_database_size(current_database())")).first()
        return int(row[0]) if row else 0
    except Exception:
        return 0


def _parse_iso_datetime(value: Any) -> datetime:
    """Parse timestamp-like metadata values into timezone-aware UTC datetimes."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip())
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


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
    client_kwargs: Dict[str, Any] = {"api_key": settings.OPENAI_API_KEY}
    if settings.OPENAI_BASE_URL:
        client_kwargs["base_url"] = settings.OPENAI_BASE_URL

    client = AsyncOpenAI(**client_kwargs)
    try:
        resp = await client.embeddings.create(
            model=settings.OPENAI_EMBED_MODEL,
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
        for attempt in range(1, settings.OLLAMA_MAX_RETRIES + 1):
            try:
                resp = await client.post(
                    settings.OLLAMA_API_URL,
                    json={"model": settings.OLLAMA_EMBED_MODEL, "prompt": text},
                )
                resp.raise_for_status()
                data = resp.json()
                embedding = data.get("embedding")
                if not embedding or not isinstance(embedding, list):
                    raise EmbeddingError("Ollama returned invalid embedding format.")
                return _normalize(embedding)
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                if attempt < settings.OLLAMA_MAX_RETRIES:
                    delay = settings.OLLAMA_RETRY_DELAY_SECONDS * (2 ** (attempt - 1))
                    await asyncio.sleep(delay)
                    continue
                raise EmbeddingError(
                    f"Ollama request failed after {settings.OLLAMA_MAX_RETRIES} attempts: {exc}"
                ) from exc
            except httpx.HTTPStatusError as exc:
                raise EmbeddingError(f"Ollama HTTP error {exc.response.status_code}: {exc}") from exc
            except Exception as exc:
                raise EmbeddingError(f"Unexpected embedding error: {exc}") from exc
    raise EmbeddingError("Ollama embedding failed: exhausted retries")  # pragma: no cover


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
    if not settings.USE_CONTEXTUAL_EMBEDDINGS or not settings.LLM_ENABLED:
        return chunk, False

    try:
        prompt = (
            f"<document>\n{full_document[:25000]}\n</document>\n"
            f"<chunk>\n{chunk}\n</chunk>\n"
            "Given the document and the specific chunk, write a 1-2 sentence "
            "summary capturing the chunk's topic and its relationship to the "
            "wider document. Use keywords that help semantic search.\n"
            "Summary:"
        )
        client = AsyncOpenAI(
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL,
        )
        try:
            resp = await client.chat.completions.create(
                model=settings.LLM_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
            )
            context_text = resp.choices[0].message.content.strip()
        finally:
            await client.close()

        if context_text:
            combined = f"Context: {context_text}\n\n{chunk}"
            return combined[: settings.CHUNK_SIZE * 2], True
        return chunk, False
    except Exception as exc:
        logger.warning(f"Contextual embedding LLM call failed: {exc}")
        return chunk, False


# ---------------------------------------------------------------------------
# Source upsert
# ---------------------------------------------------------------------------
def upsert_source(session: Session, source: str) -> int:
    """Insert or return id of a source record."""
    existing = session.exec(select(Source).where(Source.source == source)).first()
    if existing:
        return existing.id
    obj = Source(source=source)
    session.add(obj)
    session.commit()
    session.refresh(obj)
    return obj.id


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
    if not urls:
        return 0

    # Validate lengths
    if not (len(urls) == len(contents) == len(metadatas) == len(chunk_numbers)):
        logger.error("Input list lengths mismatch in add_documents_to_db.")
        return 0

    # Collect valid items (non-empty content)
    valid = [
        (u, c, m, n, (full_documents[i] if full_documents else None))
        for i, (u, c, m, n) in enumerate(zip(urls, contents, metadatas, chunk_numbers))
        if c and c.strip()
    ]
    if not valid:
        return 0

    v_urls, v_contents, v_metas, v_chunks, v_fulldocs = zip(*valid)

    # Collect first_seen_at from existing rows before deletion (preserves re-index timestamps)
    first_seen_map: Dict[Tuple[str, int], datetime] = {}
    unique_urls = list(set(v_urls))
    try:
        to_delete = session.exec(select(CrawledPage).where(CrawledPage.url.in_(unique_urls))).all()
        for row in to_delete:
            first_seen_map[(row.url, row.chunk_number)] = row.first_seen_at
            session.delete(row)
        if to_delete:
            session.commit()
    except SQLAlchemyError as exc:
        logger.warning(f"Delete before insert failed: {exc}")
        first_seen_map.clear()
        session.rollback()

    # Apply contextual enrichment if enabled
    if settings.USE_CONTEXTUAL_EMBEDDINGS and settings.LLM_ENABLED and full_documents:
        enriched = await asyncio.gather(
            *[generate_contextual_text(fd or "", c) for c, fd in zip(v_contents, v_fulldocs)]
        )
        embed_texts = [e[0] for e in enriched]
    else:
        embed_texts = list(v_contents)

    # Embed in batches
    _now = datetime.now(timezone.utc)
    total_added = 0
    batch_size = settings.BATCH_SIZE
    for batch_start in range(0, len(embed_texts), batch_size):
        batch_end = batch_start + batch_size
        b_texts = embed_texts[batch_start:batch_end]
        b_urls = v_urls[batch_start:batch_end]
        b_metas = v_metas[batch_start:batch_end]
        b_chunks = v_chunks[batch_start:batch_end]

        try:
            embeddings = await create_embeddings_batch(b_texts)
        except EmbeddingError as exc:
            logger.error(f"Skipping batch due to embedding error: {exc}")
            continue

        rows = []
        for i, emb in enumerate(embeddings):
            source_str = b_metas[i].get("source", "")
            source_id = upsert_source(session, source_str) if source_str else None
            meta_with_size = {"chunk_size": len(b_texts[i]), **b_metas[i]}
            content_class = str(meta_with_size.get("content_class") or ContentClass.TEXT.value)
            is_active = bool(meta_with_size.get("is_active", True))
            crawl_timestamp = _parse_iso_datetime(
                meta_with_size.get("crawl_timestamp") or meta_with_size.get("crawl_time")
            )
            content_hash = meta_with_size.get("content_hash")
            if not isinstance(content_hash, str) or not content_hash:
                content_hash = hashlib.sha256(b_texts[i].encode("utf-8")).hexdigest()
            retrieval_metadata = {
                "source": meta_with_size.get("source"),
                "url": meta_with_size.get("url"),
                "crawl_type": meta_with_size.get("crawl_type"),
                "markdown_variant": meta_with_size.get("markdown_variant"),
                "extraction_strategy": meta_with_size.get("extraction_strategy"),
                "session_id": meta_with_size.get("session_id"),
                "source_type": meta_with_size.get("source_type"),
            }
            rows.append(
                CrawledPage(
                    source_id=source_id,
                    url=b_urls[i],
                    chunk_number=b_chunks[i],
                    content=b_texts[i],
                    content_class=content_class,
                    is_active=is_active,
                    crawl_timestamp=crawl_timestamp,
                    content_hash=content_hash,
                    page_metadata=meta_with_size,
                    retrieval_metadata=retrieval_metadata,
                    embedding=emb,
                    first_seen_at=first_seen_map.get((b_urls[i], b_chunks[i]), _now),
                    last_seen_at=_now,
                    last_crawled_at=crawl_timestamp,
                )
            )
        try:
            session.add_all(rows)
            session.commit()
            total_added += len(rows)
        except SQLAlchemyError as exc:
            logger.error(f"DB insert failed for batch: {exc}")
            session.rollback()

    return total_added


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

    # Delete existing
    first_seen_code_map: Dict[Tuple[str, int], datetime] = {}
    unique_urls = list(set(urls))
    try:
        to_delete = session.exec(select(CodeExample).where(CodeExample.url.in_(unique_urls))).all()
        for row in to_delete:
            first_seen_code_map[(row.url, row.chunk_number)] = row.first_seen_at
            session.delete(row)
        if to_delete:
            session.commit()
    except SQLAlchemyError as exc:
        logger.warning(f"Delete before code insert failed: {exc}")
        first_seen_code_map.clear()
        session.rollback()

    _now_code = datetime.now(timezone.utc)

    try:
        embeddings = await create_embeddings_batch(contents)
    except EmbeddingError as exc:
        logger.error(f"Code example embedding failed: {exc}")
        return 0

    rows = []
    for i, emb in enumerate(embeddings):
        source_str = metadatas[i].get("source", "")
        source_id = upsert_source(session, source_str) if source_str else None
        meta = metadatas[i] if isinstance(metadatas[i], dict) else {}
        content_hash = meta.get("content_hash")
        if not isinstance(content_hash, str) or not content_hash:
            content_hash = hashlib.sha256(contents[i].encode("utf-8")).hexdigest()
        crawl_ts = _parse_iso_datetime(meta.get("crawl_timestamp") or meta.get("crawl_time"))
        rows.append(
            CodeExample(
                source_id=source_id,
                url=urls[i],
                chunk_number=chunk_numbers[i],
                language=languages[i],
                content=contents[i],
                summary=summaries[i],
                content_class=str(meta.get("content_class") or ContentClass.CODE.value),
                is_active=bool(meta.get("is_active", True)),
                crawl_timestamp=crawl_ts,
                content_hash=content_hash,
                ex_metadata=meta,
                embedding=emb,
                first_seen_at=first_seen_code_map.get((urls[i], chunk_numbers[i]), _now_code),
                last_seen_at=_now_code,
                last_crawled_at=crawl_ts,
            )
        )
    try:
        session.add_all(rows)
        session.commit()
        return len(rows)
    except SQLAlchemyError as exc:
        logger.error(f"Code example DB insert failed: {exc}")
        session.rollback()
        return 0


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

    hybrid = settings.USE_HYBRID_SEARCH if use_hybrid is None else use_hybrid

    try:
        query_embedding = await create_embedding(query)
    except EmbeddingError as exc:
        logger.error(f"Failed to embed query: {exc}")
        return []

    # Build filter JSON for the DB function
    filter_json = json.dumps(filter_metadata or {})

    # Use the SQL function for vector search
    try:
        raw = session.exec(
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
        # Fall back to Python-side similarity if DB function not available
        raw = _python_side_vector_search(session, query_embedding, match_count * 2, filter_metadata)

    vector_results = [
        {
            "id": r[0],
            "url": r[1],
            "chunk_number": r[2],
            "content": r[3],
            "page_metadata": r[4] if isinstance(r[4], dict) else {},
            "similarity_score": float(r[5]),
            "fts_score": 0.0,
        }
        for r in raw
    ]

    if not hybrid:
        return vector_results[:match_count]

    # FTS pass
    try:
        fts_raw = session.exec(
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
        fts_raw = []

    fts_map = {
        r[0]: float(r[5]) for r in fts_raw
    }

    # Reciprocal rank fusion
    vector_rank = {r["id"]: i + 1 for i, r in enumerate(vector_results)}
    fts_ids_ordered = [r[0] for r in fts_raw]
    fts_rank = {rid: i + 1 for i, rid in enumerate(fts_ids_ordered)}

    k = 60
    all_ids = set(vector_rank) | set(fts_rank)
    rrf_scores = {
        rid: 1 / (k + vector_rank.get(rid, len(all_ids) + k))
             + 1 / (k + fts_rank.get(rid, len(all_ids) + k))
        for rid in all_ids
    }

    # Merge results
    id_to_result: Dict[int, Dict[str, Any]] = {r["id"]: r for r in vector_results}
    for r in fts_raw:
        if r[0] not in id_to_result:
            id_to_result[r[0]] = {
                "id": r[0],
                "url": r[1],
                "chunk_number": r[2],
                "content": r[3],
                "page_metadata": r[4] if isinstance(r[4], dict) else {},
                "similarity_score": 0.0,
                "fts_score": fts_map.get(r[0], 0.0),
            }

    merged = sorted(id_to_result.values(), key=lambda x: rrf_scores.get(x["id"], 0), reverse=True)
    return merged[:match_count]


def _python_side_vector_search(
    session: Session,
    query_embedding: List[float],
    limit: int,
    filter_metadata: Optional[Dict[str, Any]],
) -> List[Any]:
    """Fallback: compute cosine similarity in Python."""
    q = np.array(query_embedding, dtype=np.float32)
    pages = session.exec(select(CrawledPage)).all()

    # Filter inactive and tombstoned pages (mirrors SQL match function behaviour)
    pages = [p for p in pages if p.is_active and p.tombstoned_at is None]

    if filter_metadata:
        pages = [
            p for p in pages
            if all(
                (p.page_metadata if isinstance(p.page_metadata, dict) else {}).get(k) == v
                for k, v in filter_metadata.items()
            )
        ]

    scored = []
    for p in pages:
        if not p.embedding:
            continue
        arr = np.array(p.embedding, dtype=np.float32)
        norm_q = np.linalg.norm(q)
        norm_a = np.linalg.norm(arr)
        if norm_q == 0 or norm_a == 0:
            sim = 0.0
        else:
            sim = float(np.dot(q, arr) / (norm_q * norm_a))
        scored.append((p.id, p.url, p.chunk_number, p.content, p.page_metadata, sim))

    scored.sort(key=lambda x: x[5], reverse=True)
    return scored[:limit]


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

    filter_dict: Dict[str, Any] = {}
    if language:
        filter_dict["language"] = language
    filter_json = json.dumps(filter_dict)

    try:
        raw = session.exec(
            text(
                "SELECT id, url, chunk_number, language, content, summary, metadata, similarity "
                "FROM match_code_examples(CAST(:emb AS vector), :cnt, CAST(:filt AS jsonb))"
            ).bindparams(emb=str(query_embedding), cnt=match_count, filt=filter_json)
        ).all()
    except Exception:
        raw = []

    return [
        {
            "id": r[0],
            "url": r[1],
            "chunk_number": r[2],
            "language": r[3],
            "content": r[4],
            "summary": r[5],
            "metadata": r[6] if isinstance(r[6], dict) else {},
            "similarity_score": float(r[7]),
        }
        for r in raw
    ]


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
        from sentence_transformers import CrossEncoder  # noqa: PLC0415
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [(query, r["content"]) for r in results]
        scores = model.predict(pairs)
        for i, r in enumerate(results):
            r["rerank_score"] = float(scores[i])
        return sorted(results, key=lambda x: x.get("rerank_score", 0), reverse=True)[:top_k]
    except Exception as exc:
        logger.warning(f"Reranking failed, returning original order: {exc}")
        return results[:top_k]


# ---------------------------------------------------------------------------
# Code example extraction helper
# ---------------------------------------------------------------------------
def extract_code_blocks(markdown: str) -> List[Dict[str, str]]:
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
