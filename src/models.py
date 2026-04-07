"""
ORM models and database engine for the Crawl4AI MCP server.
PostgreSQL/pgvector backend.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Column, Field, Session, SQLModel, create_engine

from src.config import ContentClass, settings

# ---------------------------------------------------------------------------
# Database engine
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
