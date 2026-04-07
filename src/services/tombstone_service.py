"""
Tombstoning (soft-delete) and DB size utilities.
"""

import logging
from datetime import datetime, timezone
from typing import Any, List, Optional, cast

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Session, select

from src.models import CodeExample, CrawledPage, EvictionAuditLog

logger = logging.getLogger(__name__)


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


def get_db_size_bytes(session: Session) -> int:
    """Return current DB size in bytes via pg_database_size."""
    try:
        row = session.exec(text("SELECT pg_database_size(current_database())")).first()  # type: ignore[call-overload]
        return int(row[0]) if row else 0
    except Exception:
        return 0
