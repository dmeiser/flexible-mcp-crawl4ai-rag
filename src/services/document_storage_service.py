from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Session, select

import src.utils as _utils
from src.config import ContentClass
from src.exceptions import EmbeddingError
from src.models import CodeExample, CrawledPage, Source
from src.services.content_extraction import extract_link_references

logger = logging.getLogger(__name__)


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


def _parse_iso_datetime(value: Any) -> datetime:
    """Parse timestamp-like metadata values into timezone-aware UTC datetimes."""
    if isinstance(value, datetime):
        return _ensure_utc_datetime(value)
    if isinstance(value, str) and value.strip():
        parsed = _parse_iso_string(value.strip())
        if parsed is not None:
            return parsed
    return datetime.now(timezone.utc)


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
    enriched = await asyncio.gather(
        *[_utils.generate_contextual_text(fd or "", c) for c, fd in zip(contents, full_docs)]
    )
    return [entry[0] for entry in enriched]


def _should_enrich_with_context(full_documents: Optional[List[str]]) -> bool:
    return bool(
        _utils.settings.USE_CONTEXTUAL_EMBEDDINGS and _utils.settings.effective_contextual_model_name and full_documents
    )


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
    batch_size = _utils.settings.BATCH_SIZE
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
        return await _utils.create_embeddings_batch(texts)
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
        embeddings = await _utils.create_embeddings_batch(contents)
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
