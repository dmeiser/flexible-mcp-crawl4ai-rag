from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from sqlmodel import Session, select

from src.providers.openai_stack import OpenAIConfiguration
from src.providers.openrouter_web_search import web_search_model


class WebSearchService:
    """Service facade for web search execution and short-TTL cache persistence."""

    @staticmethod
    async def execute_web_search(
        *,
        settings: Any,
        endpoint_factory: Callable[[Optional[str], Optional[str]], Any],
        query: str,
        engine: str = "auto",
        max_results: int = 5,
        allowed_domains: Optional[List[str]] = None,
        excluded_domains: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        configuration = OpenAIConfiguration(
            api_key=settings.WEB_SEARCH_API_KEY,
            base_url=settings.effective_web_search_base_url,
            max_retries=settings.effective_web_search_max_retries,
            retry_delay_seconds=settings.effective_web_search_retry_delay_seconds,
        )
        model = web_search_model(
            provider=settings.effective_web_search_provider,
            configuration=configuration,
            model_name=settings.effective_web_search_model_name,
            endpoint_factory=endpoint_factory,
            default_engine=settings.WEB_SEARCH_DEFAULT_ENGINE,
            default_max_results=settings.WEB_SEARCH_DEFAULT_MAX_RESULTS,
        )
        resolved_engine = (engine or settings.WEB_SEARCH_DEFAULT_ENGINE).strip() or settings.WEB_SEARCH_DEFAULT_ENGINE
        resolved_max_results = max(1, int(max_results or settings.WEB_SEARCH_DEFAULT_MAX_RESULTS))
        return await model.search(
            query=query,
            engine=resolved_engine,
            max_results=resolved_max_results,
            allowed_domains=allowed_domains,
            excluded_domains=excluded_domains,
        )

    @staticmethod
    async def cache_web_search_results(
        *,
        session: Session,
        result: Dict[str, Any],
        settings: Any,
        upsert_source_fn: Callable[[Session, str], int],
        crawled_page_cls: Any,
        content_class_text: str,
        embedding_dim: int,
    ) -> int:
        """Persist ephemeral web-search leads as inactive, short-TTL crawled rows."""
        sources = _web_search_sources(result)
        if not sources:
            return 0

        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(hours=max(1, int(settings.WEB_SEARCH_CACHE_TTL_HOURS)))
        source_name = str(settings.WEB_SEARCH_CACHE_SOURCE or "openrouter_web_search")
        source_id = upsert_source_fn(session, source_name)

        _prune_expired_web_search_cache(session, crawled_page_cls, source_name, now)

        rows = _build_web_search_cache_rows(
            result=result,
            sources=sources,
            source_id=source_id,
            source_name=source_name,
            now=now,
            expires_at=expires_at,
            crawled_page_cls=crawled_page_cls,
            content_class_text=content_class_text,
            embedding_dim=embedding_dim,
        )
        return _commit_web_search_cache_rows(session, rows)


def _web_search_sources(result: Dict[str, Any]) -> List[Any]:
    sources = result.get("sources")
    return sources if isinstance(sources, list) else []


def _commit_web_search_cache_rows(session: Session, rows: List[Any]) -> int:
    if not rows:
        return 0
    session.add_all(rows)
    session.commit()
    return len(rows)


def _prune_expired_web_search_cache(
    session: Session,
    crawled_page_cls: Any,
    cache_source: str,
    now: datetime,
) -> None:
    rows = session.exec(select(crawled_page_cls)).all()
    expired = [row for row in rows if _is_expired_web_cache_row(row, cache_source, now)]
    _delete_web_search_cache_rows(session, expired)


def _delete_web_search_cache_rows(session: Session, rows: List[Any]) -> None:
    for row in rows:
        session.delete(row)
    if rows:
        session.commit()


def _is_expired_web_cache_row(row: Any, cache_source: str, now: datetime) -> bool:
    return (not row.is_active) and _row_expires_before(row, now) and _row_cache_source(row) == cache_source


def _row_expires_before(row: Any, now: datetime) -> bool:
    return row.expires_at is not None and row.expires_at < now


def _row_cache_source(row: Any) -> str:
    metadata = row.page_metadata if isinstance(row.page_metadata, dict) else {}
    return str(metadata.get("source") or "")


def _build_web_search_cache_rows(
    *,
    result: Dict[str, Any],
    sources: List[Any],
    source_id: int,
    source_name: str,
    now: datetime,
    expires_at: datetime,
    crawled_page_cls: Any,
    content_class_text: str,
    embedding_dim: int,
) -> List[Any]:
    rows: List[Any] = []
    for index, item in enumerate(sources):
        row = _build_single_web_search_cache_row(
            result=result,
            item=item,
            source_id=source_id,
            source_name=source_name,
            now=now,
            expires_at=expires_at,
            index=index,
            crawled_page_cls=crawled_page_cls,
            content_class_text=content_class_text,
            embedding_dim=embedding_dim,
        )
        if row is not None:
            rows.append(row)
    return rows


def _build_single_web_search_cache_row(
    *,
    result: Dict[str, Any],
    item: Any,
    source_id: int,
    source_name: str,
    now: datetime,
    expires_at: datetime,
    index: int,
    crawled_page_cls: Any,
    content_class_text: str,
    embedding_dim: int,
) -> Optional[Any]:
    normalized_fields = _normalized_web_search_cache_fields(item)
    if normalized_fields is None:
        return None

    url, title, snippet = normalized_fields
    if not url:
        return None
    content = _web_search_cache_content(result, url, title, snippet)
    metadata = _web_search_cache_page_metadata(result, source_name, url, title, snippet, now, expires_at)
    retrieval_metadata = _web_search_cache_retrieval_metadata(source_name, url, now)

    return crawled_page_cls(
        source_id=source_id,
        url=url,
        chunk_number=index,
        content=content,
        content_class=content_class_text,
        is_active=False,
        crawl_timestamp=now,
        content_hash=hashlib.sha256(f"{result.get('query', '')}::{url}::{title}".encode("utf-8")).hexdigest(),
        source_change_id=None,
        first_seen_at=now,
        last_seen_at=now,
        last_crawled_at=now,
        expires_at=expires_at,
        is_pinned=False,
        tombstoned_at=None,
        staleness_score=0.0,
        value_score=0.0,
        hit_count=0,
        page_metadata=metadata,
        retrieval_metadata=retrieval_metadata,
        embedding=[0.0] * int(embedding_dim),
    )


def _as_web_search_source_item(item: Any) -> Optional[Dict[str, Any]]:
    return item if isinstance(item, dict) else None


def _normalized_web_search_cache_fields(item: Any) -> Optional[Tuple[str, str, str]]:
    source_item = _as_web_search_source_item(item)
    if source_item is None:
        return None
    return (
        str(source_item.get("url") or "").strip(),
        str(source_item.get("title") or "").strip(),
        str(source_item.get("snippet") or "").strip(),
    )


def _web_search_cache_content(result: Dict[str, Any], url: str, title: str, snippet: str) -> str:
    return snippet or title or str(result.get("answer") or "") or url


def _web_search_cache_page_metadata(
    result: Dict[str, Any],
    source_name: str,
    url: str,
    title: str,
    snippet: str,
    now: datetime,
    expires_at: datetime,
) -> Dict[str, Any]:
    return {
        "source": source_name,
        "url": url,
        "source_type": "web_search_cache",
        "title": title,
        "snippet": snippet,
        "query": str(result.get("query") or ""),
        "cached_at": now.isoformat(),
        "expires_at": expires_at.isoformat(),
    }


def _web_search_cache_retrieval_metadata(source_name: str, url: str, now: datetime) -> Dict[str, Any]:
    return {
        "source": source_name,
        "url": url,
        "source_type": "web_search_cache",
        "crawl_timestamp": now.isoformat(),
    }


# Backward-compatible function exports
execute_web_search = WebSearchService.execute_web_search
cache_web_search_results = WebSearchService.cache_web_search_results
