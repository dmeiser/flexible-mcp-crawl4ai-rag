"""PostgreSQL client — document storage and retrieval operations."""
import logging
import hashlib
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
from datetime import datetime, timezone

from sqlmodel import Session

from ..utils import (
    add_documents_to_db,
    search_documents,
    CrawledPage,
    Source,
    settings,
    get_session,
)
from .metadata_extractor import extract_section_info
from .web_crawler import chunk_text_according_to_settings

logger = logging.getLogger(__name__)


async def store_crawled_documents(
    session: Session,
    crawl_results: List[Dict[str, Any]],
    crawl_type: str,
) -> tuple:
    """
    Chunk and embed crawl results, then persist to crawled_pages.

    Returns:
        (pages_processed, total_chunks_stored)
    """
    all_urls: List[str] = []
    all_contents: List[str] = []
    all_metadatas: List[Dict[str, Any]] = []
    all_chunk_numbers: List[int] = []
    all_full_docs: List[str] = []

    for doc in crawl_results:
        source_url = doc["url"]
        markdown = doc.get("markdown", "")
        if not markdown:
            logger.warning(f"No markdown for {source_url}, skipping.")
            continue

        chunks = await chunk_text_according_to_settings(markdown)
        for i, chunk in enumerate(chunks):
            meta = extract_section_info(chunk)
            meta.update({
                "chunk_index": i,
                "url": source_url,
                "source": urlparse(source_url).netloc,
                "crawl_type": crawl_type,
                "crawl_time": datetime.now(timezone.utc).isoformat(),
                "crawl_timestamp": datetime.now(timezone.utc).isoformat(),
                "content_class": "text",
                "is_active": True,
                "content_hash": hashlib.sha256(chunk.encode("utf-8")).hexdigest(),
            })
            all_urls.append(source_url)
            all_contents.append(chunk)
            all_metadatas.append(meta)
            all_chunk_numbers.append(i)
            all_full_docs.append(markdown)

    if all_contents:
        await add_documents_to_db(
            session, all_urls, all_contents, all_metadatas, all_chunk_numbers, all_full_docs
        )

    return len(crawl_results), len(all_contents)


def fetch_available_sources(session: Session) -> List[str]:
    """Return sorted list of unique source domains from the sources table."""
    from sqlmodel import select
    results = session.exec(select(Source.source)).all()
    return sorted({s for s in results if s})


def execute_rag_query(
    session: Session,
    query: str,
    source: Optional[str] = None,
    match_count: int = 5,
    filter_metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Synchronous RAG query wrapper (kept for backward compat; prefer async search_documents)."""
    import asyncio
    combined: Dict[str, Any] = {}
    if source:
        combined["source"] = source
    if filter_metadata:
        combined.update(filter_metadata)

    results = asyncio.get_event_loop().run_until_complete(
        search_documents(session, query, match_count=match_count, filter_metadata=combined or None)
    )
    return [
        {"url": r.get("url"), "content": r.get("content"),
         "metadata": r.get("page_metadata"), "similarity": r.get("similarity_score")}
        for r in results
    ]
