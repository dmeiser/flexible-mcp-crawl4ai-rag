"""PostgreSQL client — document storage and retrieval operations."""

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List
from urllib.parse import urlparse

from sqlmodel import Session

from ..utils import add_documents_to_db, extract_link_references
from .metadata_extractor import extract_section_info
from .web_crawler import chunk_text_according_to_settings

logger = logging.getLogger(__name__)


def _extract_variant_values(doc: Dict[str, Any]) -> Dict[str, Any]:
    raw_variant_values = doc.get("variant_values")
    return raw_variant_values if isinstance(raw_variant_values, dict) else {}


def _extract_extra_metadata(doc: Dict[str, Any]) -> Dict[str, Any]:
    extra_metadata: Dict[str, Any] = {}
    for key in ("source_change_id", "link_graph", "media_metadata", "session_id", "run_id"):
        if key in doc:
            extra_metadata[key] = doc.get(key)
    return extra_metadata


def _base_chunk_metadata(
    source_url: str,
    crawl_type: str,
    chunk_index: int,
    chunk: str,
    selected_variant: Any,
    references_markdown: str,
    link_references: Any,
    variant_values: Dict[str, Any],
) -> Dict[str, Any]:
    metadata = extract_section_info(chunk)
    metadata.update(
        {
            "chunk_index": chunk_index,
            "url": source_url,
            "source": urlparse(source_url).netloc,
            "crawl_type": crawl_type,
            "crawl_time": datetime.now(timezone.utc).isoformat(),
            "crawl_timestamp": datetime.now(timezone.utc).isoformat(),
            "content_class": "text",
            "is_active": True,
            "content_hash": hashlib.sha256(chunk.encode("utf-8")).hexdigest(),
            "markdown_variant": selected_variant,
            "references_markdown": references_markdown,
            "link_references": link_references,
            "has_citations": bool(variant_values.get("markdown_with_citations")),
        }
    )
    return metadata


def _append_chunk_records(
    urls: List[str],
    contents: List[str],
    metadatas: List[Dict[str, Any]],
    chunk_numbers: List[int],
    full_docs: List[str],
    source_url: str,
    markdown: str,
    chunk: str,
    chunk_index: int,
    metadata: Dict[str, Any],
) -> None:
    urls.append(source_url)
    contents.append(chunk)
    metadatas.append(metadata)
    chunk_numbers.append(chunk_index)
    full_docs.append(markdown)


def _doc_chunk_metadata(
    source_url: str,
    crawl_type: str,
    chunk_index: int,
    chunk: str,
    selected_variant: Any,
    references_markdown: str,
    link_references: Any,
    variant_values: Dict[str, Any],
    extra_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    metadata = _base_chunk_metadata(
        source_url,
        crawl_type,
        chunk_index,
        chunk,
        selected_variant,
        references_markdown,
        link_references,
        variant_values,
    )
    metadata.update(extra_metadata)
    return metadata


async def _collect_doc_chunks(
    crawl_type: str,
    doc: Dict[str, Any],
    urls: List[str],
    contents: List[str],
    metadatas: List[Dict[str, Any]],
    chunk_numbers: List[int],
    full_docs: List[str],
) -> None:
    source_url = doc["url"]
    markdown = doc.get("markdown", "")
    if not markdown:
        logger.warning(f"No markdown for {source_url}, skipping.")
        return

    variant_values = _extract_variant_values(doc)
    references_markdown = variant_values.get("references_markdown") or ""
    selected_variant = doc.get("selected_variant") or doc.get("markdown_variant")
    link_references = extract_link_references(references_markdown)
    extra_metadata = _extract_extra_metadata(doc)
    chunks = await chunk_text_according_to_settings(markdown)

    for chunk_index, chunk in enumerate(chunks):
        metadata = _doc_chunk_metadata(
            source_url,
            crawl_type,
            chunk_index,
            chunk,
            selected_variant,
            references_markdown,
            link_references,
            variant_values,
            extra_metadata,
        )
        _append_chunk_records(
            urls,
            contents,
            metadatas,
            chunk_numbers,
            full_docs,
            source_url,
            markdown,
            chunk,
            chunk_index,
            metadata,
        )


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
        await _collect_doc_chunks(
            crawl_type,
            doc,
            all_urls,
            all_contents,
            all_metadatas,
            all_chunk_numbers,
            all_full_docs,
        )

    if all_contents:
        await add_documents_to_db(session, all_urls, all_contents, all_metadatas, all_chunk_numbers, all_full_docs)

    return len(crawl_results), len(all_contents)
