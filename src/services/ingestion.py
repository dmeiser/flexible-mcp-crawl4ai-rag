"""PostgreSQL client — document storage and retrieval operations."""

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from sqlmodel import Session

from src.services.content_extraction import extract_link_references
from src.services.document_storage_service import add_documents_to_db

from .metadata_extractor import extract_section_info
from .web_crawler import chunk_text_with_heading_metadata

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
    heading_path: Optional[List[str]] = None,
    heading_level: int = 0,
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
            "heading_path": heading_path if heading_path is not None else [],
            "heading_level": heading_level,
        }
    )
    return metadata


def _append_chunk_records(
    urls: List[str],
    contents: List[str],
    embed_texts: List[str],
    metadatas: List[Dict[str, Any]],
    chunk_numbers: List[int],
    full_docs: List[str],
    source_url: str,
    markdown: str,
    chunk: str,
    embed_text: str,
    chunk_index: int,
    metadata: Dict[str, Any],
) -> None:
    urls.append(source_url)
    contents.append(chunk)
    embed_texts.append(embed_text)
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
    heading_path: Optional[List[str]] = None,
    heading_level: int = 0,
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
        heading_path=heading_path,
        heading_level=heading_level,
    )
    metadata.update(extra_metadata)
    return metadata


def _variant_doc_fields(doc: Dict[str, Any]) -> Tuple[Dict[str, Any], str, Any]:
    """Extract variant metadata fields from *doc*, handling ``or`` defaults inline."""
    variant_values = _extract_variant_values(doc)
    references_markdown = variant_values.get("references_markdown") or ""
    selected_variant = doc.get("selected_variant") or doc.get("markdown_variant")
    return variant_values, references_markdown, selected_variant


def _embed_text_for_chunk(chunk: str, heading_path: List[str]) -> str:
    """Prepend the heading breadcrumb to *chunk* when a heading path is present."""
    if heading_path:
        return f"[{' > '.join(heading_path)}]\n\n{chunk}"
    return chunk


async def _collect_doc_chunks(
    crawl_type: str,
    doc: Dict[str, Any],
    urls: List[str],
    contents: List[str],
    embed_texts: List[str],
    metadatas: List[Dict[str, Any]],
    chunk_numbers: List[int],
    full_docs: List[str],
) -> None:
    source_url = doc["url"]
    markdown = doc.get("markdown", "")
    if not markdown:
        logger.warning(f"No markdown for {source_url}, skipping.")
        return

    variant_values, references_markdown, selected_variant = _variant_doc_fields(doc)
    link_references = extract_link_references(references_markdown)
    extra_metadata = _extract_extra_metadata(doc)
    chunk_pairs = await chunk_text_with_heading_metadata(markdown)

    for chunk_index, (chunk, chunk_meta) in enumerate(chunk_pairs):
        heading_path = chunk_meta.get("heading_path", [])
        heading_level = chunk_meta.get("heading_level", 0)
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
            heading_path=heading_path,
            heading_level=heading_level,
        )
        embed_text = _embed_text_for_chunk(chunk, heading_path)
        _append_chunk_records(
            urls,
            contents,
            embed_texts,
            metadatas,
            chunk_numbers,
            full_docs,
            source_url,
            markdown,
            chunk,
            embed_text,
            chunk_index,
            metadata,
        )


async def store_crawled_documents(
    session: Session,
    crawl_results: List[Dict[str, Any]],
    crawl_type: str,
    endpoint_factory: Optional[Callable[..., Any]] = None,
) -> tuple:
    """
    Chunk and embed crawl results, then persist to crawled_pages.

    Returns:
        (pages_processed, total_chunks_stored)
    """
    all_urls: List[str] = []
    all_contents: List[str] = []
    all_embed_texts: List[str] = []
    all_metadatas: List[Dict[str, Any]] = []
    all_chunk_numbers: List[int] = []
    all_full_docs: List[str] = []

    for doc in crawl_results:
        await _collect_doc_chunks(
            crawl_type,
            doc,
            all_urls,
            all_contents,
            all_embed_texts,
            all_metadatas,
            all_chunk_numbers,
            all_full_docs,
        )

    if all_contents:
        await add_documents_to_db(
            session,
            all_urls,
            all_contents,
            all_metadatas,
            all_chunk_numbers,
            all_full_docs,
            embed_texts=all_embed_texts,
        )
        await index_knowledge_graphs(session, all_urls, all_contents, endpoint_factory)

    return len(crawl_results), len(all_contents)


async def index_knowledge_graphs(
    session: Session,
    urls: List[str],
    contents: List[str],
    endpoint_factory: Optional[Callable[..., Any]] = None,
) -> None:
    from src.config import settings
    from src.services.graph_storage_service import store_knowledge_graph
    from src.services.kg_extraction_service import KnowledgeGraphExtractionService
    from src.utils import create_embedding

    logger.debug(
        "index_knowledge_graphs called: USE_GRAPH_INDEX=%s model=%s urls=%d",
        settings.USE_GRAPH_INDEX,
        settings.effective_kg_model_name,
        len(urls),
    )

    if not (settings.USE_GRAPH_INDEX and settings.effective_kg_model_name):
        return

    def _default_factory(**kwargs: Any) -> Any:
        from openai import AsyncOpenAI

        from src.services.kg_extraction_service import OpenAIEndpointAdapter

        client = AsyncOpenAI(
            api_key=settings.effective_kg_api_key,
            base_url=settings.effective_kg_base_url,
        )
        return OpenAIEndpointAdapter(client)

    factory = endpoint_factory if endpoint_factory is not None else _default_factory
    extractor = KnowledgeGraphExtractionService(endpoint_factory=factory, logger=logger)
    for url, content in zip(urls, contents):
        logger.info("Extracting KG for %s (content len=%d)", url, len(content))
        kg_data = await extractor.extract_knowledge_graph(settings, content, url)
        logger.info(
            "KG extracted for %s: %d entities, %d relationships",
            url,
            len(kg_data.get("entities", [])),
            len(kg_data.get("relationships", [])),
        )
        await store_knowledge_graph(session, kg_data, url, None, create_embedding)
