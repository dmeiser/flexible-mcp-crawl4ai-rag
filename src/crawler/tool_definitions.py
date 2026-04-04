"""MCP tool definitions — registered in src/crawl4ai_mcp.py."""
import json
import logging
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
from datetime import datetime, timezone

from fastmcp import Context

from src.utils import (
    get_session,
    settings,
    add_documents_to_db,
    add_code_examples_to_db,
    search_documents,
    search_code_examples as _search_code_examples,
    rerank_results,
    extract_code_blocks,
)
from .web_crawler import (
    is_sitemap,
    is_txt,
    parse_sitemap,
    crawl_markdown_file,
    crawl_batch,
    crawl_recursive_internal_links,
    chunk_text_according_to_settings,
)
from .metadata_extractor import extract_section_info
from .postgres_client import store_crawled_documents, fetch_available_sources, execute_rag_query

logger = logging.getLogger(__name__)


def _get_crawler(ctx: Context):
    """Retrieve the AsyncWebCrawler from lifespan context."""
    lc = ctx.lifespan_context
    if lc is None or not hasattr(lc, "crawler"):
        raise RuntimeError("Crawler not initialized — lifespan context missing.")
    return lc.crawler


async def crawl_single_page(ctx: Context, url: str) -> str:
    """
    Crawl a single web page and store its content.

    Args:
        ctx: MCP context.
        url: URL to crawl.
    """
    try:
        crawler = _get_crawler(ctx)
        from crawl4ai import CrawlerRunConfig, CacheMode  # local import to keep top-level clean
        result = await crawler.arun(url=url, config=CrawlerRunConfig(cache_mode=CacheMode.BYPASS))

        if not (result.success and result.markdown):
            return json.dumps({
                "success": False,
                "url": url,
                "error": getattr(result, "error_message", None) or "No content.",
            }, indent=2)

        chunks = await chunk_text_according_to_settings(result.markdown)

        db_urls, db_chunks, db_contents, db_metas, db_fulldocs = [], [], [], [], []
        for i, chunk in enumerate(chunks):
            meta = extract_section_info(chunk)
            meta.update({"chunk_index": i, "url": url, "source": urlparse(url).netloc,
                          "crawl_time": datetime.now(timezone.utc).isoformat()})
            db_urls.append(url)
            db_chunks.append(i)
            db_contents.append(chunk)
            db_metas.append(meta)
            db_fulldocs.append(result.markdown)

        with next(get_session()) as session:
            added = await add_documents_to_db(session, db_urls, db_contents, db_metas, db_chunks, db_fulldocs)

        # Store code examples if agentic RAG enabled
        if settings.USE_AGENTIC_RAG:
            code_blocks = extract_code_blocks(result.markdown)
            if code_blocks:
                with next(get_session()) as session:
                    await add_code_examples_to_db(
                        session,
                        urls=[url] * len(code_blocks),
                        contents=[b["content"] for b in code_blocks],
                        languages=[b["language"] for b in code_blocks],
                        summaries=[None] * len(code_blocks),
                        metadatas=[{"source": urlparse(url).netloc, "url": url}] * len(code_blocks),
                        chunk_numbers=list(range(len(code_blocks))),
                    )

        return json.dumps({
            "success": True,
            "url": url,
            "chunks_stored": added,
            "content_length": len(result.markdown),
            "links_count": {
                "internal": len((result.links or {}).get("internal", [])),
                "external": len((result.links or {}).get("external", [])),
            },
        }, indent=2)

    except Exception as exc:
        logger.error(f"crawl_single_page {url}: {exc}", exc_info=True)
        return json.dumps({"success": False, "url": url, "error": str(exc)}, indent=2)


async def smart_crawl_url(
    ctx: Context,
    url: str,
    max_depth: int = 3,
    max_concurrent: int = 10,
    chunk_size: int = 5000,
    follow_links: bool = False,
    url_pattern: Optional[str] = None,
) -> str:
    """
    Intelligently crawl a URL based on its type.

    Detects sitemap, text file, or webpage and applies the correct crawl strategy.

    Args:
        ctx: MCP context.
        url: URL to crawl.
        max_depth: Max recursion depth when follow_links is True.
        max_concurrent: Max concurrent browser sessions.
        chunk_size: Max characters per chunk (uses settings default).
        follow_links: Follow internal links recursively.
        url_pattern: Regex to filter URLs when follow_links is True.
    """
    try:
        crawler = _get_crawler(ctx)
        crawl_results: List[Dict[str, Any]] = []
        crawl_type = "webpage"

        if is_txt(url):
            crawl_results = await crawl_markdown_file(crawler, url)
            crawl_type = "text_file"
        elif is_sitemap(url):
            sitemap_urls = await parse_sitemap(url)
            if not sitemap_urls:
                return json.dumps({"success": False, "url": url, "error": "No URLs in sitemap."}, indent=2)
            crawl_results = await crawl_batch(crawler, sitemap_urls, max_concurrent=max_concurrent)
            crawl_type = "sitemap"
        elif follow_links:
            crawl_results = await crawl_recursive_internal_links(
                crawler, [url], max_depth=max_depth, max_concurrent=max_concurrent, url_pattern=url_pattern
            )
            crawl_type = "webpage_recursive"
        else:
            crawl_results = await crawl_markdown_file(crawler, url)
            crawl_type = "webpage_single"

        if not crawl_results:
            return json.dumps({"success": False, "url": url, "error": "No content found."}, indent=2)

        with next(get_session()) as session:
            pages_processed, chunks_stored = await store_crawled_documents(session, crawl_results, crawl_type)

        return json.dumps({
            "success": True,
            "url": url,
            "crawl_type": crawl_type,
            "pages_crawled": pages_processed,
            "chunks_stored": chunks_stored,
            "urls_crawled_sample": [d["url"] for d in crawl_results[:5]]
                + (["..."] if len(crawl_results) > 5 else []),
        }, indent=2)

    except Exception as exc:
        logger.error(f"smart_crawl_url {url}: {exc}", exc_info=True)
        return json.dumps({"success": False, "url": url, "error": str(exc)}, indent=2)


async def get_available_sources(ctx: Context) -> str:
    """
    List all crawled sources (domains) available in the database.

    Args:
        ctx: MCP context.
    """
    try:
        with next(get_session()) as session:
            sources = fetch_available_sources(session)
        return json.dumps({"success": True, "sources": sources, "count": len(sources)}, indent=2)
    except Exception as exc:
        logger.error(f"get_available_sources: {exc}", exc_info=True)
        return json.dumps({"success": False, "error": str(exc)}, indent=2)


async def perform_rag_query(
    ctx: Context,
    query: str,
    source: Optional[str] = None,
    match_count: int = 5,
) -> str:
    """
    Perform a RAG query over stored documents.

    Args:
        ctx: MCP context.
        query: Natural language search query.
        source: Optional domain to restrict search (e.g. "docs.example.com").
        match_count: Number of results to return.
    """
    try:
        filter_meta = {"source": source} if source else None
        with next(get_session()) as session:
            results = await search_documents(
                session, query, match_count=match_count, filter_metadata=filter_meta
            )

        if settings.USE_RERANKING and results:
            results = rerank_results(query, results, top_k=match_count)

        formatted = [
            {
                "url": r.get("url"),
                "content": r.get("content"),
                "metadata": r.get("page_metadata"),
                "similarity": r.get("similarity_score"),
            }
            for r in results
        ]
        return json.dumps({"success": True, "query": query, "results": formatted}, indent=2)

    except Exception as exc:
        logger.error(f"perform_rag_query: {exc}", exc_info=True)
        return json.dumps({"success": False, "query": query, "error": str(exc)}, indent=2)


async def search_code_examples(
    ctx: Context,
    query: str,
    language: Optional[str] = None,
    match_count: int = 5,
) -> str:
    """
    Search stored code examples by semantic similarity.
    Only available when USE_AGENTIC_RAG=true.

    Args:
        ctx: MCP context.
        query: Natural language description of the code you're looking for.
        language: Optional programming language filter (e.g. "python").
        match_count: Number of results to return.
    """
    try:
        with next(get_session()) as session:
            results = await _search_code_examples(session, query, match_count=match_count, language=language)

        return json.dumps({
            "success": True,
            "query": query,
            "results": results,
        }, indent=2)
    except Exception as exc:
        logger.error(f"search_code_examples: {exc}", exc_info=True)
        return json.dumps({"success": False, "query": query, "error": str(exc)}, indent=2)
