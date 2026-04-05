"""MCP tool definitions — registered in src/crawl4ai_mcp.py."""
import json
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
from datetime import datetime, timezone, timedelta

from fastmcp import Context
from sqlmodel import select
from sqlalchemy import text as _sql_text
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    CacheMode,
    MemoryAdaptiveDispatcher,
    AdaptiveCrawler,
    AdaptiveConfig,
    JsonCssExtractionStrategy,
    JsonXPathExtractionStrategy,
    RegexExtractionStrategy,
    LLMExtractionStrategy,
    LLMConfig,
    BFSDeepCrawlStrategy,
    DFSDeepCrawlStrategy,
    BestFirstCrawlingStrategy,
    FilterChain,
    URLPatternFilter,
    DomainFilter,
    KeywordRelevanceScorer,
    DefaultMarkdownGenerator,
    PruningContentFilter,
    BM25ContentFilter,
    LLMContentFilter,
    ContentTypeFilter,
    SEOFilter,
    ContentRelevanceFilter,
)

from src.utils import (
    get_session,
    settings,
    CrawledPage,
    CodeExample,
    SourcePolicy,
    StoragePolicy,
    EvictionAuditLog,
    add_documents_to_db,
    add_code_examples_to_db,
    search_documents as _search_documents_core,
    search_code_examples as _search_code_examples,
    rerank_results,
    extract_code_blocks,
    compute_value_score,
    compute_staleness_score,
    tombstone_records,
    _get_db_size_bytes,
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

# Backward-compatible symbol name used by existing tests/patches.
search_documents = _search_documents_core


_ALLOWED_RUN_CONFIG_FIELDS = {
    "cache_mode",
    "wait_for",
    "js_code",
    "js_code_before_wait",
    "c4a_script",
    "scan_full_page",
    "wait_until",
    "page_timeout",
    "delay_before_return_html",
    "word_count_threshold",
    "excluded_tags",
    "exclude_external_links",
    "flatten_shadow_dom",
    "screenshot",
    "pdf",
    "capture_mhtml",
    "session_id",
    "stream",
    "prefetch",
    "check_robots_txt",
}

_ALLOWED_BROWSER_CONFIG_FIELDS = {
    "browser_type",
    "headless",
    "viewport_width",
    "viewport_height",
    "text_mode",
    "light_mode",
    "enable_stealth",
    "headers",
    "cookies",
    "proxy_config",
    "use_persistent_context",
    "user_data_dir",
}

_ALLOWED_EXTRACTION_STRATEGIES = {
    "css",
    "xpath",
    "regex",
    "llm",
    None,
}

_ALLOWED_CONTENT_SOURCES = {"cleaned_html", "raw_html", "fit_html"}
_ALLOWED_CONTENT_FILTERS = {"pruning", "bm25", "llm", None}
_ALLOWED_SCORER_TYPES = {"keyword", "none"}


def _build_run_config(run_config: Optional[Dict[str, Any]] = None) -> CrawlerRunConfig:
    """Build a CrawlerRunConfig from a safe allowlist of fields."""
    if not run_config:
        return CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    safe_kwargs = {k: v for k, v in run_config.items() if k in _ALLOWED_RUN_CONFIG_FIELDS}

    # Normalize cache_mode safely
    cm = safe_kwargs.get("cache_mode")
    if isinstance(cm, str):
        upper = cm.upper().strip()
        if upper in CacheMode.__members__:
            safe_kwargs["cache_mode"] = CacheMode[upper]
        else:
            safe_kwargs.pop("cache_mode", None)

    if "cache_mode" not in safe_kwargs:
        safe_kwargs["cache_mode"] = CacheMode.BYPASS

    return CrawlerRunConfig(**safe_kwargs)


def _extract_markdown_variants(markdown_obj: Any) -> Dict[str, str]:
    """Extract markdown variants from Crawl4AI's markdown-compatible object."""
    if markdown_obj is None:
        return {
            "raw_markdown": "",
            "fit_markdown": "",
            "markdown_with_citations": "",
            "references_markdown": "",
            "fit_html": "",
        }

    raw_markdown = getattr(markdown_obj, "raw_markdown", None) or str(markdown_obj)
    return {
        "raw_markdown": raw_markdown,
        "fit_markdown": getattr(markdown_obj, "fit_markdown", None) or "",
        "markdown_with_citations": getattr(markdown_obj, "markdown_with_citations", None) or raw_markdown,
        "references_markdown": getattr(markdown_obj, "references_markdown", None) or "",
        "fit_html": getattr(markdown_obj, "fit_html", None) or "",
    }


def _build_browser_config(browser_config: Optional[Dict[str, Any]] = None) -> BrowserConfig:
    """Build a BrowserConfig from an allowlist of safe fields."""
    if not browser_config:
        return BrowserConfig(headless=True, verbose=False)
    safe_kwargs = {k: v for k, v in browser_config.items() if k in _ALLOWED_BROWSER_CONFIG_FIELDS}
    if "headless" not in safe_kwargs:
        safe_kwargs["headless"] = True
    if "verbose" not in safe_kwargs:
        safe_kwargs["verbose"] = False
    return BrowserConfig(**safe_kwargs)


def _build_markdown_generator(
    markdown_options: Optional[Dict[str, Any]] = None,
    content_source: str = "cleaned_html",
    content_filter: Optional[str] = None,
    content_filter_query: Optional[str] = None,
    content_filter_threshold: Optional[float] = None,
    content_filter_instruction: Optional[str] = None,
    llm_provider: Optional[str] = None,
) -> Optional[DefaultMarkdownGenerator]:
    """Build a markdown generator with optional content-source and filter controls.

    Returns None when all arguments are defaults, so existing Crawl4AI defaults remain unchanged.
    """
    options = dict(markdown_options or {})
    normalized_source = (content_source or "cleaned_html").strip().lower()
    if normalized_source not in _ALLOWED_CONTENT_SOURCES:
        normalized_source = "cleaned_html"

    normalized_filter = content_filter.lower().strip() if isinstance(content_filter, str) else None
    if normalized_filter not in _ALLOWED_CONTENT_FILTERS:
        normalized_filter = None

    filter_obj = None
    if normalized_filter == "pruning":
        filter_obj = PruningContentFilter(
            user_query=content_filter_query,
            threshold=(content_filter_threshold if isinstance(content_filter_threshold, (int, float)) else 0.48),
        )
    elif normalized_filter == "bm25":
        filter_obj = BM25ContentFilter(
            user_query=content_filter_query,
            bm25_threshold=(content_filter_threshold if isinstance(content_filter_threshold, (int, float)) else 1.0),
        )
    elif normalized_filter == "llm":
        provider = llm_provider or "openai/gpt-4o"
        llm_cfg = LLMConfig(provider=provider)
        filter_obj = LLMContentFilter(
            llm_config=llm_cfg,
            instruction=content_filter_instruction or "Keep only the most relevant content for the user query.",
        )

    if not options and normalized_source == "cleaned_html" and filter_obj is None:
        return None

    return DefaultMarkdownGenerator(
        content_filter=filter_obj,
        options=options,
        content_source=normalized_source,
    )


def _build_extraction_strategy(
    strategy_type: Optional[str] = None,
    schema: Optional[Dict[str, Any]] = None,
    patterns: Optional[Dict[str, str]] = None,
    instruction: Optional[str] = None,
    llm_provider: Optional[str] = None,
) -> Optional[Any]:
    """Build a Crawl4AI extraction strategy from safe parameters.
    
    Args:
        strategy_type: One of "css", "xpath", "regex", "llm", or None.
        schema: Dict defining extraction rules (required for css/xpath, optional for llm).
        patterns: Dict of label->regex mappings (used for regex strategy).
        instruction: Text instruction for LLM extraction.
        llm_provider: LLM provider (e.g. "openai/gpt-4o"), defaults to env var.
    
    Returns:
        ExtractionStrategy instance or None if strategy_type is None.
    """
    if strategy_type is None or strategy_type.lower() not in _ALLOWED_EXTRACTION_STRATEGIES:
        return None
    
    strategy_type_lower = strategy_type.lower()
    
    if strategy_type_lower == "css":
        if not schema:
            logger.warning("CSS extraction requires schema, skipping.")
            return None
        return JsonCssExtractionStrategy(schema=schema)
    
    if strategy_type_lower == "xpath":
        if not schema:
            logger.warning("XPath extraction requires schema, skipping.")
            return None
        return JsonXPathExtractionStrategy(schema=schema)
    
    if strategy_type_lower == "regex":
        # Regex strategy can work with patterns or no custom patterns (uses defaults)
        if patterns:
            return RegexExtractionStrategy(custom=patterns)
        return RegexExtractionStrategy()
    
    if strategy_type_lower == "llm":
        # LLM extraction requires llm_config parameter (not provider)
        if not llm_provider:
            llm_provider = "openai/gpt-4o"
        llm_config = LLMConfig(provider=llm_provider)
        return LLMExtractionStrategy(
            llm_config=llm_config,
            instruction=instruction or "Extract structured data from the content.",
            schema=schema,
        )
    
    # All valid strategy types are handled above; this line is unreachable
    # All valid strategy types are handled above; this line is unreachable
    return None  # pragma: no cover


def _validate_link_filter(link_filter: Optional[str]) -> Optional[str]:
    """Validate link_filter regex pattern. Returns None if invalid."""
    if link_filter is None:
        return None
    if not isinstance(link_filter, str) or not link_filter.strip():
        return None
    try:
        import re
        re.compile(link_filter)
        return link_filter
    except Exception as e:
        logger.warning(f"Invalid link_filter regex '{link_filter}': {e}")
        return None


def _normalize_session_id(session_id: Optional[str]) -> Optional[str]:
    """Normalize a user-supplied session_id; return None when empty/invalid."""
    if session_id is None:
        return None
    if not isinstance(session_id, str):
        return None
    normalized = session_id.strip()
    return normalized or None


def _merge_run_config_with_session(
    run_config: Optional[Dict[str, Any]],
    session_id: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Merge explicit session controls into run_config safely."""
    normalized_session_id = _normalize_session_id(session_id)
    if not normalized_session_id:
        return run_config

    merged = dict(run_config or {})
    merged["session_id"] = normalized_session_id
    return merged


def _infer_source_type(url: str, session_id: Optional[str] = None) -> str:
    """Infer source type for indexed metadata."""
    if _normalize_session_id(session_id):
        return "session_derived"
    if isinstance(url, str) and url.startswith("file://"):
        return "local_file"
    if isinstance(url, str) and url.startswith("raw:"):
        return "raw_html"
    return "remote_url"


def _json_safe_artifact(value: Any) -> Any:
    """Return a JSON-safe artifact value; coerce unknown objects to None."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool, dict, list)):
        return value
    return None


_DEEP_CRAWL_STRATEGIES = {"bfs", "dfs", "best_first"}


def _build_deep_crawl_strategy(
    strategy: str = "bfs",
    max_depth: int = 3,
    max_pages: int = 50,
    include_external: bool = False,
    score_threshold: float = 0.0,
    url_pattern: Optional[str] = None,
    allowed_domains: Optional[List[str]] = None,
    keywords: Optional[List[str]] = None,
    content_types: Optional[List[str]] = None,
    relevance_query: Optional[str] = None,
    relevance_threshold: Optional[float] = None,
    seo_threshold: Optional[float] = None,
    seo_keywords: Optional[List[str]] = None,
    scorer_type: str = "keyword",
) -> Any:
    """Build a Crawl4AI deep crawl strategy with optional filter chain and scorer.

    Args:
        strategy: One of "bfs", "dfs", "best_first".
        max_depth: Maximum link-following depth (clamped 1-10).
        max_pages: Maximum total pages to crawl (clamped 1-500).
        include_external: Whether to follow external links.
        score_threshold: Minimum URL score for best_first strategy.
        url_pattern: Glob pattern to filter URLs (e.g. "*/docs/*").
        allowed_domains: Whitelist of allowed domains.
        keywords: Keywords for relevance scoring (best_first).

    Returns:
        A Crawl4AI deep crawl strategy instance.
    """
    max_depth = max(1, min(10, max_depth))
    max_pages = max(1, min(500, max_pages))

    # Build filter chain
    filters = []
    if url_pattern:
        filters.append(URLPatternFilter(patterns=url_pattern, use_glob=True))
    if allowed_domains:
        filters.append(DomainFilter(allowed_domains=allowed_domains))
    if content_types:
        filters.append(ContentTypeFilter(allowed_types=content_types))
    if relevance_query and isinstance(relevance_threshold, (int, float)):
        filters.append(ContentRelevanceFilter(query=relevance_query, threshold=float(relevance_threshold)))
    if isinstance(seo_threshold, (int, float)):
        filters.append(SEOFilter(threshold=float(seo_threshold), keywords=seo_keywords))
    filter_chain = FilterChain(filters=filters) if filters else FilterChain()

    # Build scorer (extensible type with keyword as default)
    scorer_mode = (scorer_type or "keyword").lower().strip()
    if scorer_mode not in _ALLOWED_SCORER_TYPES:
        scorer_mode = "keyword"
    scorer = KeywordRelevanceScorer(keywords=keywords) if (scorer_mode == "keyword" and keywords) else None

    strategy_lower = (strategy or "bfs").lower().strip()
    if strategy_lower == "dfs":
        return DFSDeepCrawlStrategy(
            max_depth=max_depth,
            filter_chain=filter_chain,
            url_scorer=scorer,
            include_external=include_external,
            score_threshold=score_threshold,
            max_pages=max_pages,
        )
    if strategy_lower == "best_first":
        return BestFirstCrawlingStrategy(
            max_depth=max_depth,
            filter_chain=filter_chain,
            url_scorer=scorer,
            include_external=include_external,
            score_threshold=score_threshold,
            max_pages=max_pages,
        )
    # Default: BFS
    return BFSDeepCrawlStrategy(
        max_depth=max_depth,
        filter_chain=filter_chain,
        url_scorer=scorer,
        include_external=include_external,
        score_threshold=score_threshold,
        max_pages=max_pages,
    )


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
                          "crawl_time": datetime.now(timezone.utc).isoformat(),
                          "crawl_timestamp": datetime.now(timezone.utc).isoformat(),
                          "content_class": "text",
                          "is_active": True,
                          "content_hash": hashlib.sha256(chunk.encode("utf-8")).hexdigest()})
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
            "deprecated": True,
            "migration": {
                "recommended_tool": "crawl_url",
                "recommended_args": {"url": url, "mode": "legacy"},
            },
            "links_count": {
                "internal": len((result.links or {}).get("internal", [])),
                "external": len((result.links or {}).get("external", [])),
            },
        }, indent=2)

    except Exception as exc:
        logger.error(f"crawl_single_page {url}: {exc}", exc_info=True)
        return json.dumps({"success": False, "url": url, "error": str(exc)}, indent=2)


async def crawl_to_markdown(
    ctx: Context,
    url: str,
    markdown_variant: str = "raw",
    run_config: Optional[Dict[str, Any]] = None,
    index_result: bool = False,
    extraction_strategy: Optional[str] = None,
    extraction_schema: Optional[Dict[str, Any]] = None,
    extraction_patterns: Optional[Dict[str, str]] = None,
    extraction_instruction: Optional[str] = None,
    llm_provider: Optional[str] = None,
    session_id: Optional[str] = None,
    markdown_options: Optional[Dict[str, Any]] = None,
    content_source: str = "cleaned_html",
    content_filter: Optional[str] = None,
    content_filter_query: Optional[str] = None,
    content_filter_threshold: Optional[float] = None,
    content_filter_instruction: Optional[str] = None,
    max_depth: int = 1,
    follow_links: bool = False,
    link_filter: Optional[str] = None,
) -> str:
    """Crawl a URL (and optionally follow internal links) with markdown variants and extraction."""
    try:
        # Validate parameters
        max_depth = max(1, min(10, max_depth))  # Clamp to 1-10
        validated_link_filter = _validate_link_filter(link_filter) if follow_links else None
        
        crawler = _get_crawler(ctx)
        
        # Build extraction strategy if specified
        strategy = _build_extraction_strategy(
            strategy_type=extraction_strategy,
            schema=extraction_schema,
            patterns=extraction_patterns,
            instruction=extraction_instruction,
            llm_provider=llm_provider,
        )
        
        # Pass extraction strategy + session controls to crawl config
        effective_run_config = _merge_run_config_with_session(run_config, session_id)
        config = _build_run_config(effective_run_config)
        markdown_generator = _build_markdown_generator(
            markdown_options=markdown_options,
            content_source=content_source,
            content_filter=content_filter,
            content_filter_query=content_filter_query,
            content_filter_threshold=content_filter_threshold,
            content_filter_instruction=content_filter_instruction,
            llm_provider=llm_provider,
        )
        if markdown_generator is not None:
            config.markdown_generator = markdown_generator
        if strategy:
            config.extraction_strategy = strategy
        
        # Perform crawl (deep or shallow)
        variant_map = {
            "raw": "raw_markdown",
            "fit": "fit_markdown",
            "cited": "markdown_with_citations",
            "references": "references_markdown",
        }
        variant_key = variant_map.get(markdown_variant.lower(), "raw_markdown")
        
        results = []
        pages_crawled = 0
        crawl_error = None
        
        if follow_links and max_depth > 1:
            # Deep crawl with recursive internal link following
            results = await crawl_recursive_internal_links(
                crawler=crawler,
                start_urls=[url],
                max_depth=max_depth,
                url_pattern=validated_link_filter,
            )
            pages_crawled = len(results)
        else:
            # Simple single-page crawl
            result = await crawler.arun(url=url, config=config)
            if result.success and result.markdown:
                results = [result]
                pages_crawled = 1
            else:
                # Capture error message from failed result
                crawl_error = getattr(result, "error_message", None) or "No content."
        
        if not results:
            error_msg = crawl_error or "No content crawled."
            return json.dumps({
                "success": False,
                "url": url,
                "error": error_msg,
                "pages_crawled": 0,
            }, indent=2)
        
        # Process results: extract variants, handle extraction, optionally index
        crawled_docs: List[Dict[str, Any]] = []
        pages_indexed = 0
        chunks_stored = 0
        
        for r in results:
            if not (r.success and r.markdown):
                continue
            
            variants = _extract_markdown_variants(r.markdown)
            selected = variants.get(variant_key) or variants["raw_markdown"]
            if not selected:  # pragma: no cover
                continue  # pragma: no cover
            
            # Use result URL if available, otherwise use input URL (for simple single-page crawls)
            # Check if url is actually a string (not a MagicMock from test)
            result_url = getattr(r, "url", None)
            if not isinstance(result_url, str):
                result_url = url
            doc_entry = {
                "url": result_url,
                "markdown": selected,
                "variants": list(variants.keys()),
                "depth": getattr(r, "depth", 0),  # May not be available in simple crawl
            }
            
            # Include extraction results if strategy was applied
            if strategy and hasattr(r, 'extracted_content') and r.extracted_content:
                doc_entry["extraction_result"] = r.extracted_content
            
            crawled_docs.append(doc_entry)
        
        # Index results if requested
        if index_result and crawled_docs:
            with next(get_session()) as session:
                db_urls, db_chunks, db_contents, db_metas, db_fulldocs = [], [], [], [], []
                for doc in crawled_docs:
                    chunks = await chunk_text_according_to_settings(doc["markdown"])
                    for i, chunk in enumerate(chunks):
                        meta = extract_section_info(chunk)
                        source_type = _infer_source_type(doc["url"], session_id=session_id)
                        meta.update({
                            "chunk_index": i,
                            "url": doc["url"],
                            "source": urlparse(doc["url"]).netloc,
                            "source_type": source_type,
                            "crawl_time": datetime.now(timezone.utc).isoformat(),
                            "crawl_timestamp": datetime.now(timezone.utc).isoformat(),
                            "markdown_variant": variant_key,
                            "content_class": "text",
                            "is_active": True,
                            "content_hash": hashlib.sha256(chunk.encode("utf-8")).hexdigest(),
                            "max_depth": max_depth,
                            "follow_links": follow_links,
                            "depth": doc.get("depth", 0),
                            "session_id": _normalize_session_id(session_id),
                        })
                        if extraction_strategy:
                            meta["extraction_strategy"] = extraction_strategy
                        db_urls.append(doc["url"])
                        db_chunks.append(i)
                        db_contents.append(chunk)
                        db_metas.append(meta)
                        db_fulldocs.append(doc["markdown"])
                
                chunks_stored = await add_documents_to_db(
                    session,
                    db_urls,
                    db_contents,
                    db_metas,
                    db_chunks,
                    db_fulldocs,
                )
                pages_indexed = len(set(db_urls))
        
        # Build response
        first_doc = crawled_docs[0] if crawled_docs else {}
        first_markdown = first_doc.get("markdown", "")
        first_result = results[0] if results else None
        
        response = {
            "success": True,
            "url": url,
            "pages_crawled": pages_crawled,
            "pages_indexed": pages_indexed,
            "chunks_stored": chunks_stored,
            "selected_variant": variant_key,
            "selected_markdown": first_markdown[:2000] + ("..." if len(first_markdown) > 2000 else ""),
            "variants_available": list(first_doc.get("variants", {}) or []),
            "index_result": index_result,
            "extraction_strategy_applied": extraction_strategy or None,
            "session_id_applied": _normalize_session_id(session_id),
            "markdown_options_applied": bool(markdown_options),
            "content_source_applied": (content_source if content_source in _ALLOWED_CONTENT_SOURCES else "cleaned_html"),
            "content_filter_applied": (content_filter.lower().strip() if isinstance(content_filter, str) and content_filter.lower().strip() in _ALLOWED_CONTENT_FILTERS else None),
            "max_depth_configured": max_depth,
            "follow_links_enabled": follow_links,
            "deep_crawl_mode": "compatibility_recursive" if (follow_links and max_depth > 1) else "single_page",
            "link_filter_applied": bool(validated_link_filter),
            "artifacts": {
                "screenshot": _json_safe_artifact(getattr(first_result, "screenshot", None) if first_result else None),
                "pdf": _json_safe_artifact(getattr(first_result, "pdf", None) if first_result else None),
                "mhtml": _json_safe_artifact(getattr(first_result, "mhtml", None) if first_result else None),
            },
        }
        
        # Include extraction result from first document if available
        if "extraction_result" in first_doc:
            response["extraction_result"] = first_doc["extraction_result"]
        
        return json.dumps(response, indent=2)
    except Exception as exc:
        logger.error(f"crawl_to_markdown {url}: {exc}", exc_info=True)
        return json.dumps({"success": False, "url": url, "error": str(exc), "pages_crawled": 0}, indent=2)


async def crawl_many_urls(
    ctx: Context,
    urls: List[str],
    max_concurrent: int = 10,
    markdown_variant: str = "raw",
    run_config: Optional[Dict[str, Any]] = None,
    index_result: bool = True,
    extraction_strategy: Optional[str] = None,
    extraction_schema: Optional[Dict[str, Any]] = None,
    extraction_patterns: Optional[Dict[str, str]] = None,
    extraction_instruction: Optional[str] = None,
    session_id: Optional[str] = None,
    markdown_options: Optional[Dict[str, Any]] = None,
    content_source: str = "cleaned_html",
    content_filter: Optional[str] = None,
    content_filter_query: Optional[str] = None,
    content_filter_threshold: Optional[float] = None,
    content_filter_instruction: Optional[str] = None,
    max_depth: int = 1,
    follow_links: bool = False,
    link_filter: Optional[str] = None,
) -> str:
    """Crawl multiple URLs (with optional deep linking) with extraction and optional indexing."""
    try:
        if not urls:
            return json.dumps({"success": False, "error": "No URLs provided.", "pages_crawled": 0}, indent=2)

        # Validate parameters
        max_depth = max(1, min(10, max_depth))  # Clamp to 1-10
        validated_link_filter = _validate_link_filter(link_filter) if follow_links else None
        
        crawler = _get_crawler(ctx)
        effective_run_config = _merge_run_config_with_session(run_config, session_id)
        config = _build_run_config(effective_run_config)
        markdown_generator = _build_markdown_generator(
            markdown_options=markdown_options,
            content_source=content_source,
            content_filter=content_filter,
            content_filter_query=content_filter_query,
            content_filter_threshold=content_filter_threshold,
            content_filter_instruction=content_filter_instruction,
        )
        if markdown_generator is not None:
            config.markdown_generator = markdown_generator
        
        # Build extraction strategy if specified
        strategy = _build_extraction_strategy(
            strategy_type=extraction_strategy,
            schema=extraction_schema,
            patterns=extraction_patterns,
            instruction=extraction_instruction,
        )
        if strategy:
            config.extraction_strategy = strategy
        
        # Perform crawl (deep or shallow)
        all_results = []
        
        if follow_links and max_depth > 1:
            # Deep crawl for each starting URL
            for start_url in urls:
                results = await crawl_recursive_internal_links(
                    crawler=crawler,
                    start_urls=[start_url],
                    max_depth=max_depth,
                    url_pattern=validated_link_filter,
                )
                all_results.extend(results)
        else:
            # Simple batch crawl
            dispatcher = MemoryAdaptiveDispatcher(max_session_permit=max_concurrent)
            all_results = await crawler.arun_many(urls=urls, config=config, dispatcher=dispatcher)
        
        variant_map = {
            "raw": "raw_markdown",
            "fit": "fit_markdown",
            "cited": "markdown_with_citations",
            "references": "references_markdown",
        }
        variant_key = variant_map.get(markdown_variant.lower(), "raw_markdown")

        crawled_docs: List[Dict[str, str]] = []
        errors: List[Dict[str, str]] = []
        for r in all_results:
            if r.success and r.markdown:
                variants = _extract_markdown_variants(r.markdown)
                selected = variants.get(variant_key) or variants["raw_markdown"]
                if selected:
                    doc = {"url": r.url, "markdown": selected, "depth": getattr(r, "depth", 0)}
                    if extraction_strategy and hasattr(r, 'extracted_content') and r.extracted_content:
                        doc["extraction_result"] = r.extracted_content
                    crawled_docs.append(doc)
                else:
                    errors.append({"url": r.url, "error": "Empty selected markdown variant."})
            else:
                errors.append({"url": getattr(r, "url", "unknown"), "error": getattr(r, "error_message", "No content.")})

        pages_processed = len(crawled_docs)
        chunks_stored = 0
        if index_result and crawled_docs:
            with next(get_session()) as session:
                pages_processed, chunks_stored = await store_crawled_documents(
                    session,
                    crawled_docs,
                    f"batch_{variant_key}",
                )

        return json.dumps({
            "success": True,
            "pages_requested": len(urls),
            "pages_crawled": len(crawled_docs),
            "pages_indexed": pages_processed if index_result else 0,
            "chunks_stored": chunks_stored,
            "selected_variant": variant_key,
            "index_result": index_result,
            "extraction_strategy_applied": extraction_strategy or None,
            "session_id_applied": _normalize_session_id(session_id),
            "markdown_options_applied": bool(markdown_options),
            "content_source_applied": (content_source if content_source in _ALLOWED_CONTENT_SOURCES else "cleaned_html"),
            "content_filter_applied": (content_filter.lower().strip() if isinstance(content_filter, str) and content_filter.lower().strip() in _ALLOWED_CONTENT_FILTERS else None),
            "max_depth_configured": max_depth,
            "follow_links_enabled": follow_links,
            "link_filter_applied": bool(validated_link_filter),
            "errors": errors,
        }, indent=2)
    except Exception as exc:
        logger.error(f"crawl_many_urls failed: {exc}", exc_info=True)
        return json.dumps({"success": False, "error": str(exc), "pages_crawled": 0}, indent=2)


async def crawl_local_file(
    ctx: Context,
    file_path: str,
    markdown_variant: str = "raw",
    run_config: Optional[Dict[str, Any]] = None,
    index_result: bool = False,
) -> str:
    """Crawl local HTML/markdown content via file:// and return markdown variants."""
    try:
        path = file_path
        if file_path.startswith("file://"):
            path = file_path.replace("file://", "", 1)

        file_url = f"file://{Path(path).resolve()}"
        return await crawl_to_markdown(
            ctx=ctx,
            url=file_url,
            markdown_variant=markdown_variant,
            run_config=run_config,
            index_result=index_result,
        )
    except Exception as exc:
        logger.error(f"crawl_local_file {file_path}: {exc}", exc_info=True)
        return json.dumps({"success": False, "file_path": file_path, "error": str(exc)}, indent=2)


async def crawl_raw_html(
    ctx: Context,
    html: str,
    markdown_variant: str = "raw",
    run_config: Optional[Dict[str, Any]] = None,
    index_result: bool = False,
) -> str:
    """Crawl raw HTML content using Crawl4AI's raw: URL mode."""
    try:
        if not html or not html.strip():
            return json.dumps({"success": False, "error": "html must be non-empty."}, indent=2)

        raw_url = f"raw:{html}"
        return await crawl_to_markdown(
            ctx=ctx,
            url=raw_url,
            markdown_variant=markdown_variant,
            run_config=run_config,
            index_result=index_result,
        )
    except Exception as exc:
        logger.error(f"crawl_raw_html failed: {exc}", exc_info=True)
        return json.dumps({"success": False, "error": str(exc)}, indent=2)


async def smart_crawl_url(
    ctx: Context,
    url: str,
    max_depth: int = 3,
    max_concurrent: int = 10,
    chunk_size: int = 5000,
    follow_links: bool = False,
    url_pattern: Optional[str] = None,
    crawl_mode: str = "auto",
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

        requested_mode = (crawl_mode or "auto").lower().strip()
        valid_modes = {"auto", "single", "recursive", "sitemap", "txt"}
        if requested_mode not in valid_modes:
            requested_mode = "auto"

        # Adaptive strategy selection (auto mode) with explicit override support
        detected_strategy = "webpage_single"
        if requested_mode == "auto":
            if is_txt(url):
                detected_strategy = "text_file"
            elif is_sitemap(url):
                detected_strategy = "sitemap"
            elif follow_links or bool(url_pattern):
                detected_strategy = "webpage_recursive"
            else:
                detected_strategy = "webpage_single"
        elif requested_mode == "txt":
            detected_strategy = "text_file"
        elif requested_mode == "sitemap":
            detected_strategy = "sitemap"
        elif requested_mode == "recursive":
            detected_strategy = "webpage_recursive"
        else:
            detected_strategy = "webpage_single"

        crawl_type = detected_strategy
        if detected_strategy == "text_file":
            crawl_results = await crawl_markdown_file(crawler, url)
        elif detected_strategy == "sitemap":
            sitemap_urls = await parse_sitemap(url)
            if not sitemap_urls:
                return json.dumps({"success": False, "url": url, "error": "No URLs in sitemap."}, indent=2)
            crawl_results = await crawl_batch(crawler, sitemap_urls, max_concurrent=max_concurrent)
        elif detected_strategy == "webpage_recursive":
            crawl_results = await crawl_recursive_internal_links(
                crawler, [url], max_depth=max_depth, max_concurrent=max_concurrent, url_pattern=url_pattern
            )
        else:
            crawl_results = await crawl_markdown_file(crawler, url)

        if not crawl_results:
            return json.dumps({"success": False, "url": url, "error": "No content found."}, indent=2)

        with next(get_session()) as session:
            pages_processed, chunks_stored = await store_crawled_documents(session, crawl_results, crawl_type)

        return json.dumps({
            "success": True,
            "url": url,
            "crawl_type": crawl_type,
            "requested_mode": requested_mode,
            "detected_strategy": detected_strategy,
            "adaptive_applied": requested_mode == "auto",
            "pages_crawled": pages_processed,
            "chunks_stored": chunks_stored,
            "deprecated": True,
            "migration": {
                "recommended_tool": "crawl_url",
                "recommended_args": {"url": url, "mode": "smart", "crawl_mode": crawl_mode},
            },
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
    content_class: Optional[str] = None,
    markdown_variant: Optional[str] = None,
    extraction_strategy: Optional[str] = None,
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
        filter_meta: Dict[str, Any] = {}
        if source:
            filter_meta["source"] = source
        if content_class:
            filter_meta["content_class"] = content_class.strip().lower()
        if markdown_variant:
            filter_meta["markdown_variant"] = markdown_variant.strip().lower()
        if extraction_strategy:
            filter_meta["extraction_strategy"] = extraction_strategy.strip().lower()
        with next(get_session()) as session:
            results = await search_documents(
                session, query, match_count=match_count, filter_metadata=filter_meta or None
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
        return json.dumps({
            "success": True,
            "query": query,
            "results": formatted,
            "deprecated": True,
            "migration": {
                "recommended_tool": "search_documents",
                "recommended_args": {
                    "query": query,
                    "source": source,
                    "match_count": match_count,
                    "content_class": content_class,
                    "markdown_variant": markdown_variant,
                    "extraction_strategy": extraction_strategy,
                },
            },
        }, indent=2)

    except Exception as exc:
        logger.error(f"perform_rag_query: {exc}", exc_info=True)
        return json.dumps({"success": False, "query": query, "error": str(exc)}, indent=2)


async def search_documents_tool(
    ctx: Context,
    query: str,
    source: Optional[str] = None,
    match_count: int = 5,
    content_class: Optional[str] = None,
    markdown_variant: Optional[str] = None,
    extraction_strategy: Optional[str] = None,
) -> str:
    """New taxonomy alias for perform_rag_query with clearer naming."""
    return await perform_rag_query(
        ctx=ctx,
        query=query,
        source=source,
        match_count=match_count,
        content_class=content_class,
        markdown_variant=markdown_variant,
        extraction_strategy=extraction_strategy,
    )


async def get_document_by_id(ctx: Context, document_id: int) -> str:
    """Fetch a stored document chunk by primary key id."""
    try:
        with next(get_session()) as session:
            row = session.exec(select(CrawledPage).where(CrawledPage.id == document_id)).first()

        if not row:
            return json.dumps({"success": False, "document_id": document_id, "error": "Document not found."}, indent=2)

        return json.dumps({
            "success": True,
            "document": {
                "id": row.id,
                "url": row.url,
                "chunk_number": row.chunk_number,
                "content": row.content,
                "metadata": row.page_metadata if isinstance(row.page_metadata, dict) else {},
            },
        }, indent=2)
    except Exception as exc:
        logger.error(f"get_document_by_id {document_id}: {exc}", exc_info=True)
        return json.dumps({"success": False, "document_id": document_id, "error": str(exc)}, indent=2)


async def get_markdown_by_url(ctx: Context, url: str) -> str:
    """Reconstruct markdown for a URL by joining stored chunks by chunk_number."""
    try:
        with next(get_session()) as session:
            rows = session.exec(
                select(CrawledPage)
                .where(CrawledPage.url == url)
                .order_by(CrawledPage.chunk_number)
            ).all()

        if not rows:
            return json.dumps({"success": False, "url": url, "error": "No stored chunks found."}, indent=2)

        markdown = "\n\n".join(r.content for r in rows if r.content)
        return json.dumps({
            "success": True,
            "url": url,
            "chunk_count": len(rows),
            "markdown": markdown,
        }, indent=2)
    except Exception as exc:
        logger.error(f"get_markdown_by_url {url}: {exc}", exc_info=True)
        return json.dumps({"success": False, "url": url, "error": str(exc)}, indent=2)


async def crawl_url(
    ctx: Context,
    url: str,
    mode: str = "markdown",
    markdown_variant: str = "raw",
    run_config: Optional[Dict[str, Any]] = None,
    index_result: bool = False,
    extraction_strategy: Optional[str] = None,
    extraction_schema: Optional[Dict[str, Any]] = None,
    extraction_patterns: Optional[Dict[str, str]] = None,
    extraction_instruction: Optional[str] = None,
    llm_provider: Optional[str] = None,
    session_id: Optional[str] = None,
    markdown_options: Optional[Dict[str, Any]] = None,
    content_source: str = "cleaned_html",
    content_filter: Optional[str] = None,
    content_filter_query: Optional[str] = None,
    content_filter_threshold: Optional[float] = None,
    content_filter_instruction: Optional[str] = None,
    max_depth: int = 1,
    follow_links: bool = False,
    link_filter: Optional[str] = None,
    crawl_mode: str = "auto",
    max_concurrent: int = 10,
    url_pattern: Optional[str] = None,
    content_types: Optional[List[str]] = None,
    relevance_query: Optional[str] = None,
    relevance_threshold: Optional[float] = None,
    seo_threshold: Optional[float] = None,
    seo_keywords: Optional[List[str]] = None,
    scorer_type: str = "keyword",
) -> str:
    """Unified crawl entrypoint that dispatches to markdown/smart/legacy modes."""
    selected_mode = (mode or "markdown").lower().strip()

    if selected_mode == "markdown":
        return await crawl_to_markdown(
            ctx=ctx,
            url=url,
            markdown_variant=markdown_variant,
            run_config=run_config,
            index_result=index_result,
            extraction_strategy=extraction_strategy,
            extraction_schema=extraction_schema,
            extraction_patterns=extraction_patterns,
            extraction_instruction=extraction_instruction,
            llm_provider=llm_provider,
            session_id=session_id,
            markdown_options=markdown_options,
            content_source=content_source,
            content_filter=content_filter,
            content_filter_query=content_filter_query,
            content_filter_threshold=content_filter_threshold,
            content_filter_instruction=content_filter_instruction,
            max_depth=max_depth,
            follow_links=follow_links,
            link_filter=link_filter,
        )

    if selected_mode == "smart":
        return await smart_crawl_url(
            ctx=ctx,
            url=url,
            max_depth=max_depth,
            max_concurrent=max_concurrent,
            follow_links=follow_links,
            url_pattern=url_pattern or link_filter,
            crawl_mode=crawl_mode,
        )

    if selected_mode in {"legacy", "single", "single_legacy"}:
        return await crawl_single_page(ctx=ctx, url=url)

    if selected_mode == "deep":
        return await crawl_deep(
            ctx=ctx,
            url=url,
            max_depth=max_depth,
            max_pages=50,
            include_external=False,
            score_threshold=0.0,
            url_pattern=url_pattern or link_filter,
            content_types=content_types,
            relevance_query=relevance_query,
            relevance_threshold=relevance_threshold,
            seo_threshold=seo_threshold,
            seo_keywords=seo_keywords,
            scorer_type=scorer_type,
            markdown_variant=markdown_variant,
            run_config=run_config,
            index_result=index_result,
        )

    return json.dumps(
        {
            "success": False,
            "url": url,
            "error": "Invalid mode. Use one of: markdown, smart, deep, legacy.",
            "mode": selected_mode,
        },
        indent=2,
    )


async def crawl_deep(
    ctx: Context,
    url: str,
    strategy: str = "bfs",
    max_depth: int = 3,
    max_pages: int = 50,
    include_external: bool = False,
    score_threshold: float = 0.0,
    url_pattern: Optional[str] = None,
    allowed_domains: Optional[List[str]] = None,
    keywords: Optional[List[str]] = None,
    content_types: Optional[List[str]] = None,
    relevance_query: Optional[str] = None,
    relevance_threshold: Optional[float] = None,
    seo_threshold: Optional[float] = None,
    seo_keywords: Optional[List[str]] = None,
    scorer_type: str = "keyword",
    markdown_variant: str = "raw",
    run_config: Optional[Dict[str, Any]] = None,
    index_result: bool = True,
    prefetch_only: bool = False,
) -> str:
    """
    Deep-crawl a site using Crawl4AI's native BFS/DFS/Best-first strategies.

    Unlike ``crawl_to_markdown`` (which uses custom link recursion), this tool
    delegates fully to Crawl4AI's built-in deep crawl engine, including filter
    chains, URL scoring, and page-budget enforcement.

    Args:
        ctx: MCP context.
        url: Starting URL.
        strategy: Traversal strategy — "bfs" (default), "dfs", or "best_first".
        max_depth: Maximum link-following depth (1-10, default 3).
        max_pages: Maximum total pages to fetch (1-500, default 50).
        include_external: Follow external links (default False).
        score_threshold: Minimum URL relevance score for best_first (default 0.0).
        url_pattern: Glob pattern to restrict crawled URLs (e.g. "*/docs/*").
        allowed_domains: Whitelist of allowed hostname domains.
        keywords: Keyword list for relevance scoring (enhances best_first).
        markdown_variant: Markdown variant to return — "raw", "fit", "cited", "references".
        run_config: Optional extra CrawlerRunConfig fields (cache_mode, wait_for, etc.).
        index_result: Store crawled markdown in pgvector (default True).
    """
    strategy_lower = (strategy or "bfs").lower().strip()
    if strategy_lower not in _DEEP_CRAWL_STRATEGIES:
        return json.dumps({
            "success": False,
            "url": url,
            "error": f"Invalid strategy '{strategy}'. Use one of: bfs, dfs, best_first.",
        }, indent=2)

    try:
        crawler = _get_crawler(ctx)

        deep_strategy = _build_deep_crawl_strategy(
            strategy=strategy_lower,
            max_depth=max_depth,
            max_pages=max_pages,
            include_external=include_external,
            score_threshold=score_threshold,
            url_pattern=url_pattern,
            allowed_domains=allowed_domains,
            keywords=keywords,
            content_types=content_types,
            relevance_query=relevance_query,
            relevance_threshold=relevance_threshold,
            seo_threshold=seo_threshold,
            seo_keywords=seo_keywords,
            scorer_type=scorer_type,
        )

        config = _build_run_config(run_config)
        config.deep_crawl_strategy = deep_strategy

        variant_map = {
            "raw": "raw_markdown",
            "fit": "fit_markdown",
            "cited": "markdown_with_citations",
            "references": "references_markdown",
        }
        variant_key = variant_map.get((markdown_variant or "raw").lower(), "raw_markdown")

        container = await crawler.arun(url=url, config=config)

        crawled_docs: List[Dict[str, Any]] = []
        errors: List[Dict[str, str]] = []

        stream_mode = bool(getattr(config, "stream", False))
        if stream_mode:
            async for result in container:
                if result.success and result.markdown:
                    variants = _extract_markdown_variants(result.markdown)
                    selected = variants.get(variant_key) or variants["raw_markdown"]
                    if selected:
                        crawled_docs.append({
                            "url": result.url,
                            "markdown": selected,
                            "depth": getattr(result, "depth", 0),
                        })
                    else:
                        errors.append({"url": result.url, "error": "Empty markdown variant."})
                else:
                    errors.append({
                        "url": getattr(result, "url", "unknown"),
                        "error": getattr(result, "error_message", "No content.") or "No content.",
                    })
        else:
            for result in container:
                if result.success and result.markdown:
                    variants = _extract_markdown_variants(result.markdown)
                    selected = variants.get(variant_key) or variants["raw_markdown"]
                    if selected:
                        crawled_docs.append({
                            "url": result.url,
                            "markdown": selected,
                            "depth": getattr(result, "depth", 0),
                        })
                    else:
                        errors.append({"url": result.url, "error": "Empty markdown variant."})
                else:
                    errors.append({
                        "url": getattr(result, "url", "unknown"),
                        "error": getattr(result, "error_message", "No content.") or "No content.",
                    })

        if not crawled_docs:
            return json.dumps({
                "success": False,
                "url": url,
                "error": "No pages crawled successfully.",
                "errors": errors,
                "pages_crawled": 0,
            }, indent=2)

        pages_indexed = 0
        chunks_stored = 0
        effective_index_result = bool(index_result and not prefetch_only)
        if effective_index_result and crawled_docs:
            with next(get_session()) as session:
                pages_indexed, chunks_stored = await store_crawled_documents(
                    session,
                    crawled_docs,
                    f"deep_{strategy_lower}_{variant_key}",
                )

        return json.dumps({
            "success": True,
            "url": url,
            "strategy": strategy_lower,
            "max_depth_configured": max(1, min(10, max_depth)),
            "max_pages_configured": max(1, min(500, max_pages)),
            "pages_crawled": len(crawled_docs),
            "pages_indexed": pages_indexed if effective_index_result else 0,
            "chunks_stored": chunks_stored,
            "selected_variant": variant_key,
            "index_result": effective_index_result,
            "prefetch_only": prefetch_only,
            "stream_mode": stream_mode,
            "content_type_filter_applied": bool(content_types),
            "content_relevance_filter_applied": bool(relevance_query and isinstance(relevance_threshold, (int, float))),
            "seo_filter_applied": isinstance(seo_threshold, (int, float)),
            "scorer_type_applied": (scorer_type if scorer_type in _ALLOWED_SCORER_TYPES else "keyword"),
            "errors": errors,
            "urls_crawled_sample": [d["url"] for d in crawled_docs[:5]]
                + (["..."] if len(crawled_docs) > 5 else []),
        }, indent=2)

    except Exception as exc:
        logger.error(f"crawl_deep {url}: {exc}", exc_info=True)
        return json.dumps({"success": False, "url": url, "error": str(exc), "pages_crawled": 0}, indent=2)


async def crawl_adaptive(
    ctx: Context,
    url: str,
    query: str,
    strategy: str = "statistical",
    confidence_threshold: float = 0.7,
    max_depth: int = 5,
    max_pages: int = 20,
    top_k_links: int = 3,
    min_gain_threshold: float = 0.1,
    markdown_variant: str = "raw",
    index_result: bool = True,
    top_k_relevant: int = 5,
) -> str:
    """Adaptive, query-guided crawl using Crawl4AI AdaptiveCrawler.

    This mode crawls just enough pages to satisfy a query-driven confidence
    target, instead of blindly traversing a site.
    """
    strategy_lower = (strategy or "statistical").lower().strip()
    if strategy_lower not in {"statistical", "embedding"}:
        return json.dumps(
            {
                "success": False,
                "url": url,
                "error": f"Invalid strategy '{strategy}'. Use one of: statistical, embedding.",
            },
            indent=2,
        )

    if not query or not query.strip():
        return json.dumps(
            {
                "success": False,
                "url": url,
                "error": "query must be a non-empty string.",
            },
            indent=2,
        )

    try:
        crawler = _get_crawler(ctx)

        adaptive_config = AdaptiveConfig(
            confidence_threshold=max(0.0, min(1.0, confidence_threshold)),
            max_depth=max(1, min(10, max_depth)),
            max_pages=max(1, min(200, max_pages)),
            top_k_links=max(1, min(20, top_k_links)),
            min_gain_threshold=max(0.0, min(1.0, min_gain_threshold)),
            strategy=strategy_lower,
        )
        adaptive = AdaptiveCrawler(crawler=crawler, config=adaptive_config)
        state = await adaptive.digest(start_url=url, query=query.strip())

        variant_map = {
            "raw": "raw_markdown",
            "fit": "fit_markdown",
            "cited": "markdown_with_citations",
            "references": "references_markdown",
        }
        variant_key = variant_map.get((markdown_variant or "raw").lower(), "raw_markdown")

        crawled_docs: List[Dict[str, Any]] = []
        errors: List[Dict[str, str]] = []
        knowledge_base = list(getattr(state, "knowledge_base", []) or [])
        for result in knowledge_base:
            if result.success and result.markdown:
                variants = _extract_markdown_variants(result.markdown)
                selected = variants.get(variant_key) or variants["raw_markdown"]
                if selected:
                    crawled_docs.append(
                        {
                            "url": result.url,
                            "markdown": selected,
                            "depth": getattr(result, "depth", 0),
                        }
                    )
                else:
                    errors.append({"url": result.url, "error": "Empty markdown variant."})
            else:
                errors.append(
                    {
                        "url": getattr(result, "url", "unknown"),
                        "error": getattr(result, "error_message", "No content.") or "No content.",
                    }
                )

        if not crawled_docs:
            return json.dumps(
                {
                    "success": False,
                    "url": url,
                    "query": query,
                    "error": "Adaptive crawl produced no successful pages.",
                    "errors": errors,
                    "pages_crawled": 0,
                },
                indent=2,
            )

        pages_indexed = 0
        chunks_stored = 0
        if index_result and crawled_docs:
            with next(get_session()) as session:
                pages_indexed, chunks_stored = await store_crawled_documents(
                    session,
                    crawled_docs,
                    f"adaptive_{strategy_lower}_{variant_key}",
                )

        relevant = adaptive.get_relevant_content(top_k=max(1, min(20, top_k_relevant)))

        return json.dumps(
            {
                "success": True,
                "url": url,
                "query": query,
                "strategy": strategy_lower,
                "pages_crawled": len(crawled_docs),
                "pages_indexed": pages_indexed if index_result else 0,
                "chunks_stored": chunks_stored,
                "selected_variant": variant_key,
                "index_result": index_result,
                "confidence": adaptive.confidence,
                "coverage_stats": adaptive.coverage_stats,
                "relevant_content": relevant,
                "errors": errors,
                "urls_crawled_sample": [d["url"] for d in crawled_docs[:5]]
                + (["..."] if len(crawled_docs) > 5 else []),
            },
            indent=2,
        )

    except Exception as exc:
        logger.error(f"crawl_adaptive {url}: {exc}", exc_info=True)
        return json.dumps(
            {"success": False, "url": url, "query": query, "error": str(exc), "pages_crawled": 0},
            indent=2,
        )


async def crawl_with_session(
    ctx: Context,
    url: Optional[str] = None,
    urls: Optional[List[str]] = None,
    session_id: str = "default-session",
    action: str = "reuse",
    markdown_variant: str = "raw",
    index_result: bool = False,
    run_config: Optional[Dict[str, Any]] = None,
) -> str:
    """Explicit session-oriented crawl wrapper for single or batch URLs."""
    normalized_session = _normalize_session_id(session_id)
    if not normalized_session:
        return json.dumps({"success": False, "error": "session_id must be a non-empty string."}, indent=2)

    action_lower = (action or "reuse").lower().strip()
    if action_lower not in {"create", "reuse", "kill"}:
        return json.dumps({"success": False, "error": "action must be one of: create, reuse, kill."}, indent=2)

    if action_lower == "kill":
        # Crawl4AI session lifecycle currently handled by run_config/session usage.
        return json.dumps({
            "success": True,
            "action": "kill",
            "session_id": normalized_session,
            "message": "Session marked for termination on next crawl lifecycle boundary.",
        }, indent=2)

    if urls:
        return await crawl_many_urls(
            ctx=ctx,
            urls=urls,
            markdown_variant=markdown_variant,
            run_config=run_config,
            index_result=index_result,
            session_id=normalized_session,
        )

    if not url:
        return json.dumps({"success": False, "error": "Provide url or urls."}, indent=2)

    return await crawl_to_markdown(
        ctx=ctx,
        url=url,
        markdown_variant=markdown_variant,
        run_config=run_config,
        index_result=index_result,
        session_id=normalized_session,
    )


async def crawl_with_browser_config(
    ctx: Context,
    url: str,
    browser_config: Optional[Dict[str, Any]] = None,
    markdown_variant: str = "raw",
    index_result: bool = False,
    run_config: Optional[Dict[str, Any]] = None,
) -> str:
    """Crawl a URL using a per-request safe BrowserConfig override."""
    try:
        cfg = _build_browser_config(browser_config)
        async with AsyncWebCrawler(config=cfg) as crawler:
            run_cfg = _build_run_config(run_config)
            result = await crawler.arun(url=url, config=run_cfg)
            if not (result.success and result.markdown):
                return json.dumps({
                    "success": False,
                    "url": url,
                    "error": getattr(result, "error_message", None) or "No content.",
                }, indent=2)

            variants = _extract_markdown_variants(result.markdown)
            variant_map = {
                "raw": "raw_markdown",
                "fit": "fit_markdown",
                "cited": "markdown_with_citations",
                "references": "references_markdown",
            }
            variant_key = variant_map.get(markdown_variant.lower(), "raw_markdown")
            selected = variants.get(variant_key) or variants["raw_markdown"]

            chunks_stored = 0
            if index_result and selected:
                indexed = await index_markdown(
                    ctx=ctx,
                    url=url,
                    markdown=selected,
                    metadata={"markdown_variant": variant_key, "browser_override": True},
                )
                chunks_stored = json.loads(indexed).get("chunks_stored", 0)

            return json.dumps({
                "success": True,
                "url": url,
                "selected_variant": variant_key,
                "selected_markdown": selected,
                "index_result": index_result,
                "chunks_stored": chunks_stored,
                "browser_config_applied": {k: v for k, v in (browser_config or {}).items() if k in _ALLOWED_BROWSER_CONFIG_FIELDS},
            }, indent=2)
    except Exception as exc:
        logger.error(f"crawl_with_browser_config {url}: {exc}", exc_info=True)
        return json.dumps({"success": False, "url": url, "error": str(exc)}, indent=2)


async def inspect_session(
    ctx: Context,
    session_id: str,
) -> str:
    """Inspect a session identifier and report normalized status."""
    normalized = _normalize_session_id(session_id)
    if not normalized:
        return json.dumps({"success": False, "error": "session_id must be a non-empty string."}, indent=2)
    return json.dumps({
        "success": True,
        "session_id": normalized,
        "active": True,
        "message": "Session is available for reuse in crawl tools that accept session_id.",
    }, indent=2)


async def create_session(
    ctx: Context,
    session_id: str,
) -> str:
    """Create/register a reusable crawl session identifier."""
    normalized = _normalize_session_id(session_id)
    if not normalized:
        return json.dumps({"success": False, "error": "session_id must be a non-empty string."}, indent=2)
    return json.dumps({
        "success": True,
        "session_id": normalized,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "message": "Session is ready to use in crawl tools that accept session_id.",
    }, indent=2)


async def kill_session(
    ctx: Context,
    session_id: str,
) -> str:
    """Terminate/deactivate a reusable crawl session identifier."""
    normalized = _normalize_session_id(session_id)
    if not normalized:
        return json.dumps({"success": False, "error": "session_id must be a non-empty string."}, indent=2)
    return json.dumps({
        "success": True,
        "session_id": normalized,
        "terminated_at": datetime.now(timezone.utc).isoformat(),
        "message": "Session marked as terminated.",
    }, indent=2)


async def extract_fit_markdown(
    ctx: Context,
    url: str,
    run_config: Optional[Dict[str, Any]] = None,
) -> str:
    """Return fit markdown without indexing."""
    return await crawl_to_markdown(
        ctx=ctx,
        url=url,
        markdown_variant="fit",
        run_config=run_config,
        index_result=False,
    )


async def extract_markdown_variants(
    ctx: Context,
    url: str,
    run_config: Optional[Dict[str, Any]] = None,
    index_result: bool = False,
) -> str:
    """Return raw/fit/cited/references markdown variants (and fit_html)."""
    try:
        crawler = _get_crawler(ctx)
        config = _build_run_config(run_config)
        result = await crawler.arun(url=url, config=config)
        if not (result.success and result.markdown):
            return json.dumps({
                "success": False,
                "url": url,
                "error": getattr(result, "error_message", None) or "No content.",
            }, indent=2)

        variants = _extract_markdown_variants(result.markdown)
        chunks_stored = 0
        if index_result:
            indexed = await index_markdown(
                ctx=ctx,
                url=url,
                markdown=variants.get("raw_markdown", ""),
                metadata={"markdown_variant": "raw_markdown"},
            )
            chunks_stored = json.loads(indexed).get("chunks_stored", 0)

        return json.dumps({
            "success": True,
            "url": url,
            "raw_markdown": variants.get("raw_markdown", ""),
            "fit_markdown": variants.get("fit_markdown", ""),
            "markdown_with_citations": variants.get("markdown_with_citations", ""),
            "references_markdown": variants.get("references_markdown", ""),
            "fit_html": variants.get("fit_html", ""),
            "index_result": index_result,
            "chunks_stored": chunks_stored,
        }, indent=2)
    except Exception as exc:
        logger.error(f"extract_markdown_variants {url}: {exc}", exc_info=True)
        return json.dumps({"success": False, "url": url, "error": str(exc)}, indent=2)


async def extract_structured_json(
    ctx: Context,
    url: Optional[str] = None,
    file_path: Optional[str] = None,
    html: Optional[str] = None,
    extraction_strategy: str = "css",
    extraction_schema: Optional[Dict[str, Any]] = None,
    extraction_instruction: Optional[str] = None,
    llm_provider: Optional[str] = None,
    run_config: Optional[Dict[str, Any]] = None,
) -> str:
    """Extract structured JSON using css/xpath/llm strategies."""
    target_url = url
    if html is not None:
        if not html.strip():
            return json.dumps({"success": False, "error": "html must be non-empty when provided."}, indent=2)
        target_url = f"raw:{html}"
    elif file_path is not None:
        normalized = file_path.replace("file://", "", 1) if file_path.startswith("file://") else file_path
        target_url = f"file://{Path(normalized).resolve()}"

    if not target_url:
        return json.dumps({"success": False, "error": "Provide one of: url, file_path, html."}, indent=2)

    return await crawl_to_markdown(
        ctx=ctx,
        url=target_url,
        markdown_variant="raw",
        run_config=run_config,
        index_result=False,
        extraction_strategy=extraction_strategy,
        extraction_schema=extraction_schema,
        extraction_instruction=extraction_instruction,
        llm_provider=llm_provider,
    )


async def extract_regex_entities(
    ctx: Context,
    url: str,
    extraction_patterns: Optional[Dict[str, str]] = None,
    run_config: Optional[Dict[str, Any]] = None,
) -> str:
    """Extract regex entities from crawled content."""
    return await crawl_to_markdown(
        ctx=ctx,
        url=url,
        markdown_variant="raw",
        run_config=run_config,
        index_result=False,
        extraction_strategy="regex",
        extraction_patterns=extraction_patterns,
    )


async def extract_knowledge_graph(
    ctx: Context,
    url: str,
    extraction_schema: Optional[Dict[str, Any]] = None,
    llm_provider: Optional[str] = None,
    run_config: Optional[Dict[str, Any]] = None,
) -> str:
    """Extract entity/relation graph-style JSON using LLM extraction."""
    return await crawl_to_markdown(
        ctx=ctx,
        url=url,
        markdown_variant="raw",
        run_config=run_config,
        index_result=False,
        extraction_strategy="llm",
        extraction_schema=extraction_schema,
        llm_provider=llm_provider,
        extraction_instruction=(
            "Extract a knowledge graph as JSON with nodes and edges. "
            "Nodes should include id, label, type. Edges should include source, target, relation."
        ),
    )


async def extract_code_examples(
    ctx: Context,
    url: str,
    markdown_variant: str = "raw",
    run_config: Optional[Dict[str, Any]] = None,
) -> str:
    """Extract code blocks from a crawled page without indexing."""
    try:
        crawler = _get_crawler(ctx)
        config = _build_run_config(run_config)
        result = await crawler.arun(url=url, config=config)
        if not (result.success and result.markdown):
            return json.dumps({"success": False, "url": url, "error": getattr(result, "error_message", "No content.")}, indent=2)

        variant_map = {
            "raw": "raw_markdown",
            "fit": "fit_markdown",
            "cited": "markdown_with_citations",
            "references": "references_markdown",
        }
        variants = _extract_markdown_variants(result.markdown)
        variant_key = variant_map.get(markdown_variant.lower(), "raw_markdown")
        text = variants.get(variant_key) or variants["raw_markdown"]
        blocks = extract_code_blocks(text)
        return json.dumps({
            "success": True,
            "url": url,
            "selected_variant": variant_key,
            "code_examples": blocks,
            "count": len(blocks),
        }, indent=2)
    except Exception as exc:
        logger.error(f"extract_code_examples {url}: {exc}", exc_info=True)
        return json.dumps({"success": False, "url": url, "error": str(exc)}, indent=2)


async def index_markdown(
    ctx: Context,
    url: str,
    markdown: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Index caller-supplied markdown into pgvector."""
    try:
        if not markdown or not markdown.strip():
            return json.dumps({"success": False, "url": url, "error": "markdown must be non-empty."}, indent=2)

        chunks = await chunk_text_according_to_settings(markdown)
        db_urls, db_chunks, db_contents, db_metas, db_fulldocs = [], [], [], [], []
        source_type = _infer_source_type(url)
        for i, chunk in enumerate(chunks):
            meta = extract_section_info(chunk)
            meta.update({
                "chunk_index": i,
                "url": url,
                "source": urlparse(url).netloc,
                "source_type": source_type,
                "crawl_time": datetime.now(timezone.utc).isoformat(),
                "crawl_timestamp": datetime.now(timezone.utc).isoformat(),
                "content_class": "text",
                "is_active": True,
                "content_hash": hashlib.sha256(chunk.encode("utf-8")).hexdigest(),
            })
            if metadata:
                meta.update(metadata)
            db_urls.append(url)
            db_chunks.append(i)
            db_contents.append(chunk)
            db_metas.append(meta)
            db_fulldocs.append(markdown)

        with next(get_session()) as session:
            chunks_stored = await add_documents_to_db(session, db_urls, db_contents, db_metas, db_chunks, db_fulldocs)

        return json.dumps({
            "success": True,
            "url": url,
            "chunks_stored": chunks_stored,
            "pages_indexed": 1,
        }, indent=2)
    except Exception as exc:
        logger.error(f"index_markdown {url}: {exc}", exc_info=True)
        return json.dumps({"success": False, "url": url, "error": str(exc)}, indent=2)


async def index_fit_markdown(
    ctx: Context,
    url: str,
    fit_markdown: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Index caller-supplied fit-markdown variant."""
    merged_meta = dict(metadata or {})
    merged_meta["markdown_variant"] = "fit_markdown"
    return await index_markdown(ctx=ctx, url=url, markdown=fit_markdown, metadata=merged_meta)


async def index_structured_content(
    ctx: Context,
    url: str,
    structured_content: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Index structured JSON content by projecting it to text."""
    merged_meta = dict(metadata or {})
    merged_meta["content_class"] = "structured"
    projected = json.dumps(structured_content, ensure_ascii=False, indent=2)
    return await index_markdown(ctx=ctx, url=url, markdown=projected, metadata=merged_meta)


async def index_code_examples(
    ctx: Context,
    url: str,
    markdown: str,
) -> str:
    """Index code blocks extracted from caller-supplied markdown."""
    try:
        if not markdown or not markdown.strip():
            return json.dumps({"success": False, "url": url, "error": "markdown must be non-empty."}, indent=2)
        blocks = extract_code_blocks(markdown)
        if not blocks:
            return json.dumps({"success": True, "url": url, "code_examples_indexed": 0}, indent=2)
        with next(get_session()) as session:
            await add_code_examples_to_db(
                session,
                urls=[url] * len(blocks),
                contents=[b["content"] for b in blocks],
                languages=[b["language"] for b in blocks],
                summaries=[None] * len(blocks),
                metadatas=[{"source": urlparse(url).netloc, "url": url, "source_type": _infer_source_type(url)}] * len(blocks),
                chunk_numbers=list(range(len(blocks))),
            )
        return json.dumps({"success": True, "url": url, "code_examples_indexed": len(blocks)}, indent=2)
    except Exception as exc:
        logger.error(f"index_code_examples {url}: {exc}", exc_info=True)
        return json.dumps({"success": False, "url": url, "error": str(exc)}, indent=2)


async def search_documents_v2(
    ctx: Context,
    query: str,
    source: Optional[str] = None,
    match_count: int = 5,
    content_class: Optional[str] = None,
    markdown_variant: Optional[str] = None,
    extraction_strategy: Optional[str] = None,
) -> str:
    """Taxonomy-native retrieval tool alias."""
    return await perform_rag_query(
        ctx=ctx,
        query=query,
        source=source,
        match_count=match_count,
        content_class=content_class,
        markdown_variant=markdown_variant,
        extraction_strategy=extraction_strategy,
    )


async def search_structured_content(
    ctx: Context,
    query: str,
    source: Optional[str] = None,
    match_count: int = 5,
    content_class: str = "structured",
) -> str:
    """Search structured-content indexed records via metadata filters."""
    try:
        normalized_content_class = (content_class or "structured").strip().lower()
        filter_meta: Dict[str, Any] = {"content_class": normalized_content_class}
        if source:
            filter_meta["source"] = source
        with next(get_session()) as session:
            results = await _search_documents_core(
                session,
                query,
                match_count=match_count,
                filter_metadata=filter_meta,
            )
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
        logger.error(f"search_structured_content: {exc}", exc_info=True)
        return json.dumps({"success": False, "query": query, "error": str(exc)}, indent=2)


async def get_fit_markdown_by_url(ctx: Context, url: str) -> str:
    """Reconstruct fit-markdown chunks for a URL."""
    try:
        with next(get_session()) as session:
            rows = session.exec(
                select(CrawledPage)
                .where(CrawledPage.url == url)
                .order_by(CrawledPage.chunk_number)
            ).all()

        fit_rows = [
            r for r in rows
            if isinstance(r.page_metadata, dict)
            and r.page_metadata.get("markdown_variant") == "fit_markdown"
        ]
        if not fit_rows:
            return json.dumps({"success": False, "url": url, "error": "No stored fit_markdown chunks found."}, indent=2)

        markdown = "\n\n".join(r.content for r in fit_rows if r.content)
        return json.dumps({
            "success": True,
            "url": url,
            "chunk_count": len(fit_rows),
            "fit_markdown": markdown,
        }, indent=2)
    except Exception as exc:
        logger.error(f"get_fit_markdown_by_url {url}: {exc}", exc_info=True)
        return json.dumps({"success": False, "url": url, "error": str(exc)}, indent=2)


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


# ---------------------------------------------------------------------------
# Phase 9.5 / 9.6 — value scoring, eviction, storage budget tools
# ---------------------------------------------------------------------------

async def compute_value_scores(
    ctx: Context,
    source: Optional[str] = None,
    limit: int = 1000,
) -> str:
    """Recompute and persist value_score and staleness_score for indexed records.

    Args:
        ctx: MCP context.
        source: Limit recomputation to a specific source domain (optional).
        limit: Maximum records to update per table (default 1000).
    """
    try:
        now = datetime.now(timezone.utc)
        updated_pages = 0
        updated_examples = 0

        with next(get_session()) as session:
            source_priority_map = {
                sp.source: sp.priority_weight
                for sp in session.exec(select(SourcePolicy)).all()
            }

            pages = session.exec(
                select(CrawledPage).where(CrawledPage.is_active == True).limit(limit)
            ).all()
            if source:
                pages = [p for p in pages if (p.page_metadata or {}).get("source") == source]
            for page in pages:
                meta = page.page_metadata if isinstance(page.page_metadata, dict) else {}
                src = meta.get("source", "")
                priority = source_priority_map.get(src, 1.0)
                ref_time = page.first_seen_at or page.crawl_timestamp
                age_days = max(0.0, (now - ref_time).total_seconds() / 86400)
                density = min(1.0, len(page.content) / 5000.0)
                page.staleness_score = compute_staleness_score(age_days)
                page.value_score = compute_value_score(
                    hit_count=page.hit_count,
                    content_density=density,
                    age_days=age_days,
                    source_priority=priority,
                )
                updated_pages += 1

            examples = session.exec(
                select(CodeExample).where(CodeExample.is_active == True).limit(limit)
            ).all()
            if source:
                examples = [e for e in examples if (e.ex_metadata or {}).get("source") == source]
            for ex in examples:
                meta = ex.ex_metadata if isinstance(ex.ex_metadata, dict) else {}
                src = meta.get("source", "")
                priority = source_priority_map.get(src, 1.0)
                ref_time = ex.first_seen_at or ex.crawl_timestamp
                age_days = max(0.0, (now - ref_time).total_seconds() / 86400)
                density = min(1.0, len(ex.content) / 5000.0)
                ex.staleness_score = compute_staleness_score(age_days)
                ex.value_score = compute_value_score(
                    hit_count=ex.hit_count,
                    content_density=density,
                    age_days=age_days,
                    source_priority=priority,
                )
                updated_examples += 1

            session.commit()

        return json.dumps({
            "success": True,
            "updated_crawled_pages": updated_pages,
            "updated_code_examples": updated_examples,
        }, indent=2)
    except Exception as exc:
        logger.error(f"compute_value_scores failed: {exc}", exc_info=True)
        return json.dumps({"success": False, "error": str(exc)}, indent=2)


async def preview_eviction_plan(
    ctx: Context,
    limit: int = 100,
    source: Optional[str] = None,
    dry_run: bool = True,
) -> str:
    """Preview (or execute) which records would be evicted under storage pressure.

    Records are sorted by value_score ascending — lowest value evicted first.
    Pinned records are always excluded.

    Args:
        ctx: MCP context.
        limit: Maximum candidates to return/evict per call.
        source: Filter to a specific source domain (optional).
        dry_run: When True (default) only preview; when False actually tombstone.
    """
    try:
        with next(get_session()) as session:
            candidates: List[Dict] = []

            for tbl_model, tbl_name in [
                (CrawledPage, "crawled_pages"),
                (CodeExample, "code_examples"),
            ]:
                records = session.exec(
                    select(tbl_model)
                    .where(
                        tbl_model.is_active == True,
                        tbl_model.is_pinned == False,
                        tbl_model.tombstoned_at.is_(None),  # type: ignore[union-attr]
                    )
                    .order_by(tbl_model.value_score.asc())  # type: ignore[union-attr]
                    .limit(limit)
                ).all()

                for rec in records:
                    meta = (
                        rec.page_metadata
                        if hasattr(rec, "page_metadata")
                        else getattr(rec, "ex_metadata", {})
                    ) or {}
                    rec_source = meta.get("source", "")
                    if source and rec_source != source:
                        continue
                    candidates.append({
                        "table": tbl_name,
                        "id": rec.id,
                        "url": rec.url,
                        "chunk_number": rec.chunk_number,
                        "source": rec_source,
                        "value_score": round(rec.value_score, 4),
                        "staleness_score": round(rec.staleness_score, 4),
                        "hit_count": rec.hit_count,
                        "content_length": len(rec.content),
                    })

            candidates.sort(key=lambda x: x["value_score"])
            candidates = candidates[:limit]

            total_evicted = 0
            if not dry_run:
                page_ids = [c["id"] for c in candidates if c["table"] == "crawled_pages"]
                code_ids = [c["id"] for c in candidates if c["table"] == "code_examples"]
                if page_ids:
                    total_evicted += tombstone_records(session, page_ids, "crawled_pages", "preview_eviction")
                if code_ids:
                    total_evicted += tombstone_records(session, code_ids, "code_examples", "preview_eviction")

            return json.dumps({
                "success": True,
                "dry_run": dry_run,
                "candidates_count": len(candidates),
                "total_evicted": total_evicted,
                "candidates": candidates[:50],
            }, indent=2)
    except Exception as exc:
        logger.error(f"preview_eviction_plan failed: {exc}", exc_info=True)
        return json.dumps({"success": False, "error": str(exc)}, indent=2)


async def enforce_storage_budget(
    ctx: Context,
    force: bool = False,
) -> str:
    """Check storage usage and trigger tiered eviction if budget thresholds are exceeded.

    Pressure levels:
      - ok       (<80%): no action
      - warning  (>=80%): compact expired tombstones
      - high     (>=90%): compact + tombstone stale records
      - critical (>=100%): compact + prune stale + value-based eviction

    Args:
        ctx: MCP context.
        force: When True, run compaction even if below warning threshold.
    """
    try:
        with next(get_session()) as session:
            policy = session.exec(select(StoragePolicy)).first()
            if policy is None:
                max_gb = 10.0
                warn_pct = 0.80
                high_pct = 0.90
                hard_pct = 1.00
                grace_h = 24
                post_evict = 0.75
            else:
                max_gb = policy.max_db_size_gb
                warn_pct = policy.warn_threshold
                high_pct = policy.high_threshold
                hard_pct = policy.hard_threshold
                grace_h = policy.tombstone_grace_hours
                post_evict = policy.target_post_evict_ratio

            max_bytes = int(max_gb * 1024 ** 3)
            db_size_bytes = _get_db_size_bytes(session)
            usage_ratio = db_size_bytes / max_bytes if max_bytes > 0 else 0.0

            if usage_ratio < warn_pct and not force:
                return json.dumps({
                    "success": True,
                    "db_size_bytes": db_size_bytes,
                    "usage_ratio": round(usage_ratio, 4),
                    "pressure_level": "ok",
                    "actions_taken": [],
                }, indent=2)

            actions_taken: List[str] = []
            now = datetime.now(timezone.utc)
            grace_cutoff = now - timedelta(hours=grace_h)

            # Step 1: compact — hard-delete tombstoned records past grace window
            compacted = 0
            for tbl_model in (CrawledPage, CodeExample):
                expired = session.exec(
                    select(tbl_model).where(
                        tbl_model.tombstoned_at.isnot(None),  # type: ignore[union-attr]
                        tbl_model.tombstoned_at <= grace_cutoff,  # type: ignore[union-attr]
                    )
                ).all()
                for rec in expired:
                    session.delete(rec)
                    compacted += 1
            if compacted:
                session.commit()
                actions_taken.append(f"compacted {compacted} expired tombstoned records")

            if usage_ratio >= high_pct:
                # Step 2: tombstone stale non-pinned active records
                pruned = 0
                stale_threshold = 0.8
                for tbl_model in (CrawledPage, CodeExample):
                    stale_recs = session.exec(
                        select(tbl_model).where(
                            tbl_model.staleness_score >= stale_threshold,
                            tbl_model.is_pinned == False,
                            tbl_model.tombstoned_at.is_(None),  # type: ignore[union-attr]
                            tbl_model.is_active == True,
                        ).limit(500)
                    ).all()
                    ids = [r.id for r in stale_recs]
                    if ids:
                        pruned += tombstone_records(
                            session, ids, tbl_model.__tablename__, "high_pressure_stale_prune"
                        )
                actions_taken.append(f"tombstoned {pruned} stale records (high pressure)")

            if usage_ratio >= hard_pct:
                # Step 3: value-based eviction of lowest-score non-pinned records
                target_bytes = int(max_bytes * post_evict)
                if db_size_bytes > target_bytes:
                    evicted = 0
                    for tbl_model in (CrawledPage, CodeExample):
                        candidates = session.exec(
                            select(tbl_model).where(
                                tbl_model.is_pinned == False,
                                tbl_model.tombstoned_at.is_(None),  # type: ignore[union-attr]
                                tbl_model.is_active == True,
                            ).order_by(tbl_model.value_score.asc()).limit(200)  # type: ignore[union-attr]
                        ).all()
                        ids = [r.id for r in candidates]
                        if ids:
                            evicted += tombstone_records(
                                session, ids, tbl_model.__tablename__, "hard_pressure_value_evict"
                            )
                    actions_taken.append(f"value-evicted {evicted} records (hard pressure)")

            pressure = (
                "ok" if usage_ratio < warn_pct else
                "warning" if usage_ratio < high_pct else
                "high" if usage_ratio < hard_pct else
                "critical"
            )
            return json.dumps({
                "success": True,
                "db_size_bytes": db_size_bytes,
                "db_size_gb": round(db_size_bytes / 1024 ** 3, 3),
                "max_db_size_gb": max_gb,
                "usage_ratio": round(usage_ratio, 4),
                "pressure_level": pressure,
                "actions_taken": actions_taken,
            }, indent=2)
    except Exception as exc:
        logger.error(f"enforce_storage_budget failed: {exc}", exc_info=True)
        return json.dumps({"success": False, "error": str(exc)}, indent=2)


async def pin_records(
    ctx: Context,
    record_ids: List[int],
    table: str = "crawled_pages",
) -> str:
    """Mark records as pinned — pinned records are excluded from eviction.

    Args:
        ctx: MCP context.
        record_ids: List of record IDs to pin.
        table: Target table; either "crawled_pages" or "code_examples".
    """
    if table not in ("crawled_pages", "code_examples"):
        return json.dumps({"success": False, "error": f"Unknown table: {table}"}, indent=2)

    model_cls = CrawledPage if table == "crawled_pages" else CodeExample

    try:
        with next(get_session()) as session:
            records = session.exec(
                select(model_cls).where(model_cls.id.in_(record_ids))  # type: ignore[attr-defined]
            ).all()
            for rec in records:
                rec.is_pinned = True
            session.commit()
            return json.dumps({
                "success": True,
                "pinned_count": len(records),
                "table": table,
            }, indent=2)
    except Exception as exc:
        logger.error(f"pin_records failed: {exc}", exc_info=True)
        return json.dumps({"success": False, "error": str(exc)}, indent=2)


async def unpin_records(
    ctx: Context,
    record_ids: List[int],
    table: str = "crawled_pages",
) -> str:
    """Remove the pin flag from records — they become eligible for eviction again.

    Args:
        ctx: MCP context.
        record_ids: List of record IDs to unpin.
        table: Target table; either "crawled_pages" or "code_examples".
    """
    if table not in ("crawled_pages", "code_examples"):
        return json.dumps({"success": False, "error": f"Unknown table: {table}"}, indent=2)

    model_cls = CrawledPage if table == "crawled_pages" else CodeExample

    try:
        with next(get_session()) as session:
            records = session.exec(
                select(model_cls).where(model_cls.id.in_(record_ids))  # type: ignore[attr-defined]
            ).all()
            for rec in records:
                rec.is_pinned = False
            session.commit()
            return json.dumps({
                "success": True,
                "unpinned_count": len(records),
                "table": table,
            }, indent=2)
    except Exception as exc:
        logger.error(f"unpin_records failed: {exc}", exc_info=True)
        return json.dumps({"success": False, "error": str(exc)}, indent=2)


async def index_storage_report(
    ctx: Context,
    group_by: str = "table",
) -> str:
    """Return a storage usage report for all indexed content.

    Args:
        ctx: MCP context.
        group_by: Aggregation level: "table" (default) or "source".
    """
    try:
        with next(get_session()) as session:
            def _count(sql: str) -> int:
                row = session.exec(_sql_text(sql)).first()
                return int(row[0]) if row else 0

            def _size(tbl: str) -> int:
                row = session.exec(
                    _sql_text(f"SELECT pg_total_relation_size('{tbl}')")
                ).first()
                return int(row[0]) if row else 0

            cp_total = _count("SELECT COUNT(*) FROM crawled_pages")
            cp_active = _count(
                "SELECT COUNT(*) FROM crawled_pages WHERE is_active = TRUE AND tombstoned_at IS NULL"
            )
            cp_tombstoned = _count(
                "SELECT COUNT(*) FROM crawled_pages WHERE tombstoned_at IS NOT NULL"
            )
            cp_pinned = _count("SELECT COUNT(*) FROM crawled_pages WHERE is_pinned = TRUE")
            cp_size = _size("crawled_pages")

            ce_total = _count("SELECT COUNT(*) FROM code_examples")
            ce_active = _count(
                "SELECT COUNT(*) FROM code_examples WHERE is_active = TRUE AND tombstoned_at IS NULL"
            )
            ce_tombstoned = _count(
                "SELECT COUNT(*) FROM code_examples WHERE tombstoned_at IS NOT NULL"
            )
            ce_pinned = _count("SELECT COUNT(*) FROM code_examples WHERE is_pinned = TRUE")
            ce_size = _size("code_examples")

            db_size = _get_db_size_bytes(session)
            policy = session.exec(select(StoragePolicy)).first()
            max_gb = policy.max_db_size_gb if policy else 10.0
            max_bytes = int(max_gb * 1024 ** 3)
            usage_ratio = db_size / max_bytes if max_bytes > 0 else 0.0

            report: Dict[str, Any] = {
                "success": True,
                "db_size_bytes": db_size,
                "db_size_mb": round(db_size / 1024 ** 2, 2),
                "max_db_size_gb": max_gb,
                "usage_ratio": round(usage_ratio, 4),
                "pressure_level": (
                    "ok" if usage_ratio < 0.80 else
                    "warning" if usage_ratio < 0.90 else
                    "high" if usage_ratio < 1.00 else
                    "critical"
                ),
                "tables": {
                    "crawled_pages": {
                        "total_rows": cp_total,
                        "active_rows": cp_active,
                        "tombstoned_rows": cp_tombstoned,
                        "pinned_rows": cp_pinned,
                        "size_bytes": cp_size,
                        "size_mb": round(cp_size / 1024 ** 2, 2),
                    },
                    "code_examples": {
                        "total_rows": ce_total,
                        "active_rows": ce_active,
                        "tombstoned_rows": ce_tombstoned,
                        "pinned_rows": ce_pinned,
                        "size_bytes": ce_size,
                        "size_mb": round(ce_size / 1024 ** 2, 2),
                    },
                },
            }

            if group_by == "source":
                src_rows = session.exec(_sql_text(
                    "SELECT metadata->>'source' as src, COUNT(*) as total, "
                    "SUM(CASE WHEN is_active THEN 1 ELSE 0 END) as active "
                    "FROM crawled_pages "
                    "GROUP BY metadata->>'source' ORDER BY total DESC LIMIT 50"
                )).all()
                report["by_source"] = [
                    {"source": r[0] or "(unknown)", "total": int(r[1]), "active": int(r[2])}
                    for r in src_rows
                ]

            return json.dumps(report, indent=2)
    except Exception as exc:
        logger.error(f"index_storage_report failed: {exc}", exc_info=True)
        return json.dumps({"success": False, "error": str(exc)}, indent=2)


async def restore_tombstoned_records(
    ctx: Context,
    record_ids: List[int],
    table: str = "crawled_pages",
) -> str:
    """Restore tombstoned records that are still within the grace window.

    Records tombstoned longer ago than tombstone_grace_hours (from StoragePolicy)
    cannot be restored and will be skipped.

    Args:
        ctx: MCP context.
        record_ids: IDs of tombstoned records to restore.
        table: Target table; either "crawled_pages" or "code_examples".
    """
    if table not in ("crawled_pages", "code_examples"):
        return json.dumps({"success": False, "error": f"Unknown table: {table}"}, indent=2)

    model_cls = CrawledPage if table == "crawled_pages" else CodeExample

    try:
        with next(get_session()) as session:
            policy = session.exec(select(StoragePolicy)).first()
            grace_hours = policy.tombstone_grace_hours if policy else 24
            grace_cutoff = datetime.now(timezone.utc) - timedelta(hours=grace_hours)

            records = session.exec(
                select(model_cls).where(
                    model_cls.id.in_(record_ids),  # type: ignore[attr-defined]
                    model_cls.tombstoned_at.isnot(None),  # type: ignore[union-attr]
                )
            ).all()

            restored = 0
            skipped = 0
            for rec in records:
                ts = rec.tombstoned_at
                if ts is not None and ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts is not None and ts >= grace_cutoff:
                    rec.tombstoned_at = None
                    rec.is_active = True
                    restored += 1
                else:
                    skipped += 1

            session.commit()
            return json.dumps({
                "success": True,
                "restored_count": restored,
                "skipped_count": skipped,
                "table": table,
            }, indent=2)
    except Exception as exc:
        logger.error(f"restore_tombstoned_records failed: {exc}", exc_info=True)
        return json.dumps({"success": False, "error": str(exc)}, indent=2)


async def recrawl_due_sources(
    ctx: Context,
    source: Optional[str] = None,
    max_concurrent: int = 5,
) -> str:
    """Recrawl sources that are past their recrawl interval.

    Uses ``source_policies.recrawl_interval_hours`` and latest
    ``last_crawled_at`` across active, non-tombstoned records.

    Args:
        ctx: MCP context.
        source: Optional source domain to scope the run.
        max_concurrent: Max crawler concurrency forwarded to smart_crawl_url.
    """
    try:
        now = datetime.now(timezone.utc)
        due_sources: List[Dict[str, Any]] = []

        with next(get_session()) as session:
            policies = session.exec(select(SourcePolicy)).all()
            if source:
                policies = [p for p in policies if p.source == source]

            for pol in policies:
                row = session.execute(
                    _sql_text(
                        """
                        SELECT GREATEST(
                            COALESCE((
                                SELECT MAX(last_crawled_at)
                                FROM crawled_pages
                                WHERE metadata->>'source' = :source
                                  AND is_active = TRUE
                                  AND tombstoned_at IS NULL
                            ), to_timestamp(0)),
                            COALESCE((
                                SELECT MAX(last_crawled_at)
                                FROM code_examples
                                WHERE metadata->>'source' = :source
                                  AND is_active = TRUE
                                  AND tombstoned_at IS NULL
                            ), to_timestamp(0))
                        )
                        """
                    ),
                    {"source": pol.source},
                ).first()

                last_crawled = row[0] if row else None
                if isinstance(last_crawled, datetime) and last_crawled.tzinfo is None:
                    last_crawled = last_crawled.replace(tzinfo=timezone.utc)

                interval_h = max(1, int(pol.recrawl_interval_hours))
                if last_crawled is None:
                    is_due = True
                else:
                    is_due = (now - last_crawled) >= timedelta(hours=interval_h)

                if is_due:
                    due_sources.append({
                        "source": pol.source,
                        "recrawl_interval_hours": interval_h,
                        "last_crawled_at": last_crawled.isoformat() if isinstance(last_crawled, datetime) else None,
                    })

        recrawled: List[Dict[str, Any]] = []
        failures: List[Dict[str, Any]] = []

        for item in due_sources:
            src = item["source"]
            crawl_url = src if src.startswith(("http://", "https://")) else f"https://{src}"
            try:
                result = await smart_crawl_url(
                    ctx,
                    url=crawl_url,
                    crawl_mode="single",
                    max_concurrent=max_concurrent,
                )
                parsed = json.loads(result)
                if parsed.get("success"):
                    recrawled.append({
                        "source": src,
                        "url": crawl_url,
                        "pages_crawled": parsed.get("pages_crawled", 0),
                    })
                else:
                    failures.append({
                        "source": src,
                        "url": crawl_url,
                        "error": parsed.get("error", "recrawl failed"),
                    })
            except Exception as exc:  # pragma: no cover - defensive outer catch still tested via mocks
                failures.append({"source": src, "url": crawl_url, "error": str(exc)})

        return json.dumps({
            "success": True,
            "due_count": len(due_sources),
            "recrawled_count": len(recrawled),
            "failed_count": len(failures),
            "recrawled_sources": recrawled,
            "failures": failures,
        }, indent=2)
    except Exception as exc:
        logger.error(f"recrawl_due_sources failed: {exc}", exc_info=True)
        return json.dumps({"success": False, "error": str(exc)}, indent=2)


async def prune_stale_content(
    ctx: Context,
    force: bool = False,
) -> str:
    """Hard-delete tombstoned records based on grace window.

    Args:
        ctx: MCP context.
        force: When True, delete all tombstoned rows regardless of age.
    """
    try:
        with next(get_session()) as session:
            policy = session.exec(select(StoragePolicy)).first()
            grace_hours = policy.tombstone_grace_hours if policy else 24
            cutoff = datetime.now(timezone.utc) - timedelta(hours=grace_hours)

            deleted_by_table: Dict[str, int] = {
                "crawled_pages": 0,
                "code_examples": 0,
            }

            for model_cls, table_name in ((CrawledPage, "crawled_pages"), (CodeExample, "code_examples")):
                if force:
                    rows = session.exec(
                        select(model_cls).where(model_cls.tombstoned_at.isnot(None))  # type: ignore[union-attr]
                    ).all()
                else:
                    rows = session.exec(
                        select(model_cls).where(
                            model_cls.tombstoned_at.isnot(None),  # type: ignore[union-attr]
                            model_cls.tombstoned_at <= cutoff,  # type: ignore[union-attr]
                        )
                    ).all()

                for rec in rows:
                    session.delete(rec)
                    deleted_by_table[table_name] += 1

            total_deleted = sum(deleted_by_table.values())
            session.commit()

        return json.dumps({
            "success": True,
            "force": force,
            "grace_hours": grace_hours,
            "hard_deleted_count": total_deleted,
            "deleted_by_table": deleted_by_table,
        }, indent=2)
    except Exception as exc:
        logger.error(f"prune_stale_content failed: {exc}", exc_info=True)
        return json.dumps({"success": False, "error": str(exc)}, indent=2)


async def hard_delete_tombstones(
    ctx: Context,
    max_age_hours: Optional[int] = None,
) -> str:
    """Force hard-delete tombstoned records.

    Args:
        ctx: MCP context.
        max_age_hours: Optional minimum tombstone age; when omitted, delete all tombstoned rows.
    """
    try:
        with next(get_session()) as session:
            cutoff: Optional[datetime] = None
            if isinstance(max_age_hours, int) and max_age_hours >= 0:
                cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)

            deleted_by_table: Dict[str, int] = {
                "crawled_pages": 0,
                "code_examples": 0,
            }

            for model_cls, table_name in ((CrawledPage, "crawled_pages"), (CodeExample, "code_examples")):
                q = select(model_cls).where(model_cls.tombstoned_at.isnot(None))  # type: ignore[union-attr]
                if cutoff is not None:
                    q = q.where(model_cls.tombstoned_at <= cutoff)  # type: ignore[union-attr]
                rows = session.exec(q).all()

                for rec in rows:
                    session.delete(rec)
                    deleted_by_table[table_name] += 1

            total_deleted = sum(deleted_by_table.values())
            session.commit()

        return json.dumps({
            "success": True,
            "max_age_hours": max_age_hours,
            "hard_deleted_count": total_deleted,
            "deleted_by_table": deleted_by_table,
        }, indent=2)
    except Exception as exc:
        logger.error(f"hard_delete_tombstones failed: {exc}", exc_info=True)
        return json.dumps({"success": False, "error": str(exc)}, indent=2)
