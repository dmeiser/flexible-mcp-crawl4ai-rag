"""MCP tool definitions — registered in src/crawl4ai_mcp.py."""
import json
import logging
import hashlib
import re
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
    create_embedding,
    extract_code_blocks,
    compute_value_score,
    compute_staleness_score,
    tombstone_records,
    _get_db_size_bytes,
    MarkdownIndexPolicy,
    extract_link_references,
)
from .web_crawler import (
    crawl_recursive_internal_links,
    chunk_text_according_to_settings,
)
from .metadata_extractor import (
    extract_section_info,
    extract_link_graph,
    extract_media_metadata,
)
from .postgres_client import store_crawled_documents
from .url_scorers import build_url_scorer, get_supported_scorer_types

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
_ALLOWED_SCORER_TYPES = set(get_supported_scorer_types()) | {"none"}
_ALLOWED_INDEX_VARIANT_OVERRIDES = {"raw", "fit", "both"}
_ALLOWED_STRUCTURED_PROJECTION_MODES = {"hybrid", "raw_json", "flattened_text"}
_ALLOWED_SCHEMA_STRATEGIES = {"css", "xpath"}
_SCHEMA_CACHE_DIR = Path(".cache/generated_extraction_schemas")


def _normalize_markdown_index_policy(policy_value: Any) -> str:
    """Normalize configured markdown index policy to supported string values."""
    if isinstance(policy_value, MarkdownIndexPolicy):
        return policy_value.value
    normalized = str(policy_value or "").strip().lower()
    if normalized in {
        MarkdownIndexPolicy.RAW_ONLY.value,
        MarkdownIndexPolicy.FIT_ONLY.value,
        MarkdownIndexPolicy.BOTH_BY_DEFAULT.value,
    }:
        return normalized
    return MarkdownIndexPolicy.BOTH_BY_DEFAULT.value


def _normalize_index_variants_override(index_variants: Optional[str]) -> Optional[str]:
    """Normalize per-run override for indexed markdown variants."""
    if not isinstance(index_variants, str):
        return None
    normalized = index_variants.strip().lower()
    if normalized in _ALLOWED_INDEX_VARIANT_OVERRIDES:
        return normalized
    return None


def _resolve_variants_to_index(
    variants: Dict[str, str],
    index_variants: Optional[str] = None,
    fallback_enabled: bool = True,
) -> tuple[List[str], str, Optional[str], List[str]]:
    """Resolve markdown variants to index based on global policy + per-run override.

    Returns tuple of:
    - resolved variant keys to index (raw_markdown / fit_markdown)
    - effective policy string
    - normalized per-run override (or None)
    - fallback notes
    """
    override = _normalize_index_variants_override(index_variants)
    effective_policy = _normalize_markdown_index_policy(settings.MARKDOWN_INDEX_POLICY)

    if override == "raw":
        preferred = ["raw_markdown"]
    elif override == "fit":
        preferred = ["fit_markdown"]
    elif override == "both":
        preferred = ["raw_markdown", "fit_markdown"]
    elif effective_policy == MarkdownIndexPolicy.RAW_ONLY.value:
        preferred = ["raw_markdown"]
    elif effective_policy == MarkdownIndexPolicy.FIT_ONLY.value:
        preferred = ["fit_markdown"]
    else:
        preferred = ["raw_markdown", "fit_markdown"]

    resolved: List[str] = [variant for variant in preferred if variants.get(variant)]
    fallback_notes: List[str] = []

    if missing := [variant for variant in preferred if variant not in resolved]:
        if fallback_enabled:
            if "raw_markdown" in missing and variants.get("fit_markdown") and "fit_markdown" not in resolved:
                resolved.append("fit_markdown")
                fallback_notes.append("raw_markdown unavailable; indexed fit_markdown instead")
            if "fit_markdown" in missing and variants.get("raw_markdown") and "raw_markdown" not in resolved:
                resolved.append("raw_markdown")
                fallback_notes.append("fit_markdown unavailable; indexed raw_markdown instead")
        else:
            fallback_notes.append("fallback disabled; unavailable requested variants were skipped")

    # Deterministic ordering for downstream assertions and responses.
    ordered = [variant for variant in ("raw_markdown", "fit_markdown") if variant in resolved]
    return ordered, effective_policy, override, fallback_notes


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


def _build_reference_metadata(variant_values: Optional[Dict[str, str]]) -> Dict[str, Any]:
    """Build persisted citation/reference metadata for indexed chunks."""
    variants = variant_values or {}
    references_markdown = variants.get("references_markdown") or ""
    return {
        "references_markdown": references_markdown,
        "link_references": extract_link_references(references_markdown),
        "has_citations": bool(variants.get("markdown_with_citations")),
    }


def _build_requested_provenance(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Build caller-facing provenance info from stored metadata."""
    meta = metadata if isinstance(metadata, dict) else {}
    return {
        "source": meta.get("source"),
        "url": meta.get("url"),
        "source_type": meta.get("source_type"),
        "crawl_type": meta.get("crawl_type"),
        "crawl_timestamp": meta.get("crawl_timestamp") or meta.get("crawl_time"),
        "session_id": meta.get("session_id"),
        "markdown_variant": meta.get("markdown_variant"),
        "extraction_strategy": meta.get("extraction_strategy"),
        "references_markdown": meta.get("references_markdown") or "",
        "link_references": meta.get("link_references") if isinstance(meta.get("link_references"), list) else [],
        "has_link_references": bool(meta.get("references_markdown") or meta.get("link_references")),
        "has_citations": bool(meta.get("has_citations", False)),
    }


def _select_rows_for_variant(rows: List[Any], preferred_variant: str) -> List[Any]:
    """Prefer rows matching a specific markdown variant, with fallback to all rows."""
    preferred_rows = [
        row for row in rows
        if isinstance(getattr(row, "page_metadata", None), dict)
        and row.page_metadata.get("markdown_variant") == preferred_variant
    ]
    return preferred_rows or rows


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
        re.compile(link_filter)
        return link_filter
    except Exception as e:
        logger.warning(f"Invalid link_filter regex '{link_filter}': {e}")
        return None


def _flatten_structured_content(value: Any, prefix: str = "") -> List[str]:
    """Flatten nested structured content to deterministic path=value lines."""
    if value is None:
        return []
    if isinstance(value, dict):
        lines: List[str] = []
        for key in sorted(value.keys()):
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            lines.extend(_flatten_structured_content(value[key], next_prefix))
        return lines
    if isinstance(value, list):
        lines: List[str] = []
        for idx, item in enumerate(value):
            next_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            lines.extend(_flatten_structured_content(item, next_prefix))
        return lines
    value_str = str(value).strip()
    if not value_str:
        return []
    return [f"{prefix}={value_str}" if prefix else value_str]


def _normalize_extraction_records(extracted_content: Any) -> List[Dict[str, Any]]:
    """Normalize extraction outputs to a stable records list contract."""
    if extracted_content is None:
        return []
    if isinstance(extracted_content, list):
        records: List[Dict[str, Any]] = []
        for item in extracted_content:
            if isinstance(item, dict):
                records.append(item)
            elif item is not None:
                records.append({"value": item})
        return records
    if isinstance(extracted_content, dict):
        return [extracted_content]
    return [{"value": extracted_content}]


def _normalize_structured_contract(
    *,
    strategy: str,
    extracted_content: Any,
    source_type: str,
    source_value: str,
    schema: Optional[Dict[str, Any]] = None,
    fit_source_used: bool = False,
) -> Dict[str, Any]:
    """Build a normalized structured extraction response contract."""
    records = _normalize_extraction_records(extracted_content)
    flattened_projection = "\n".join(_flatten_structured_content(records))
    return {
        "contract_version": "1.0",
        "source": {
            "type": source_type,
            "value": source_value,
            "fit_source_used": fit_source_used,
        },
        "extraction": {
            "strategy": strategy,
            "schema": schema,
            "record_count": len(records),
        },
        "data": {
            "records": records,
            "raw": extracted_content,
        },
        "indexing": {
            "recommended_content_class": "structured",
            "recommended_model": "hybrid_json_vector_v1",
            "flattened_projection": flattened_projection,
        },
    }


def _build_schema_from_sample_html(sample_html: str, strategy: str = "css") -> Dict[str, Any]:
    """Generate a lightweight extraction schema from sample HTML."""
    normalized_strategy = (strategy or "css").strip().lower()
    selector_for = "xpath" if normalized_strategy == "xpath" else "selector"

    fields: List[Dict[str, Any]] = []
    candidates = [
        ("title", "title", "/html/head/title/text()"),
        ("heading", "h1", "//h1[1]/text()"),
        ("paragraph", "p", "//p[1]/text()"),
        ("link", "a", "//a[1]/@href"),
    ]

    for name, css_selector, xpath_selector in candidates:
        pattern = rf"<\s*{css_selector}\b" if css_selector != "title" else r"<\s*title\b"
        if not re.search(pattern, sample_html, flags=re.IGNORECASE):
            continue
        field: Dict[str, Any] = {"name": name, "type": "text"}
        field[selector_for] = xpath_selector if normalized_strategy == "xpath" else css_selector
        fields.append(field)

    if not fields:
        fallback_field: Dict[str, Any] = {"name": "content", "type": "text"}
        fallback_field[selector_for] = "//body" if normalized_strategy == "xpath" else "body"
        fields.append(fallback_field)

    schema: Dict[str, Any] = {"fields": fields}
    if normalized_strategy == "css":
        schema["baseSelector"] = "body"
    return schema


def _validate_generated_schema(schema: Optional[Dict[str, Any]], strategy: str = "css") -> Dict[str, Any]:
    """Validate extraction schema structure for css/xpath strategies."""
    normalized_strategy = (strategy or "css").strip().lower()
    if normalized_strategy not in _ALLOWED_SCHEMA_STRATEGIES:
        return {"valid": False, "errors": ["Only css/xpath schemas are supported."], "normalized_schema": None}

    if not isinstance(schema, dict):
        return {"valid": False, "errors": ["schema must be a dictionary."], "normalized_schema": None}

    fields = schema.get("fields")
    if not isinstance(fields, list) or not fields:
        return {"valid": False, "errors": ["schema.fields must be a non-empty list."], "normalized_schema": None}

    selector_key = "xpath" if normalized_strategy == "xpath" else "selector"
    normalized_fields: List[Dict[str, Any]] = []
    errors: List[str] = []

    for index, field in enumerate(fields):
        if not isinstance(field, dict):
            errors.append(f"fields[{index}] must be an object")
            continue
        name = str(field.get("name") or "").strip()
        selector = str(field.get(selector_key) or "").strip()
        if not name:
            errors.append(f"fields[{index}].name is required")
        if not selector:
            errors.append(f"fields[{index}].{selector_key} is required")
        if name and selector:
            normalized_fields.append({
                "name": name,
                selector_key: selector,
                "type": str(field.get("type") or "text"),
            })

    if errors:
        return {"valid": False, "errors": errors, "normalized_schema": None}

    normalized_schema: Dict[str, Any] = {"fields": normalized_fields}
    if normalized_strategy == "css":
        normalized_schema["baseSelector"] = str(schema.get("baseSelector") or "body")
    return {"valid": True, "errors": [], "normalized_schema": normalized_schema}


def _cache_generated_schema(
    schema: Dict[str, Any],
    sample_html: str,
    strategy: str,
    schema_name: Optional[str] = None,
) -> str:
    """Persist generated schema for deterministic reuse."""
    _SCHEMA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    normalized_strategy = (strategy or "css").strip().lower()
    cache_seed = f"{normalized_strategy}:{sample_html}:{json.dumps(schema, sort_keys=True)}"
    default_name = f"{normalized_strategy}_{hashlib.sha256(cache_seed.encode('utf-8')).hexdigest()[:16]}.json"
    target_name = schema_name.strip() if isinstance(schema_name, str) and schema_name.strip() else default_name
    if not target_name.endswith(".json"):
        target_name = f"{target_name}.json"
    target_path = _SCHEMA_CACHE_DIR / Path(target_name).name
    target_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(target_path)


def _project_structured_content(
    structured_content: Any,
    projection_mode: str,
) -> tuple[str, Dict[str, Any]]:
    """Project structured content for vector indexing while preserving raw JSON."""
    normalized_mode = projection_mode.strip().lower() if isinstance(projection_mode, str) else "hybrid"
    if normalized_mode not in _ALLOWED_STRUCTURED_PROJECTION_MODES:
        normalized_mode = "hybrid"

    raw_json_text = json.dumps(structured_content, ensure_ascii=False, indent=2)
    flattened_lines = _flatten_structured_content(structured_content)
    flattened_text = "\n".join(flattened_lines)

    if normalized_mode == "raw_json":
        return raw_json_text, {"projection_mode": normalized_mode, "flattened_text": flattened_text}
    if normalized_mode == "flattened_text":
        return flattened_text or raw_json_text, {"projection_mode": normalized_mode, "raw_json": structured_content}
    return flattened_text or raw_json_text, {
        "projection_mode": "hybrid",
        "raw_json": structured_content,
        "flattened_text": flattened_text,
        "indexing_model": "hybrid_json_vector_v1",
    }


def _build_adaptive_knowledge_base_export(
    crawled_docs: List[Dict[str, Any]],
    format_name: str = "json",
) -> Dict[str, Any]:
    """Build exportable adaptive knowledge base payload in common formats."""
    normalized_format = (format_name or "json").strip().lower()
    if normalized_format not in {"json", "jsonl", "markdown"}:
        normalized_format = "json"

    records = [
        {
            "url": doc.get("url"),
            "depth": doc.get("depth"),
            "selected_variant": doc.get("selected_variant"),
            "markdown": doc.get("markdown"),
        }
        for doc in crawled_docs
    ]

    if normalized_format == "jsonl":
        return {
            "format": "jsonl",
            "record_count": len(records),
            "data": "\n".join(json.dumps(row, ensure_ascii=False) for row in records),
        }

    if normalized_format == "markdown":
        sections: List[str] = []
        for idx, row in enumerate(records, start=1):
            sections.append(f"## {idx}. {row.get('url')}")
            sections.append(f"- depth: {row.get('depth')}")
            sections.append(f"- variant: {row.get('selected_variant')}")
            sections.append("")
            sections.append(row.get("markdown") or "")
            sections.append("")
        return {
            "format": "markdown",
            "record_count": len(records),
            "data": "\n".join(sections).strip(),
        }

    return {
        "format": "json",
        "record_count": len(records),
        "data": records,
    }


def _build_adaptive_answer(query: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate deterministic adaptive answer bundle from retrieved rows."""
    if not rows:
        return {
            "query": query,
            "answer": "No relevant content found in adaptive crawl results.",
            "supporting_results": [],
        }

    top = rows[0]
    snippet = (top.get("content") or "").strip()
    answer = snippet[:600] + ("..." if len(snippet) > 600 else "")
    supporting = [
        {
            "url": row.get("url"),
            "similarity": row.get("similarity_score"),
            "metadata": row.get("page_metadata") if isinstance(row.get("page_metadata"), dict) else {},
        }
        for row in rows
    ]
    return {
        "query": query,
        "answer": answer,
        "supporting_results": supporting,
    }


def _normalize_session_id(session_id: Optional[str]) -> Optional[str]:
    """Normalize a user-supplied session_id; return None when empty/invalid."""
    if session_id is None:
        return None
    if not isinstance(session_id, str):
        return None
    normalized = session_id.strip()
    return normalized or None


def _normalize_run_id(run_id: Optional[str]) -> Optional[str]:
    """Normalize a user-supplied run_id; return None when empty/invalid."""
    if run_id is None:
        return None
    if not isinstance(run_id, str):
        return None
    normalized = run_id.strip()
    return normalized or None


def _generate_run_id(prefix: str = "crawl") -> str:
    """Generate a deterministic-enough run id for crawl/index operations."""
    now_iso = datetime.now(timezone.utc).isoformat()
    digest = hashlib.sha1(now_iso.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}-{digest}"


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


def _canonical_url_key(url: Any) -> str:
    """Return a deterministic canonical URL key for safeguard grouping."""
    if not isinstance(url, str) or not url.strip():
        return ""
    parsed = urlparse(url.strip())
    if parsed.scheme or parsed.netloc:
        cleaned = parsed._replace(fragment="")
        return cleaned.geturl()
    return parsed.path or url.strip()


def _extract_source_change_id(crawl_result: Any) -> Optional[str]:
    """Extract optional ETag/Last-Modified identifier from crawl result headers."""
    headers = getattr(crawl_result, "response_headers", None)
    if not isinstance(headers, dict):
        headers = getattr(crawl_result, "headers", None)
    if not isinstance(headers, dict):
        return None

    normalized = {str(k).lower(): str(v) for k, v in headers.items() if k is not None and v is not None}
    etag = normalized.get("etag")
    if etag:
        return f"etag:{etag}"
    last_modified = normalized.get("last-modified")
    if last_modified:
        return f"last-modified:{last_modified}"
    return None


def _parse_datetime_utc(value: Any) -> Optional[datetime]:
    """Parse datetime-like value into UTC datetime when possible."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip())
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _is_result_fresh(
    metadata: Dict[str, Any],
    staleness_threshold: float,
    as_of_dt: Optional[datetime],
    require_fresh: bool = True,
) -> bool:
    """Determine whether a result passes freshness criteria."""
    if require_fresh:
        staleness_value = metadata.get("staleness_score")
        if isinstance(staleness_value, (int, float)) and float(staleness_value) > staleness_threshold:
            return False

        expires_at = _parse_datetime_utc(metadata.get("expires_at"))
        if expires_at is not None and expires_at <= datetime.now(timezone.utc):
            return False

    if as_of_dt is not None:
        crawl_ts = _parse_datetime_utc(metadata.get("crawl_timestamp") or metadata.get("crawl_time"))
        if crawl_ts is not None and crawl_ts > as_of_dt:
            return False

    return True


def _compute_freshness_from_metadata(metadata: Dict[str, Any]) -> float:
    """Compute a bounded freshness score [0,1] using metadata fallbacks."""
    staleness_value = metadata.get("staleness_score")
    if isinstance(staleness_value, (int, float)):
        return max(0.0, min(1.0, 1.0 - float(staleness_value)))

    crawl_ts = _parse_datetime_utc(metadata.get("crawl_timestamp") or metadata.get("crawl_time"))
    if crawl_ts is None:
        return 0.5
    age_days = max(0.0, (datetime.now(timezone.utc) - crawl_ts).total_seconds() / 86400)
    return max(0.0, min(1.0, float(pow(2.718281828, -age_days / 90.0))))


def _eviction_sort_key(candidate: Dict[str, Any]) -> tuple[float, float, int, datetime]:
    """Deterministic candidate ordering:
    1) lower value_score
    2) higher staleness_score
    3) lower hit_count
    4) older last_seen_at
    """
    last_seen = _parse_datetime_utc(candidate.get("last_seen_at")) or datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (
        float(candidate.get("value_score") or 0.0),
        -float(candidate.get("staleness_score") or 0.0),
        int(candidate.get("hit_count") or 0),
        last_seen,
    )


def _apply_min_active_docs_safeguard(
    candidates: List[Dict[str, Any]],
    source_policy_map: Dict[str, int],
) -> List[Dict[str, Any]]:
    """Filter candidates so each source keeps min_active_docs records (best-effort)."""
    if not candidates:
        return []

    active_counts: Dict[str, int] = {}
    for c in candidates:
        src = str(c.get("source") or "")
        active_counts[src] = active_counts.get(src, 0) + 1

    selected_by_source: Dict[str, int] = {}
    filtered: List[Dict[str, Any]] = []
    for c in sorted(candidates, key=_eviction_sort_key):
        src = str(c.get("source") or "")
        min_docs = int(source_policy_map.get(src, 0))
        selected_count = selected_by_source.get(src, 0)
        if min_docs > 0 and (active_counts.get(src, 0) - selected_count) <= min_docs:
            continue
        filtered.append(c)
        selected_by_source[src] = selected_count + 1

    return filtered


def _build_active_coverage_maps(session: Any) -> tuple[Dict[str, int], Dict[tuple[str, str], int]]:
    """Build active coverage maps for source and canonical URL safeguards."""
    source_counts: Dict[str, int] = {}
    canonical_counts: Dict[tuple[str, str], int] = {}

    for model_cls in (CrawledPage, CodeExample):
        rows = session.exec(
            select(model_cls).where(
                model_cls.is_active == True,
                model_cls.tombstoned_at.is_(None),  # type: ignore[union-attr]
            )
        ).all()

        for row in rows:
            metadata = (
                row.page_metadata
                if hasattr(row, "page_metadata") and isinstance(row.page_metadata, dict)
                else getattr(row, "ex_metadata", {})
            ) or {}
            source = str(metadata.get("source") or "")
            canonical = _canonical_url_key(metadata.get("canonical_url") or row.url)
            source_counts[source] = source_counts.get(source, 0) + 1
            key = (source, canonical)
            canonical_counts[key] = canonical_counts.get(key, 0) + 1

    return source_counts, canonical_counts


def _apply_eviction_safeguards(
    candidates: List[Dict[str, Any]],
    source_policy_map: Dict[str, int],
    source_active_counts: Dict[str, int],
    canonical_active_counts: Dict[tuple[str, str], int],
) -> List[Dict[str, Any]]:
    """Apply min_active_docs + last-representation safeguards with deterministic ordering."""
    if not candidates:
        return []

    projected_source_counts = dict(source_active_counts)
    projected_canonical_counts = dict(canonical_active_counts)

    selected: List[Dict[str, Any]] = []
    for candidate in sorted(candidates, key=_eviction_sort_key):
        source = str(candidate.get("source") or "")
        min_docs = int(source_policy_map.get(source, 0))
        remaining_for_source = projected_source_counts.get(source, 0) - 1
        if min_docs > 0 and remaining_for_source < min_docs:
            continue

        canonical_guard_enabled = bool(candidate.get("canonical_guard", False))
        canonical_key = _canonical_url_key(candidate.get("canonical_key") or candidate.get("url"))
        source_canonical_key = (source, canonical_key)
        remaining_for_canonical = projected_canonical_counts.get(source_canonical_key, 0) - 1
        if canonical_guard_enabled and canonical_key and remaining_for_canonical < 1:
            continue

        selected.append(candidate)
        projected_source_counts[source] = max(0, remaining_for_source)
        if canonical_guard_enabled and canonical_key:
            projected_canonical_counts[source_canonical_key] = max(0, remaining_for_canonical)

    return selected


def _record_metadata(record: Any) -> Dict[str, Any]:
    """Extract metadata dict from crawled page/code example records."""
    if hasattr(record, "page_metadata") and isinstance(record.page_metadata, dict):
        return record.page_metadata
    candidate = getattr(record, "ex_metadata", {})
    return candidate if isinstance(candidate, dict) else {}


def _estimate_record_size_bytes(record: Any) -> int:
    """Estimate a record's storage footprint for policy budgeting."""
    content = getattr(record, "content", "") or ""
    if not isinstance(content, str):
        content = str(content)
    metadata = _record_metadata(record)
    metadata_size = len(json.dumps(metadata, ensure_ascii=False)) if metadata else 0
    return len(content.encode("utf-8")) + metadata_size


def _is_dead_page_error(error_message: str) -> bool:
    """Detect dead-page style failures (404/410 or explicit markers)."""
    if not isinstance(error_message, str):
        return False
    lowered = error_message.lower()
    return (
        "404" in lowered
        or "410" in lowered
        or "not found" in lowered
        or "gone" in lowered
    )


def _compute_retry_backoff_hours(policy: Any) -> int:
    """Compute exponential retry backoff using per-source policy defaults."""
    failures = int(getattr(policy, "consecutive_failures", 0) or 0)
    base = max(1, int(getattr(policy, "retry_backoff_base_hours", 2) or 2))
    max_backoff = max(base, int(getattr(policy, "max_retry_backoff_hours", 168) or 168))
    if failures <= 0:
        return 0
    return min(max_backoff, base * (2 ** max(0, failures - 1)))


def _enforce_source_quotas(session: Any, source_policies: Dict[str, Any]) -> Dict[str, Any]:
    """Enforce per-source quotas by tombstoning lowest-value rows first."""
    if not source_policies:
        return {"quota_evicted": 0, "sources_over_quota": []}

    rows_by_source: Dict[str, List[Dict[str, Any]]] = {}
    for model_cls, table_name in ((CrawledPage, "crawled_pages"), (CodeExample, "code_examples")):
        rows = session.exec(
            select(model_cls).where(
                model_cls.is_active == True,
                model_cls.tombstoned_at.is_(None),  # type: ignore[union-attr]
                model_cls.is_pinned == False,
            )
        ).all()
        for row in rows:
            metadata = _record_metadata(row)
            src = str(metadata.get("source") or "")
            if src not in source_policies:
                continue
            rows_by_source.setdefault(src, []).append(
                {
                    "table": table_name,
                    "id": row.id,
                    "value_score": float(getattr(row, "value_score", 0.0) or 0.0),
                    "staleness_score": float(getattr(row, "staleness_score", 0.0) or 0.0),
                    "hit_count": int(getattr(row, "hit_count", 0) or 0),
                    "last_seen_at": (
                        _parse_datetime_utc(getattr(row, "last_seen_at", None)).isoformat()
                        if _parse_datetime_utc(getattr(row, "last_seen_at", None)) is not None
                        else None
                    ),
                    "size_bytes": _estimate_record_size_bytes(row),
                }
            )

    over_quota_sources: List[str] = []
    total_evicted = 0
    for source, policy in source_policies.items():
        quota_mb = getattr(policy, "max_source_size_mb", None)
        if not isinstance(quota_mb, int) or quota_mb <= 0:
            continue
        quota_bytes = quota_mb * 1024 * 1024
        records = rows_by_source.get(source, [])
        used = sum(r["size_bytes"] for r in records)
        if used <= quota_bytes:
            continue

        over_quota_sources.append(source)
        selected: List[Dict[str, Any]] = []
        for candidate in sorted(records, key=_eviction_sort_key):
            if used <= quota_bytes:
                break
            selected.append(candidate)
            used -= candidate["size_bytes"]

        page_ids = [c["id"] for c in selected if c["table"] == "crawled_pages"]
        code_ids = [c["id"] for c in selected if c["table"] == "code_examples"]
        if page_ids:
            total_evicted += tombstone_records(session, page_ids, "crawled_pages", "source_quota_prune")
        if code_ids:
            total_evicted += tombstone_records(session, code_ids, "code_examples", "source_quota_prune")

    return {"quota_evicted": total_evicted, "sources_over_quota": over_quota_sources}


def _enforce_table_budgets(
    session: Any,
    max_crawled_pages_mb: Optional[int],
    max_code_examples_mb: Optional[int],
) -> Dict[str, int]:
    """Enforce per-table budgets using approximate content+metadata size."""
    budget_map = {
        "crawled_pages": max_crawled_pages_mb,
        "code_examples": max_code_examples_mb,
    }
    evicted_by_table = {"crawled_pages": 0, "code_examples": 0}

    for model_cls, table_name in ((CrawledPage, "crawled_pages"), (CodeExample, "code_examples")):
        max_mb = budget_map.get(table_name)
        if not isinstance(max_mb, int) or max_mb <= 0:
            continue
        limit_bytes = max_mb * 1024 * 1024
        rows = session.exec(
            select(model_cls).where(
                model_cls.is_active == True,
                model_cls.tombstoned_at.is_(None),  # type: ignore[union-attr]
                model_cls.is_pinned == False,
            )
        ).all()
        current_bytes = sum(_estimate_record_size_bytes(r) for r in rows)
        if current_bytes <= limit_bytes:
            continue

        candidates = [
            {
                "id": r.id,
                "value_score": float(getattr(r, "value_score", 0.0) or 0.0),
                "staleness_score": float(getattr(r, "staleness_score", 0.0) or 0.0),
                "hit_count": int(getattr(r, "hit_count", 0) or 0),
                "last_seen_at": (
                    _parse_datetime_utc(getattr(r, "last_seen_at", None)).isoformat()
                    if _parse_datetime_utc(getattr(r, "last_seen_at", None)) is not None
                    else None
                ),
                "size_bytes": _estimate_record_size_bytes(r),
            }
            for r in rows
        ]

        to_tombstone: List[int] = []
        for c in sorted(candidates, key=_eviction_sort_key):
            if current_bytes <= limit_bytes:
                break
            to_tombstone.append(c["id"])
            current_bytes -= c["size_bytes"]

        if to_tombstone:
            evicted_by_table[table_name] += tombstone_records(
                session, to_tombstone, table_name, f"table_budget_prune:{table_name}"
            )

    return evicted_by_table


def _apply_hard_ttl_delete(session: Any, source_policies: Dict[str, Any]) -> Dict[str, int]:
    """Hard-delete active records that exceeded per-source TTL."""
    now = datetime.now(timezone.utc)
    deleted = {"crawled_pages": 0, "code_examples": 0}

    for model_cls, table_name in ((CrawledPage, "crawled_pages"), (CodeExample, "code_examples")):
        rows = session.exec(
            select(model_cls).where(
                model_cls.is_active == True,
                model_cls.tombstoned_at.is_(None),  # type: ignore[union-attr]
            )
        ).all()

        for row in rows:
            metadata = _record_metadata(row)
            source = str(metadata.get("source") or "")
            policy = source_policies.get(source)
            ttl_days = int(getattr(policy, "ttl_days", 90)) if policy is not None else 90
            if ttl_days <= 0:
                continue
            ttl_boundary = timedelta(days=ttl_days)
            reference_time = (
                _parse_datetime_utc(getattr(row, "last_seen_at", None))
                or _parse_datetime_utc(getattr(row, "first_seen_at", None))
                or _parse_datetime_utc(getattr(row, "crawl_timestamp", None))
                or now
            )
            if now - reference_time > ttl_boundary:
                session.delete(row)
                deleted[table_name] += 1

    if deleted["crawled_pages"] or deleted["code_examples"]:
        session.commit()
    return deleted


def _retire_source_duplicates_and_superseded(session: Any, source: str) -> Dict[str, int]:
    """Retire duplicate/superseded active records for a source."""
    duplicate_page_ids: List[int] = []
    duplicate_code_ids: List[int] = []
    superseded_page_ids: List[int] = []
    superseded_code_ids: List[int] = []

    for model_cls, dup_target, sup_target in (
        (CrawledPage, duplicate_page_ids, superseded_page_ids),
        (CodeExample, duplicate_code_ids, superseded_code_ids),
    ):
        rows = session.exec(
            select(model_cls).where(
                model_cls.is_active == True,
                model_cls.tombstoned_at.is_(None),  # type: ignore[union-attr]
            )
        ).all()
        scoped = []
        for row in rows:
            metadata = _record_metadata(row)
            if str(metadata.get("source") or "") != source:
                continue
            scoped.append((row, metadata))

        by_canonical: Dict[str, List[Any]] = {}
        for row, metadata in scoped:
            canonical = _canonical_url_key(metadata.get("canonical_url") or row.url)
            by_canonical.setdefault(canonical, []).append((row, metadata))

        for canonical_rows in by_canonical.values():
            canonical_rows_sorted = sorted(
                canonical_rows,
                key=lambda pair: (
                    _parse_datetime_utc(getattr(pair[0], "last_crawled_at", None))
                    or datetime(1970, 1, 1, tzinfo=timezone.utc),
                    float(getattr(pair[0], "value_score", 0.0) or 0.0),
                    int(getattr(pair[0], "hit_count", 0) or 0),
                ),
                reverse=True,
            )
            survivor = canonical_rows_sorted[0][0]
            seen_hashes = {str(getattr(survivor, "content_hash", "") or "")}

            for row, _metadata in canonical_rows_sorted[1:]:
                row_hash = str(getattr(row, "content_hash", "") or "")
                if row_hash and row_hash in seen_hashes:
                    dup_target.append(row.id)
                else:
                    sup_target.append(row.id)
                    if row_hash:
                        seen_hashes.add(row_hash)

    retired_duplicates = 0
    retired_superseded = 0
    if duplicate_page_ids:
        retired_duplicates += tombstone_records(session, duplicate_page_ids, "crawled_pages", "duplicate_content_hash")
    if duplicate_code_ids:
        retired_duplicates += tombstone_records(session, duplicate_code_ids, "code_examples", "duplicate_content_hash")
    if superseded_page_ids:
        retired_superseded += tombstone_records(session, superseded_page_ids, "crawled_pages", "superseded_canonical_url")
    if superseded_code_ids:
        retired_superseded += tombstone_records(session, superseded_code_ids, "code_examples", "superseded_canonical_url")

    return {
        "duplicate_retired": retired_duplicates,
        "superseded_retired": retired_superseded,
    }


async def _run_selective_reembed(session: Any, row_ids: List[int]) -> int:
    """Re-embed selected crawled page records and refresh freshness metadata."""
    if not row_ids:
        return 0

    rows = session.exec(select(CrawledPage).where(CrawledPage.id.in_(row_ids))).all()
    updated = 0
    now = datetime.now(timezone.utc)
    for row in rows:
        content = getattr(row, "content", "")
        if not isinstance(content, str) or not content.strip():
            continue
        try:
            row.embedding = await create_embedding(content)
            row.last_crawled_at = now
            row.staleness_score = 0.0
            updated += 1
        except Exception as exc:  # pragma: no cover - external provider errors are non-deterministic
            logger.warning(f"selective re-embed failed for row id={row.id}: {exc}")
    if updated:
        session.commit()
    return updated


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
    scorer = build_url_scorer(scorer_mode, keywords=keywords)

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


async def crawl_to_markdown(
    ctx: Context,
    url: str,
    markdown_variant: str = "raw",
    run_config: Optional[Dict[str, Any]] = None,
    index_result: bool = False,
    index_variants: Optional[str] = None,
    extraction_strategy: Optional[str] = None,
    extraction_schema: Optional[Dict[str, Any]] = None,
    extraction_patterns: Optional[Dict[str, str]] = None,
    extraction_instruction: Optional[str] = None,
    llm_provider: Optional[str] = None,
    session_id: Optional[str] = None,
    run_id: Optional[str] = None,
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
        effective_run_id = _normalize_run_id(run_id) or _generate_run_id("crawl")
        
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
            logger.warning(
                "Using compatibility helper crawl_recursive_internal_links for follow_links mode; "
                "prefer crawl_deep for native Crawl4AI deep crawling strategies."
            )
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
        indexed_variant_keys: set[str] = set()
        fallback_notes: List[str] = []
        resolved_override = _normalize_index_variants_override(index_variants)
        if index_result and isinstance(index_variants, str) and resolved_override is None:
            fallback_notes.append(
                "invalid index_variants override; using global markdown index policy"
            )
        
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
                "variant_values": variants,
                "depth": getattr(r, "depth", 0),  # May not be available in simple crawl
                "raw_result": r,
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
                    variant_values: Dict[str, str] = doc.get("variant_values", {})
                    doc_link_graph = extract_link_graph(doc.get("markdown", ""), base_url=doc.get("url"))
                    doc_media_metadata = extract_media_metadata(doc.get("markdown", ""))
                    source_change_id = _extract_source_change_id(doc.get("raw_result"))
                    variant_keys_to_index, effective_policy, _, doc_fallback_notes = _resolve_variants_to_index(
                        variants=variant_values,
                        index_variants=index_variants,
                        fallback_enabled=bool(settings.MARKDOWN_FALLBACK_ENABLED),
                    )
                    fallback_notes.extend(doc_fallback_notes)

                    for variant_to_index in variant_keys_to_index:
                        selected_variant_markdown = variant_values.get(variant_to_index) or ""
                        if not selected_variant_markdown:
                            continue
                        chunks = await chunk_text_according_to_settings(selected_variant_markdown)
                        indexed_variant_keys.add(variant_to_index)
                        reference_meta = _build_reference_metadata(variant_values)
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
                                "markdown_variant": variant_to_index,
                                "content_class": "text",
                                "is_active": True,
                                "content_hash": hashlib.sha256(chunk.encode("utf-8")).hexdigest(),
                                "max_depth": max_depth,
                                "follow_links": follow_links,
                                "depth": doc.get("depth", 0),
                                "session_id": _normalize_session_id(session_id),
                                "run_id": effective_run_id,
                                "source_change_id": source_change_id,
                                "markdown_index_policy": effective_policy,
                                "index_variants_override": resolved_override,
                                "link_graph": doc_link_graph,
                                "media_metadata": doc_media_metadata,
                            })
                            meta.update(reference_meta)
                            if extraction_strategy:
                                meta["extraction_strategy"] = extraction_strategy
                            db_urls.append(doc["url"])
                            db_chunks.append(i)
                            db_contents.append(chunk)
                            db_metas.append(meta)
                            db_fulldocs.append(selected_variant_markdown)
                
                if db_urls:
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
            "indexed_variants": sorted(indexed_variant_keys),
            "index_variants_override": resolved_override,
            "markdown_index_policy": _normalize_markdown_index_policy(settings.MARKDOWN_INDEX_POLICY),
            "index_fallback_enabled": bool(settings.MARKDOWN_FALLBACK_ENABLED),
            "index_fallback_notes": fallback_notes,
            "extraction_strategy_applied": extraction_strategy or None,
            "session_id_applied": _normalize_session_id(session_id),
            "run_id": effective_run_id,
            "markdown_options_applied": bool(markdown_options),
            "content_source_applied": (content_source if content_source in _ALLOWED_CONTENT_SOURCES else "cleaned_html"),
            "content_filter_applied": (content_filter.lower().strip() if isinstance(content_filter, str) and content_filter.lower().strip() in _ALLOWED_CONTENT_FILTERS else None),
            "max_depth_configured": max_depth,
            "follow_links_enabled": follow_links,
            "deep_crawl_mode": "compatibility_recursive" if (follow_links and max_depth > 1) else "single_page",
            "compatibility_helper_used": bool(follow_links and max_depth > 1),
            "link_filter_applied": bool(validated_link_filter),
            "artifacts": {
                "screenshot": _json_safe_artifact(getattr(first_result, "screenshot", None) if first_result else None),
                "pdf": _json_safe_artifact(getattr(first_result, "pdf", None) if first_result else None),
                "mhtml": _json_safe_artifact(getattr(first_result, "mhtml", None) if first_result else None),
            },
            "link_graph": extract_link_graph(first_markdown, base_url=url),
            "media_metadata": extract_media_metadata(first_markdown),
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
    run_id: Optional[str] = None,
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
        effective_run_id = _normalize_run_id(run_id) or _generate_run_id("crawl-many")
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
            logger.warning(
                "Using compatibility helper crawl_recursive_internal_links in crawl_many_urls; "
                "prefer crawl_deep for native Crawl4AI deep crawling strategies."
            )
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
                    doc = {
                        "url": r.url,
                        "markdown": selected,
                        "depth": getattr(r, "depth", 0),
                        "variant_values": variants,
                        "selected_variant": variant_key,
                        "link_graph": extract_link_graph(selected, base_url=r.url),
                        "media_metadata": extract_media_metadata(selected),
                        "source_change_id": _extract_source_change_id(r),
                        "run_id": effective_run_id,
                    }
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
            "run_id": effective_run_id,
            "markdown_options_applied": bool(markdown_options),
            "content_source_applied": (content_source if content_source in _ALLOWED_CONTENT_SOURCES else "cleaned_html"),
            "content_filter_applied": (content_filter.lower().strip() if isinstance(content_filter, str) and content_filter.lower().strip() in _ALLOWED_CONTENT_FILTERS else None),
            "max_depth_configured": max_depth,
            "follow_links_enabled": follow_links,
            "link_filter_applied": bool(validated_link_filter),
            "compatibility_helper_used": bool(follow_links and max_depth > 1),
            "link_graph": extract_link_graph(crawled_docs[0]["markdown"], base_url=crawled_docs[0]["url"]) if crawled_docs else {},
            "media_metadata": extract_media_metadata(crawled_docs[0]["markdown"]) if crawled_docs else {},
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


async def ingest_content_directory(
    ctx: Context,
    directory_path: str,
    include_patterns: Optional[List[str]] = None,
    markdown_variant: str = "raw",
    index_result: bool = True,
) -> str:
    """Ingestion wrapper for offline docs/testing/generated HTML directories."""
    try:
        root = Path(directory_path).expanduser().resolve()
        if not root.exists() or not root.is_dir():
            return json.dumps({"success": False, "error": "directory_path must point to an existing directory."}, indent=2)

        default_patterns = ["**/*.md", "**/*.markdown", "**/*.html", "**/*.htm"]
        patterns = include_patterns if isinstance(include_patterns, list) and include_patterns else default_patterns

        discovered: List[Path] = []
        for pattern in patterns:
            if not isinstance(pattern, str) or not pattern.strip():
                continue
            discovered.extend(root.glob(pattern.strip()))

        # deterministic + unique
        files = sorted({p for p in discovered if p.is_file()})
        if not files:
            return json.dumps(
                {
                    "success": True,
                    "directory": str(root),
                    "files_discovered": 0,
                    "files_processed": 0,
                    "indexed_count": 0,
                    "errors": [],
                },
                indent=2,
            )

        processed = 0
        indexed_count = 0
        errors: List[Dict[str, str]] = []

        for path in files:
            try:
                suffix = path.suffix.lower()
                if suffix in {".html", ".htm"}:
                    html = path.read_text(encoding="utf-8", errors="ignore")
                    result = await crawl_raw_html(
                        ctx=ctx,
                        html=html,
                        markdown_variant=markdown_variant,
                        index_result=index_result,
                    )
                    data = json.loads(result)
                    if data.get("success"):
                        indexed_count += int(data.get("pages_indexed", 0) or 0)
                        processed += 1
                    else:
                        errors.append({"file": str(path), "error": data.get("error", "crawl_raw_html failed")})
                    continue

                markdown_text = path.read_text(encoding="utf-8", errors="ignore")
                if index_result:
                    file_url = f"file://{path}"
                    result = await index_markdown(
                        ctx=ctx,
                        url=file_url,
                        markdown=markdown_text,
                        metadata={"source_type": "local_file", "ingestion_wrapper": "directory"},
                    )
                    data = json.loads(result)
                    if data.get("success"):
                        indexed_count += 1
                        processed += 1
                    else:
                        errors.append({"file": str(path), "error": data.get("error", "index_markdown failed")})
                else:
                    processed += 1
            except Exception as file_exc:  # pragma: no cover - defensive branch
                errors.append({"file": str(path), "error": str(file_exc)})

        return json.dumps(
            {
                "success": True,
                "directory": str(root),
                "files_discovered": len(files),
                "files_processed": processed,
                "indexed_count": indexed_count,
                "errors": errors,
            },
            indent=2,
        )
    except Exception as exc:
        logger.error(f"ingest_content_directory failed: {exc}", exc_info=True)
        return json.dumps({"success": False, "error": str(exc)}, indent=2)


async def get_document_by_id(ctx: Context, document_id: int, include_provenance: bool = False) -> str:
    """Fetch a stored document chunk by primary key id."""
    try:
        with next(get_session()) as session:
            row = session.exec(select(CrawledPage).where(CrawledPage.id == document_id)).first()

        if not row:
            return json.dumps({"success": False, "document_id": document_id, "error": "Document not found."}, indent=2)

        payload = {
            "success": True,
            "document": {
                "id": row.id,
                "url": row.url,
                "chunk_number": row.chunk_number,
                "content": row.content,
                "metadata": row.page_metadata if isinstance(row.page_metadata, dict) else {},
            },
        }
        if include_provenance:
            payload["document"]["provenance"] = _build_requested_provenance(row.page_metadata)

        return json.dumps(payload, indent=2)
    except Exception as exc:
        logger.error(f"get_document_by_id {document_id}: {exc}", exc_info=True)
        return json.dumps({"success": False, "document_id": document_id, "error": str(exc)}, indent=2)


async def get_markdown_by_url(ctx: Context, url: str, include_provenance: bool = False) -> str:
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

        selected_rows = _select_rows_for_variant(rows, preferred_variant="raw_markdown")
        markdown = "\n\n".join(r.content for r in selected_rows if r.content)
        payload = {
            "success": True,
            "url": url,
            "chunk_count": len(selected_rows),
            "markdown": markdown,
            "selected_variant": (
                selected_rows[0].page_metadata.get("markdown_variant")
                if selected_rows and isinstance(selected_rows[0].page_metadata, dict)
                else None
            ),
        }
        if include_provenance and selected_rows:
            payload["provenance"] = _build_requested_provenance(selected_rows[0].page_metadata)

        return json.dumps(payload, indent=2)
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
    index_variants: Optional[str] = None,
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
    """Unified crawl entrypoint that dispatches to markdown/deep modes."""
    selected_mode = (mode or "markdown").lower().strip()

    if selected_mode == "markdown":
        return await crawl_to_markdown(
            ctx=ctx,
            url=url,
            markdown_variant=markdown_variant,
            run_config=run_config,
            index_result=index_result,
            index_variants=index_variants,
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

    if selected_mode in {"smart", "legacy", "single", "single_legacy"}:
        return json.dumps(
            {
                "success": False,
                "url": url,
                "error": (
                    f"Mode '{selected_mode}' has been removed. "
                    "Use mode='markdown' (or mode='deep' for deep crawling) instead."
                ),
                "mode": selected_mode,
            },
            indent=2,
        )

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
            "error": "Invalid mode. Use one of: markdown, deep.",
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
    run_id: Optional[str] = None,
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
        effective_run_id = _normalize_run_id(run_id) or _generate_run_id("crawl-deep")

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
                            "variant_values": variants,
                            "selected_variant": variant_key,
                            "source_change_id": _extract_source_change_id(result),
                            "run_id": effective_run_id,
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
                            "variant_values": variants,
                            "selected_variant": variant_key,
                            "source_change_id": _extract_source_change_id(result),
                            "run_id": effective_run_id,
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
            "run_id": effective_run_id,
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
    export_knowledge_base: bool = False,
    knowledge_base_format: str = "json",
    answer_query: Optional[str] = None,
    answer_match_count: int = 3,
    run_id: Optional[str] = None,
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
        effective_run_id = _normalize_run_id(run_id) or _generate_run_id("crawl-adaptive")

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
                            "variant_values": variants,
                            "selected_variant": variant_key,
                            "source_change_id": _extract_source_change_id(result),
                            "run_id": effective_run_id,
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

        adaptive_answer: Optional[Dict[str, Any]] = None
        if isinstance(answer_query, str) and answer_query.strip():
            if index_result and pages_indexed > 0:
                with next(get_session()) as session:
                    scoped_rows = await search_documents(
                        session,
                        answer_query.strip(),
                        match_count=max(1, min(10, answer_match_count)),
                        filter_metadata={"crawl_type": f"adaptive_{strategy_lower}_{variant_key}"},
                    )
                adaptive_answer = _build_adaptive_answer(answer_query.strip(), scoped_rows)
            else:
                pseudo_rows = [
                    {
                        "url": doc.get("url"),
                        "content": doc.get("markdown"),
                        "page_metadata": {
                            "markdown_variant": doc.get("selected_variant"),
                            "depth": doc.get("depth"),
                            "crawl_type": f"adaptive_{strategy_lower}_{variant_key}",
                        },
                        "similarity_score": 0.0,
                    }
                    for doc in crawled_docs[: max(1, min(10, answer_match_count))]
                ]
                adaptive_answer = _build_adaptive_answer(answer_query.strip(), pseudo_rows)

        kb_export = None
        if export_knowledge_base:
            kb_export = _build_adaptive_knowledge_base_export(
                crawled_docs,
                format_name=knowledge_base_format,
            )

        payload = {
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
                "confidence_threshold": max(0.0, min(1.0, confidence_threshold)),
                "stopped_when_confident": bool(adaptive.confidence >= max(0.0, min(1.0, confidence_threshold))),
                "coverage_stats": adaptive.coverage_stats,
                "relevant_content": relevant,
                "errors": errors,
                "run_id": effective_run_id,
                "urls_crawled_sample": [d["url"] for d in crawled_docs[:5]]
                + (["..."] if len(crawled_docs) > 5 else []),
        }
        if kb_export is not None:
            payload["knowledge_base_export"] = kb_export
        if adaptive_answer is not None:
            payload["adaptive_answer"] = adaptive_answer

        return json.dumps(payload, indent=2)

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
    run_id: Optional[str] = None,
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
            run_id=run_id,
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
        run_id=run_id,
    )


async def crawl_with_auth_hooks(
    ctx: Context,
    url: str,
    session_id: str,
    markdown_variant: str = "raw",
    index_result: bool = False,
    run_config: Optional[Dict[str, Any]] = None,
    custom_headers: Optional[Dict[str, str]] = None,
    cookies: Optional[List[Dict[str, Any]]] = None,
    local_storage: Optional[Dict[str, str]] = None,
    route_block_patterns: Optional[List[str]] = None,
    pre_navigation_js: Optional[str] = None,
    final_scroll: bool = False,
    post_navigation_js: Optional[str] = None,
) -> str:
    """Auth-capable crawl wrapper with safe hook-like controls.

    Direct exposure:
    - custom headers/cookies
    - pre/post navigation JS hooks
    - route/resource block patterns (via c4a_script hints)
    - optional final-page scroll hook

    Preset wrappers (crawl_login_required/crawl_paginated) compose this tool.
    """
    normalized_session = _normalize_session_id(session_id)
    if not normalized_session:
        return json.dumps({"success": False, "error": "session_id must be a non-empty string."}, indent=2)

    safe_headers = {
        str(k): str(v)
        for k, v in (custom_headers or {}).items()
        if isinstance(k, str) and isinstance(v, str)
    }
    safe_cookies = cookies if isinstance(cookies, list) else []

    hook_script_parts: List[str] = []
    if isinstance(pre_navigation_js, str) and pre_navigation_js.strip():
        hook_script_parts.append(pre_navigation_js.strip())
    if isinstance(local_storage, dict) and local_storage:
        for key, value in local_storage.items():
            hook_script_parts.append(
                f"window.localStorage.setItem({json.dumps(str(key))}, {json.dumps(str(value))});"
            )
    if final_scroll:
        hook_script_parts.append("window.scrollTo(0, document.body.scrollHeight);")
    if isinstance(post_navigation_js, str) and post_navigation_js.strip():
        hook_script_parts.append(post_navigation_js.strip())

    merged_run_config = dict(run_config or {})
    if hook_script_parts:
        merged_run_config["js_code_before_wait"] = "\n".join(hook_script_parts)
    if route_block_patterns:
        # Keep this safe + declarative: pass pattern hints to c4a_script for compatible runtimes.
        merged_run_config["c4a_script"] = {
            "route_block_patterns": [p for p in route_block_patterns if isinstance(p, str) and p.strip()],
        }

    browser_override: Dict[str, Any] = {}
    if safe_headers:
        browser_override["headers"] = safe_headers
    if safe_cookies:
        browser_override["cookies"] = safe_cookies

    crawl_result = await crawl_with_browser_config(
        ctx=ctx,
        url=url,
        browser_config=browser_override or None,
        markdown_variant=markdown_variant,
        index_result=index_result,
        run_config=_merge_run_config_with_session(merged_run_config, normalized_session),
    )
    payload = json.loads(crawl_result)
    payload["auth_hooks_applied"] = {
        "session_id": normalized_session,
        "custom_headers": bool(safe_headers),
        "cookies": bool(safe_cookies),
        "local_storage": bool(local_storage),
        "route_block_patterns": bool(route_block_patterns),
        "final_scroll": bool(final_scroll),
        "pre_navigation_js": bool(pre_navigation_js),
        "post_navigation_js": bool(post_navigation_js),
    }
    payload["workflow_mode"] = "direct_auth_hooks"
    return json.dumps(payload, indent=2)


async def crawl_login_required(
    ctx: Context,
    url: str,
    session_id: str,
    login_script: Optional[str] = None,
    markdown_variant: str = "raw",
    index_result: bool = False,
    custom_headers: Optional[Dict[str, str]] = None,
    cookies: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Preset wrapper for login-required documentation crawls."""
    response = await crawl_with_auth_hooks(
        ctx=ctx,
        url=url,
        session_id=session_id,
        markdown_variant=markdown_variant,
        index_result=index_result,
        custom_headers=custom_headers,
        cookies=cookies,
        pre_navigation_js=login_script,
        final_scroll=True,
    )
    payload = json.loads(response)
    payload["workflow_mode"] = "login_required_preset"
    return json.dumps(payload, indent=2)


async def crawl_paginated(
    ctx: Context,
    start_url: str,
    session_id: str,
    additional_urls: Optional[List[str]] = None,
    markdown_variant: str = "raw",
    index_result: bool = False,
    max_concurrent: int = 5,
) -> str:
    """Preset wrapper for paginated/load-more style crawls."""
    normalized_session = _normalize_session_id(session_id)
    if not normalized_session:
        return json.dumps({"success": False, "error": "session_id must be a non-empty string."}, indent=2)

    urls = [start_url]
    for candidate in additional_urls or []:
        if isinstance(candidate, str) and candidate.strip() and candidate not in urls:
            urls.append(candidate.strip())

    result = await crawl_many_urls(
        ctx=ctx,
        urls=urls,
        max_concurrent=max(1, min(20, max_concurrent)),
        markdown_variant=markdown_variant,
        index_result=index_result,
        session_id=normalized_session,
    )
    payload = json.loads(result)
    payload["workflow_mode"] = "paginated_preset"
    payload["start_url"] = start_url
    payload["session_id_applied"] = normalized_session
    return json.dumps(payload, indent=2)


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
    index_variants: Optional[str] = None,
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
        indexed_variants: List[str] = []
        fallback_notes: List[str] = []
        resolved_override = _normalize_index_variants_override(index_variants)
        if index_result and isinstance(index_variants, str) and resolved_override is None:
            fallback_notes.append(
                "invalid index_variants override; using global markdown index policy"
            )
        if index_result:
            variant_keys_to_index, effective_policy, _, resolved_fallback_notes = _resolve_variants_to_index(
                variants=variants,
                index_variants=index_variants,
                fallback_enabled=bool(settings.MARKDOWN_FALLBACK_ENABLED),
            )
            fallback_notes.extend(resolved_fallback_notes)

            db_urls: List[str] = []
            db_chunks: List[int] = []
            db_contents: List[str] = []
            db_metas: List[Dict[str, Any]] = []
            db_fulldocs: List[str] = []

            for variant_key in variant_keys_to_index:
                markdown_value = variants.get(variant_key) or ""
                if not markdown_value:
                    continue
                indexed_variants.append(variant_key)
                chunks = await chunk_text_according_to_settings(markdown_value)
                reference_meta = _build_reference_metadata(variants)
                for i, chunk in enumerate(chunks):
                    meta = extract_section_info(chunk)
                    meta.update({
                        "chunk_index": i,
                        "url": url,
                        "source": urlparse(url).netloc,
                        "source_type": _infer_source_type(url),
                        "crawl_time": datetime.now(timezone.utc).isoformat(),
                        "crawl_timestamp": datetime.now(timezone.utc).isoformat(),
                        "markdown_variant": variant_key,
                        "content_class": "text",
                        "is_active": True,
                        "content_hash": hashlib.sha256(chunk.encode("utf-8")).hexdigest(),
                        "markdown_index_policy": effective_policy,
                        "index_variants_override": resolved_override,
                    })
                    meta.update(reference_meta)
                    db_urls.append(url)
                    db_chunks.append(i)
                    db_contents.append(chunk)
                    db_metas.append(meta)
                    db_fulldocs.append(markdown_value)

            if db_urls:
                with next(get_session()) as session:
                    chunks_stored = await add_documents_to_db(
                        session,
                        db_urls,
                        db_contents,
                        db_metas,
                        db_chunks,
                        db_fulldocs,
                    )

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
            "indexed_variants": indexed_variants,
            "index_variants_override": resolved_override,
            "markdown_index_policy": _normalize_markdown_index_policy(settings.MARKDOWN_INDEX_POLICY),
            "index_fallback_enabled": bool(settings.MARKDOWN_FALLBACK_ENABLED),
            "index_fallback_notes": fallback_notes,
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
    content_source: str = "cleaned_html",
    fit_source: bool = False,
    fit_markdown: Optional[str] = None,
) -> str:
    """Extract structured JSON using css/xpath/llm strategies and return a normalized contract."""
    strategy = (extraction_strategy or "css").strip().lower()

    if fit_markdown is not None:
        if strategy != "regex":
            return json.dumps(
                {
                    "success": False,
                    "error": "fit_markdown extraction is currently supported only for regex strategy.",
                },
                indent=2,
            )
        regex_patterns = extraction_schema if isinstance(extraction_schema, dict) else {}
        extracted_regex: Dict[str, Any] = {}
        for key, pattern in regex_patterns.items():
            try:
                extracted_regex[key] = re.findall(str(pattern), fit_markdown)
            except re.error:
                extracted_regex[key] = []

        normalized = _normalize_structured_contract(
            strategy="regex",
            extracted_content=extracted_regex,
            source_type="fit_markdown",
            source_value="inline",
            schema=regex_patterns,
            fit_source_used=True,
        )
        return json.dumps({"success": True, "normalized_output": normalized}, indent=2)

    target_url = url
    source_type = "url"
    if html is not None:
        if not html.strip():
            return json.dumps({"success": False, "error": "html must be non-empty when provided."}, indent=2)
        target_url = f"raw:{html}"
        source_type = "raw_html"
    elif file_path is not None:
        normalized = file_path.replace("file://", "", 1) if file_path.startswith("file://") else file_path
        target_url = f"file://{Path(normalized).resolve()}"
        source_type = "file"

    if not target_url:
        return json.dumps({"success": False, "error": "Provide one of: url, file_path, html."}, indent=2)

    selected_content_source = "fit_html" if fit_source else content_source
    extracted_response = await crawl_to_markdown(
        ctx=ctx,
        url=target_url,
        markdown_variant="raw",
        run_config=run_config,
        index_result=False,
        extraction_strategy=strategy,
        extraction_schema=extraction_schema,
        extraction_instruction=extraction_instruction,
        llm_provider=llm_provider,
        content_source=selected_content_source,
    )
    payload = json.loads(extracted_response)
    if not payload.get("success"):
        return json.dumps(payload, indent=2)

    normalized = _normalize_structured_contract(
        strategy=strategy,
        extracted_content=payload.get("extraction_result"),
        source_type=source_type,
        source_value=target_url,
        schema=extraction_schema,
        fit_source_used=bool(fit_source),
    )
    payload["normalized_output"] = normalized
    return json.dumps(payload, indent=2)


async def generate_extraction_schema(
    ctx: Context,
    sample_html: str,
    strategy: str = "css",
    schema_name: Optional[str] = None,
    cache_schema: bool = True,
) -> str:
    """Generate extraction schema from sample HTML, validate it, and optionally cache it."""
    if not sample_html or not sample_html.strip():
        return json.dumps({"success": False, "error": "sample_html must be non-empty."}, indent=2)

    generated = _build_schema_from_sample_html(sample_html, strategy=strategy)
    validation = _validate_generated_schema(generated, strategy=strategy)
    if not validation["valid"]:
        return json.dumps(
            {
                "success": False,
                "strategy": strategy,
                "validation": validation,
            },
            indent=2,
        )

    cache_path = None
    if cache_schema:
        cache_path = _cache_generated_schema(
            validation["normalized_schema"],
            sample_html=sample_html,
            strategy=strategy,
            schema_name=schema_name,
        )

    return json.dumps(
        {
            "success": True,
            "strategy": strategy,
            "schema": validation["normalized_schema"],
            "validation": validation,
            "cached": bool(cache_path),
            "cache_path": cache_path,
        },
        indent=2,
    )


async def validate_extraction_schema(
    ctx: Context,
    schema: Dict[str, Any],
    strategy: str = "css",
) -> str:
    """Validate a caller-provided extraction schema."""
    validation = _validate_generated_schema(schema, strategy=strategy)
    return json.dumps(
        {
            "success": bool(validation["valid"]),
            "strategy": strategy,
            "validation": validation,
        },
        indent=2,
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
    run_id: Optional[str] = None,
    chunking_strategy: Optional[str] = None,
) -> str:
    """Index caller-supplied markdown into pgvector.

    Args:
        url: Canonical URL to associate with the stored chunks.
        markdown: Markdown text to chunk and embed.
        metadata: Optional extra metadata merged into each chunk's page_metadata.
        run_id: Optional run identifier; auto-generated if omitted.
        chunking_strategy: Override the server-wide chunking strategy for this
            call.  Accepted values: ``"paragraph"``, ``"sentence"``,
            ``"fixed"``.  Defaults to the server-wide ``CHUNK_STRATEGY``.
    """
    try:
        if not markdown or not markdown.strip():
            return json.dumps({"success": False, "url": url, "error": "markdown must be non-empty."}, indent=2)

        chunks = await chunk_text_according_to_settings(markdown, strategy=chunking_strategy)
        db_urls, db_chunks, db_contents, db_metas, db_fulldocs = [], [], [], [], []
        source_type = _infer_source_type(url)
        effective_run_id = _normalize_run_id(run_id) or _generate_run_id("index")
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
                "run_id": effective_run_id,
            })
            if metadata:
                meta.update(metadata)
            db_urls.append(url)
            db_chunks.append(i)
            db_contents.append(chunk)
            db_metas.append(meta)
            db_fulldocs.append(markdown)

        first_chunk_id: Optional[int] = None
        with next(get_session()) as session:
            chunks_stored = await add_documents_to_db(session, db_urls, db_contents, db_metas, db_chunks, db_fulldocs)
            if chunks_stored > 0:
                first_row = session.exec(
                    select(CrawledPage)
                    .where(CrawledPage.url == url)
                    .order_by(CrawledPage.chunk_number)
                ).first()
                if first_row is not None:
                    first_chunk_id = first_row.id

        return json.dumps({
            "success": True,
            "url": url,
            "chunks_stored": chunks_stored,
            "pages_indexed": 1,
            "run_id": effective_run_id,
            "first_chunk_id": first_chunk_id,
            "chunking_strategy_applied": chunking_strategy or None,
        }, indent=2)
    except Exception as exc:
        logger.error(f"index_markdown {url}: {exc}", exc_info=True)
        return json.dumps({"success": False, "url": url, "error": str(exc)}, indent=2)


async def index_fit_markdown(
    ctx: Context,
    url: str,
    fit_markdown: str,
    metadata: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
) -> str:
    """Index caller-supplied fit-markdown variant."""
    merged_meta = dict(metadata or {})
    merged_meta["markdown_variant"] = "fit_markdown"
    return await index_markdown(ctx=ctx, url=url, markdown=fit_markdown, metadata=merged_meta, run_id=run_id)


async def index_structured_content(
    ctx: Context,
    url: str,
    structured_content: Any,
    metadata: Optional[Dict[str, Any]] = None,
    projection_mode: str = "hybrid",
    run_id: Optional[str] = None,
) -> str:
    """Index structured content using raw-json, flattened-text, or hybrid projection."""
    merged_meta = dict(metadata or {})
    merged_meta["content_class"] = "structured"
    projected, projection_meta = _project_structured_content(structured_content, projection_mode)
    merged_meta.update(projection_meta)
    return await index_markdown(ctx=ctx, url=url, markdown=projected, metadata=merged_meta, run_id=run_id)


async def index_code_examples(
    ctx: Context,
    url: str,
    markdown: str,
    run_id: Optional[str] = None,
) -> str:
    """Index code blocks extracted from caller-supplied markdown."""
    try:
        if not markdown or not markdown.strip():
            return json.dumps({"success": False, "url": url, "error": "markdown must be non-empty."}, indent=2)
        blocks = extract_code_blocks(markdown)
        if not blocks:
            return json.dumps({"success": True, "url": url, "code_examples_indexed": 0}, indent=2)
        with next(get_session()) as session:
            effective_run_id = _normalize_run_id(run_id) or _generate_run_id("index-code")
            await add_code_examples_to_db(
                session,
                urls=[url] * len(blocks),
                contents=[b["content"] for b in blocks],
                languages=[b["language"] for b in blocks],
                summaries=[None] * len(blocks),
                metadatas=[{"source": urlparse(url).netloc, "url": url, "source_type": _infer_source_type(url), "run_id": effective_run_id}] * len(blocks),
                chunk_numbers=list(range(len(blocks))),
            )
        return json.dumps({"success": True, "url": url, "code_examples_indexed": len(blocks), "run_id": effective_run_id}, indent=2)
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
    fresh_only: bool = True,
    staleness_threshold: float = 0.5,
    as_of: Optional[str] = None,
    recency_bias: float = 0.0,
    include_provenance: bool = False,
) -> str:
    """Taxonomy-native retrieval tool."""
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
                session,
                query,
                match_count=match_count,
                filter_metadata=filter_meta or None,
            )

        as_of_dt = _parse_datetime_utc(as_of) if as_of else None
        if fresh_only or as_of_dt is not None:
            threshold = max(0.0, min(1.0, float(staleness_threshold)))
            results = [
                row for row in results
                if _is_result_fresh(
                    row.get("page_metadata") if isinstance(row.get("page_metadata"), dict) else {},
                    threshold,
                    as_of_dt,
                    bool(fresh_only),
                )
            ]

        if settings.USE_RERANKING and results:
            results = rerank_results(query, results, top_k=match_count)

        recency_weight = max(0.0, min(1.0, float(recency_bias)))
        if recency_weight > 0 and results:
            rescored: List[Dict[str, Any]] = []
            for row in results:
                metadata = row.get("page_metadata") if isinstance(row.get("page_metadata"), dict) else {}
                freshness = _compute_freshness_from_metadata(metadata)
                similarity = float(row.get("similarity_score") or 0.0)
                final_score = similarity * (1.0 + recency_weight * freshness)
                row_copy = dict(row)
                row_copy["freshness_score"] = round(freshness, 6)
                row_copy["final_score"] = round(final_score, 6)
                rescored.append(row_copy)
            results = sorted(rescored, key=lambda r: float(r.get("final_score") or 0.0), reverse=True)[:match_count]

        formatted = []
        for r in results:
            entry = {
                "url": r.get("url"),
                "content": r.get("content"),
                "metadata": r.get("page_metadata"),
                "similarity": r.get("similarity_score"),
            }
            if "final_score" in r:
                entry["final_score"] = r.get("final_score")
            if "freshness_score" in r:
                entry["freshness_score"] = r.get("freshness_score")
            if include_provenance:
                entry["provenance"] = _build_requested_provenance(r.get("page_metadata"))
            formatted.append(entry)
        return json.dumps({
            "success": True,
            "query": query,
            "fresh_only": bool(fresh_only),
            "staleness_threshold": max(0.0, min(1.0, float(staleness_threshold))),
            "as_of": as_of_dt.isoformat() if as_of_dt else None,
            "recency_bias": recency_weight,
            "results": formatted,
        }, indent=2)
    except Exception as exc:
        logger.error(f"search_documents_v2: {exc}", exc_info=True)
        return json.dumps({"success": False, "query": query, "error": str(exc)}, indent=2)


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


async def get_fit_markdown_by_url(ctx: Context, url: str, include_provenance: bool = False) -> str:
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
        payload = {
            "success": True,
            "url": url,
            "chunk_count": len(fit_rows),
            "fit_markdown": markdown,
            "selected_variant": "fit_markdown",
        }
        if include_provenance and fit_rows:
            payload["provenance"] = _build_requested_provenance(fit_rows[0].page_metadata)

        return json.dumps(payload, indent=2)
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


async def search_raw_markdown(
    ctx: Context,
    query: str,
    source: Optional[str] = None,
    match_count: int = 5,
    fresh_only: bool = True,
    recency_bias: float = 0.0,
    include_provenance: bool = False,
) -> str:
    """Search only raw markdown chunks."""
    return await search_documents_v2(
        ctx=ctx,
        query=query,
        source=source,
        match_count=match_count,
        markdown_variant="raw_markdown",
        fresh_only=fresh_only,
        recency_bias=recency_bias,
        include_provenance=include_provenance,
    )


async def search_fit_markdown(
    ctx: Context,
    query: str,
    source: Optional[str] = None,
    match_count: int = 5,
    fresh_only: bool = True,
    recency_bias: float = 0.0,
    include_provenance: bool = False,
) -> str:
    """Search only fit-markdown chunks."""
    return await search_documents_v2(
        ctx=ctx,
        query=query,
        source=source,
        match_count=match_count,
        markdown_variant="fit_markdown",
        fresh_only=fresh_only,
        recency_bias=recency_bias,
        include_provenance=include_provenance,
    )


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
            source_policy_map = {
                sp.source: int(sp.min_active_docs)
                for sp in session.exec(select(SourcePolicy)).all()
            }

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
                        "canonical_key": _canonical_url_key(meta.get("canonical_url") or rec.url),
                        "canonical_guard": bool(meta.get("canonical_url") or meta.get("markdown_variant")),
                        "chunk_number": rec.chunk_number,
                        "source": rec_source,
                        "value_score": round(rec.value_score, 4),
                        "staleness_score": round(rec.staleness_score, 4),
                        "hit_count": rec.hit_count,
                        "last_seen_at": (
                            _parse_datetime_utc(getattr(rec, "last_seen_at", None)).isoformat()
                            if _parse_datetime_utc(getattr(rec, "last_seen_at", None)) is not None
                            else None
                        ),
                        "content_length": len(rec.content),
                    })

            source_active_counts, canonical_active_counts = _build_active_coverage_maps(session)

            candidates = _apply_eviction_safeguards(
                candidates,
                source_policy_map,
                source_active_counts,
                canonical_active_counts,
            )
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
            source_policy_records = session.exec(select(SourcePolicy)).all()
            source_policy_map = {
                sp.source: int(sp.min_active_docs)
                for sp in source_policy_records
            }
            source_policies_by_source = {sp.source: sp for sp in source_policy_records}
            if policy is None:
                max_gb = 10.0
                warn_pct = 0.80
                high_pct = 0.90
                hard_pct = 1.00
                grace_h = 24
                post_evict = 0.75
                max_crawled_pages_mb: Optional[int] = None
                max_code_examples_mb: Optional[int] = None
            else:
                max_gb = policy.max_db_size_gb
                warn_pct = policy.warn_threshold
                high_pct = policy.high_threshold
                hard_pct = policy.hard_threshold
                grace_h = policy.tombstone_grace_hours
                post_evict = policy.target_post_evict_ratio
                max_crawled_pages_mb = policy.max_crawled_pages_mb
                max_code_examples_mb = policy.max_code_examples_mb

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

            # Step 0: policy-driven hard TTL purge
            ttl_deleted = _apply_hard_ttl_delete(session, source_policies_by_source)
            ttl_total = ttl_deleted["crawled_pages"] + ttl_deleted["code_examples"]
            if ttl_total:
                actions_taken.append(
                    f"hard-ttl deleted {ttl_total} records "
                    f"(pages={ttl_deleted['crawled_pages']}, code={ttl_deleted['code_examples']})"
                )

            # Step 0.5: per-source and per-table budget shaping
            quota_result = _enforce_source_quotas(session, source_policies_by_source)
            if quota_result["quota_evicted"]:
                actions_taken.append(
                    f"source-quota tombstoned {quota_result['quota_evicted']} records"
                )

            table_budget_evicted = _enforce_table_budgets(
                session,
                max_crawled_pages_mb=max_crawled_pages_mb,
                max_code_examples_mb=max_code_examples_mb,
            )
            table_budget_total = table_budget_evicted["crawled_pages"] + table_budget_evicted["code_examples"]
            if table_budget_total:
                actions_taken.append(
                    f"table-budget tombstoned {table_budget_total} records "
                    f"(pages={table_budget_evicted['crawled_pages']}, code={table_budget_evicted['code_examples']})"
                )

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
                    source_active_counts, canonical_active_counts = _build_active_coverage_maps(session)
                    all_candidates: List[Dict[str, Any]] = []
                    for tbl_model, table_name in ((CrawledPage, "crawled_pages"), (CodeExample, "code_examples")):
                        rows = session.exec(
                            select(tbl_model).where(
                                tbl_model.is_pinned == False,
                                tbl_model.tombstoned_at.is_(None),  # type: ignore[union-attr]
                                tbl_model.is_active == True,
                            ).limit(200)
                        ).all()

                        for r in rows:
                            meta = (
                                r.page_metadata
                                if hasattr(r, "page_metadata") and isinstance(r.page_metadata, dict)
                                else getattr(r, "ex_metadata", {})
                            ) or {}
                            all_candidates.append(
                                {
                                    "table": table_name,
                                    "id": r.id,
                                    "url": r.url,
                                    "canonical_key": _canonical_url_key(meta.get("canonical_url") or r.url),
                                    "canonical_guard": bool(meta.get("canonical_url") or meta.get("markdown_variant")),
                                    "source": str(meta.get("source", "")),
                                    "value_score": float(getattr(r, "value_score", 0.0) or 0.0),
                                    "staleness_score": float(getattr(r, "staleness_score", 0.0) or 0.0),
                                    "hit_count": int(getattr(r, "hit_count", 0) or 0),
                                    "last_seen_at": (
                                        _parse_datetime_utc(getattr(r, "last_seen_at", None)).isoformat()
                                        if _parse_datetime_utc(getattr(r, "last_seen_at", None)) is not None
                                        else None
                                    ),
                                }
                            )

                    selected_candidates = _apply_eviction_safeguards(
                        all_candidates,
                        source_policy_map,
                        source_active_counts,
                        canonical_active_counts,
                    )
                    selected_candidates = selected_candidates[:200]

                    page_ids = [c["id"] for c in selected_candidates if c["table"] == "crawled_pages"]
                    code_ids = [c["id"] for c in selected_candidates if c["table"] == "code_examples"]
                    if page_ids:
                        evicted += tombstone_records(
                            session, page_ids, "crawled_pages", "hard_pressure_value_evict"
                        )
                    if code_ids:
                        evicted += tombstone_records(
                            session, code_ids, "code_examples", "hard_pressure_value_evict"
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
                "source_quota_overrides": quota_result.get("sources_over_quota", []),
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
        max_concurrent: Reserved for future scheduler concurrency tuning.
    """
    try:
        now = datetime.now(timezone.utc)
        due_sources: List[Dict[str, Any]] = []
        backoff_skipped: List[Dict[str, Any]] = []
        recrawled: List[Dict[str, Any]] = []
        failures: List[Dict[str, Any]] = []
        source_policy_map: Dict[str, Any] = {}
        dead_page_tombstoned = 0
        duplicate_retired = 0
        superseded_retired = 0

        with next(get_session()) as session:
            policies = session.exec(select(SourcePolicy)).all()
            if source:
                policies = [p for p in policies if p.source == source]
            source_policy_map = {p.source: p for p in policies}

            for pol in policies:
                next_retry_at = _parse_datetime_utc(getattr(pol, "next_retry_at", None))
                if next_retry_at is not None and next_retry_at > now:
                    backoff_skipped.append(
                        {
                            "source": pol.source,
                            "next_retry_at": next_retry_at.isoformat(),
                        }
                    )
                    continue

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

                interval_h = max(1, int(getattr(pol, "recrawl_interval_hours", 168) or 168))
                if last_crawled is None:
                    is_due = True
                else:
                    is_due = (now - last_crawled) >= timedelta(hours=interval_h)

                if is_due:
                    due_sources.append(
                        {
                            "source": pol.source,
                            "recrawl_interval_hours": interval_h,
                            "last_crawled_at": last_crawled.isoformat() if isinstance(last_crawled, datetime) else None,
                        }
                    )

            for item in due_sources:
                src = item["source"]
                policy = source_policy_map.get(src)
                crawl_url = src if src.startswith(("http://", "https://")) else f"https://{src}"
                try:
                    result = await crawl_to_markdown(
                        ctx=ctx,
                        url=crawl_url,
                        markdown_variant="raw",
                        index_result=True,
                    )
                    parsed = json.loads(result)
                    if parsed.get("success"):
                        recrawled.append(
                            {
                                "source": src,
                                "url": crawl_url,
                                "pages_crawled": parsed.get("pages_crawled", 1),
                            }
                        )

                        if policy is not None:
                            policy.consecutive_failures = 0
                            policy.next_retry_at = None

                        try:
                            retire_stats = _retire_source_duplicates_and_superseded(session, src)
                            duplicate_retired += retire_stats["duplicate_retired"]
                            superseded_retired += retire_stats["superseded_retired"]
                        except Exception as retire_exc:  # pragma: no cover - defensive
                            logger.warning(f"retire_source_duplicates_and_superseded failed for {src}: {retire_exc}")
                    else:
                        error_message = parsed.get("error", "recrawl failed")
                        failures.append({"source": src, "url": crawl_url, "error": error_message})

                        if policy is not None:
                            policy.consecutive_failures = int(getattr(policy, "consecutive_failures", 0) or 0) + 1
                            backoff_h = _compute_retry_backoff_hours(policy)
                            policy.next_retry_at = now + timedelta(hours=backoff_h) if backoff_h > 0 else None

                        if _is_dead_page_error(error_message):
                            page_rows = session.exec(
                                select(CrawledPage).where(
                                    CrawledPage.is_active == True,
                                    CrawledPage.tombstoned_at.is_(None),
                                )
                            ).all()
                            code_rows = session.exec(
                                select(CodeExample).where(
                                    CodeExample.is_active == True,
                                    CodeExample.tombstoned_at.is_(None),
                                )
                            ).all()
                            page_ids = [
                                row.id
                                for row in page_rows
                                if str(_record_metadata(row).get("source") or "") == src
                            ]
                            code_ids = [
                                row.id
                                for row in code_rows
                                if str(_record_metadata(row).get("source") or "") == src
                            ]
                            if page_ids:
                                dead_page_tombstoned += tombstone_records(
                                    session,
                                    page_ids,
                                    "crawled_pages",
                                    "dead_page_policy_404_410",
                                )
                            if code_ids:
                                dead_page_tombstoned += tombstone_records(
                                    session,
                                    code_ids,
                                    "code_examples",
                                    "dead_page_policy_404_410",
                                )

                except Exception as exc:  # pragma: no cover - defensive outer catch still tested via mocks
                    failures.append({"source": src, "url": crawl_url, "error": str(exc)})
                    if policy is not None:
                        policy.consecutive_failures = int(getattr(policy, "consecutive_failures", 0) or 0) + 1
                        backoff_h = _compute_retry_backoff_hours(policy)
                        policy.next_retry_at = now + timedelta(hours=backoff_h) if backoff_h > 0 else None

            session.commit()

        return json.dumps(
            {
                "success": True,
                "due_count": len(due_sources),
                "backoff_skipped_count": len(backoff_skipped),
                "recrawled_count": len(recrawled),
                "failed_count": len(failures),
                "dead_page_tombstoned": dead_page_tombstoned,
                "duplicate_retired": duplicate_retired,
                "superseded_retired": superseded_retired,
                "recrawled_sources": recrawled,
                "backoff_skipped": backoff_skipped,
                "failures": failures,
            },
            indent=2,
        )
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
            source_policies = {sp.source: sp for sp in session.exec(select(SourcePolicy)).all()}
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

            ttl_deleted = _apply_hard_ttl_delete(session, source_policies)
            total_deleted += ttl_deleted["crawled_pages"] + ttl_deleted["code_examples"]
            session.commit()

        return json.dumps({
            "success": True,
            "force": force,
            "grace_hours": grace_hours,
            "hard_deleted_count": total_deleted,
            "deleted_by_table": deleted_by_table,
            "hard_ttl_deleted_by_table": ttl_deleted,
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


async def detect_content_drift(
    ctx: Context,
    source: Optional[str] = None,
    major_threshold: float = 0.85,
    minor_threshold: float = 0.60,
    trigger_selective_reembed: bool = False,
) -> str:
    """Classify indexed URLs as unchanged/minor/major drift using lifecycle signals.

    This is scheduler-oriented maintenance and intentionally not MCP-exposed.
    """
    try:
        major_cut = max(0.0, min(1.0, float(major_threshold)))
        minor_cut = max(0.0, min(major_cut, float(minor_threshold)))

        with next(get_session()) as session:
            active_rows = session.exec(
                select(CrawledPage).where(
                    CrawledPage.is_active == True,
                    CrawledPage.tombstoned_at.is_(None),
                )
            ).all()
            active_rows = [
                row
                for row in active_rows
                if bool(getattr(row, "is_active", True))
                and getattr(row, "tombstoned_at", None) is None
            ]

            removed_rows = session.exec(
                select(CrawledPage).where(
                    (CrawledPage.is_active == False) | CrawledPage.tombstoned_at.isnot(None)
                )
            ).all()
            removed_rows = [
                row
                for row in removed_rows
                if (not bool(getattr(row, "is_active", True)))
                or getattr(row, "tombstoned_at", None) is not None
            ]

            if source:
                active_rows = [
                    row for row in active_rows
                    if isinstance(row.page_metadata, dict) and row.page_metadata.get("source") == source
                ]
                removed_rows = [
                    row for row in removed_rows
                    if isinstance(row.page_metadata, dict) and row.page_metadata.get("source") == source
                ]

            unchanged: List[int] = []
            minor: List[int] = []
            major: List[int] = []

            for row in active_rows:
                staleness = float(getattr(row, "staleness_score", 0.0) or 0.0)
                if staleness >= major_cut:
                    major.append(row.id)
                elif staleness >= minor_cut:
                    minor.append(row.id)
                else:
                    unchanged.append(row.id)

            selective_reembed_candidates = [
                row.id
                for row in active_rows
                if row.id in major and int(getattr(row, "hit_count", 0) or 0) > 0
            ]

            selective_reembedded = 0
            if trigger_selective_reembed and selective_reembed_candidates:
                selective_reembedded = await _run_selective_reembed(
                    session,
                    selective_reembed_candidates,
                )

        return json.dumps(
            {
                "success": True,
                "source": source,
                "trigger_selective_reembed": trigger_selective_reembed,
                "thresholds": {"minor": minor_cut, "major": major_cut},
                "counts": {
                    "total": len(active_rows) + len(removed_rows),
                    "unchanged": len(unchanged),
                    "minor_update": len(minor),
                    "major_rewrite": len(major),
                    "removed": len(removed_rows),
                },
                "selective_reembed_candidate_count": len(selective_reembed_candidates),
                "selective_reembed_executed_count": selective_reembedded,
                "selective_reembed_candidates": selective_reembed_candidates[:100],
            },
            indent=2,
        )
    except Exception as exc:
        logger.error(f"detect_content_drift failed: {exc}", exc_info=True)
        return json.dumps({"success": False, "error": str(exc)}, indent=2)
