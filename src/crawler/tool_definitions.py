"""MCP tool definitions — registered in src/crawl4ai_mcp.py."""

import hashlib
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, cast
from urllib.parse import urlparse

from crawl4ai import (
    AdaptiveConfig,
    AdaptiveCrawler,
    AsyncWebCrawler,
    BestFirstCrawlingStrategy,
    BFSDeepCrawlStrategy,
    BM25ContentFilter,
    BrowserConfig,
    CacheMode,
    ContentRelevanceFilter,
    ContentTypeFilter,
    CrawlerRunConfig,
    DefaultMarkdownGenerator,
    DFSDeepCrawlStrategy,
    DomainFilter,
    FilterChain,
    JsonCssExtractionStrategy,
    JsonXPathExtractionStrategy,
    LLMConfig,
    LLMContentFilter,
    LLMExtractionStrategy,
    MemoryAdaptiveDispatcher,
    PruningContentFilter,
    RegexExtractionStrategy,
    SEOFilter,
    URLPatternFilter,
)
from fastmcp import Context
from sqlalchemy import text as _sql_text
from sqlmodel import Session, select

from src.utils import (
    CodeExample,
    CrawledPage,
    MarkdownIndexPolicy,
    SourcePolicy,
    StoragePolicy,
    _get_db_size_bytes,
    add_code_examples_to_db,
    add_documents_to_db,
    compute_staleness_score,
    compute_value_score,
    create_embedding,
    extract_code_blocks,
    extract_link_references,
    get_session,
    rerank_results,
)
from src.utils import search_code_examples as _search_code_examples
from src.utils import search_documents as _search_documents_core
from src.utils import (
    settings,
    tombstone_records,
)

from .metadata_extractor import (
    extract_link_graph,
    extract_media_metadata,
    extract_section_info,
)
from .postgres_client import store_crawled_documents
from .url_scorers import build_url_scorer, get_supported_scorer_types
from .web_crawler import (
    chunk_text_according_to_settings,
    crawl_recursive_internal_links,
)

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


_INDEX_VARIANTS_BY_OVERRIDE: Dict[str, List[str]] = {
    "raw": ["raw_markdown"],
    "fit": ["fit_markdown"],
    "both": ["raw_markdown", "fit_markdown"],
}

_INDEX_VARIANTS_BY_POLICY: Dict[str, List[str]] = {
    MarkdownIndexPolicy.RAW_ONLY.value: ["raw_markdown"],
    MarkdownIndexPolicy.FIT_ONLY.value: ["fit_markdown"],
    MarkdownIndexPolicy.BOTH_BY_DEFAULT.value: ["raw_markdown", "fit_markdown"],
}


def _preferred_index_variants(override: Optional[str], effective_policy: str) -> List[str]:
    if override in _INDEX_VARIANTS_BY_OVERRIDE:
        return _INDEX_VARIANTS_BY_OVERRIDE[override]
    return _INDEX_VARIANTS_BY_POLICY.get(effective_policy, ["raw_markdown", "fit_markdown"])


def _resolved_preferred_variants(variants: Dict[str, str], preferred: List[str]) -> List[str]:
    return [variant for variant in preferred if variants.get(variant)]


def _missing_preferred_variants(preferred: List[str], resolved: List[str]) -> List[str]:
    return [variant for variant in preferred if variant not in resolved]


def _append_fallback_variant(
    resolved: List[str],
    fallback_notes: List[str],
    *,
    requested_variant: str,
    fallback_variant: str,
    variants: Dict[str, str],
    message: str,
) -> None:
    if requested_variant not in _missing_preferred_variants([requested_variant], resolved):
        return
    if not variants.get(fallback_variant):
        return
    if fallback_variant in resolved:
        return
    resolved.append(fallback_variant)
    fallback_notes.append(message)


def _apply_variant_fallbacks(
    variants: Dict[str, str],
    preferred: List[str],
    fallback_enabled: bool,
) -> tuple[List[str], List[str]]:
    resolved = _resolved_preferred_variants(variants, preferred)
    missing = _missing_preferred_variants(preferred, resolved)
    if not missing:
        return resolved, []
    if not fallback_enabled:
        return resolved, ["fallback disabled; unavailable requested variants were skipped"]

    fallback_notes: List[str] = []
    _append_fallback_variant(
        resolved,
        fallback_notes,
        requested_variant="raw_markdown",
        fallback_variant="fit_markdown",
        variants=variants,
        message="raw_markdown unavailable; indexed fit_markdown instead",
    )
    _append_fallback_variant(
        resolved,
        fallback_notes,
        requested_variant="fit_markdown",
        fallback_variant="raw_markdown",
        variants=variants,
        message="fit_markdown unavailable; indexed raw_markdown instead",
    )
    return resolved, fallback_notes


def _resolve_variants_to_index(
    variants: Dict[str, str],
    index_variants: Optional[str] = None,
    fallback_enabled: bool = True,
) -> tuple[List[str], str, Optional[str], List[str]]:
    """Resolve markdown variants to index based on global policy + per-run override."""
    override = _normalize_index_variants_override(index_variants)
    effective_policy = _normalize_markdown_index_policy(settings.MARKDOWN_INDEX_POLICY)
    preferred = _preferred_index_variants(override, effective_policy)
    resolved, fallback_notes = _apply_variant_fallbacks(variants, preferred, fallback_enabled)
    ordered = [variant for variant in ("raw_markdown", "fit_markdown") if variant in resolved]
    return ordered, effective_policy, override, fallback_notes


def _allowed_run_config_kwargs(run_config: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in run_config.items() if k in _ALLOWED_RUN_CONFIG_FIELDS}


def _normalized_cache_mode(cache_mode: Any) -> Optional[CacheMode]:
    if isinstance(cache_mode, CacheMode):
        return cache_mode
    if not isinstance(cache_mode, str):
        return None
    key = cache_mode.upper().strip()
    if key in CacheMode.__members__:
        return CacheMode[key]
    return None


def _with_cache_mode_default(safe_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    normalized = _normalized_cache_mode(safe_kwargs.get("cache_mode"))
    if normalized is None:
        safe_kwargs.pop("cache_mode", None)
        safe_kwargs["cache_mode"] = CacheMode.BYPASS
    else:
        safe_kwargs["cache_mode"] = normalized
    return safe_kwargs


def _build_run_config(run_config: Optional[Dict[str, Any]] = None) -> CrawlerRunConfig:
    """Build a CrawlerRunConfig from a safe allowlist of fields."""
    if not run_config:
        return CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
    safe_kwargs = _allowed_run_config_kwargs(run_config)
    return CrawlerRunConfig(**_with_cache_mode_default(safe_kwargs))


def _empty_markdown_variants() -> Dict[str, str]:
    return {
        "raw_markdown": "",
        "fit_markdown": "",
        "markdown_with_citations": "",
        "references_markdown": "",
        "fit_html": "",
    }


def _variant_text(markdown_obj: Any, attr: str, default: str = "") -> str:
    value = getattr(markdown_obj, attr, None)
    return value if isinstance(value, str) else default


def _extract_markdown_variants(markdown_obj: Any) -> Dict[str, str]:
    """Extract markdown variants from Crawl4AI's markdown-compatible object."""
    if markdown_obj is None:
        return _empty_markdown_variants()

    raw_markdown = _variant_text(markdown_obj, "raw_markdown", str(markdown_obj))
    return {
        "raw_markdown": raw_markdown,
        "fit_markdown": _variant_text(markdown_obj, "fit_markdown"),
        "markdown_with_citations": _variant_text(markdown_obj, "markdown_with_citations", raw_markdown),
        "references_markdown": _variant_text(markdown_obj, "references_markdown"),
        "fit_html": _variant_text(markdown_obj, "fit_html"),
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
    link_references = _provenance_link_references(meta)
    references_markdown = _provenance_references_markdown(meta)
    return {
        "source": meta.get("source"),
        "url": meta.get("url"),
        "source_type": meta.get("source_type"),
        "crawl_type": meta.get("crawl_type"),
        "crawl_timestamp": meta.get("crawl_timestamp") or meta.get("crawl_time"),
        "session_id": meta.get("session_id"),
        "markdown_variant": meta.get("markdown_variant"),
        "extraction_strategy": meta.get("extraction_strategy"),
        "references_markdown": references_markdown,
        "link_references": link_references,
        "has_link_references": bool(references_markdown or link_references),
        "has_citations": bool(meta.get("has_citations", False)),
    }


def _provenance_link_references(meta: Dict[str, Any]) -> List[Any]:
    value = meta.get("link_references")
    return value if isinstance(value, list) else []


def _provenance_references_markdown(meta: Dict[str, Any]) -> str:
    value = meta.get("references_markdown")
    return value if isinstance(value, str) else ""


def _select_rows_for_variant(rows: List[Any], preferred_variant: str) -> List[Any]:
    """Prefer rows matching a specific markdown variant, with fallback to all rows."""
    preferred_rows = [
        row
        for row in rows
        if isinstance(getattr(row, "page_metadata", None), dict)
        and row.page_metadata.get("markdown_variant") == preferred_variant
    ]
    return preferred_rows or rows


def _defaulted_browser_kwargs(safe_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    safe_kwargs.setdefault("headless", True)
    safe_kwargs.setdefault("verbose", False)
    return safe_kwargs


def _build_browser_config(browser_config: Optional[Dict[str, Any]] = None) -> BrowserConfig:
    """Build a BrowserConfig from an allowlist of safe fields."""
    if not browser_config:
        return BrowserConfig(headless=True, verbose=False)
    safe_kwargs = {k: v for k, v in browser_config.items() if k in _ALLOWED_BROWSER_CONFIG_FIELDS}
    return BrowserConfig(**_defaulted_browser_kwargs(safe_kwargs))


def _normalized_content_source(content_source: str) -> str:
    normalized_source = (content_source or "cleaned_html").strip().lower()
    if normalized_source in _ALLOWED_CONTENT_SOURCES:
        return normalized_source
    return "cleaned_html"


def _normalized_content_filter(content_filter: Optional[str]) -> Optional[str]:
    if not isinstance(content_filter, str):
        return None
    normalized_filter = content_filter.lower().strip()
    if normalized_filter in _ALLOWED_CONTENT_FILTERS:
        return normalized_filter
    return None


def _numeric_threshold(value: Optional[float], default: float) -> float:
    return value if isinstance(value, (int, float)) else default


def _pruning_filter(query: Optional[str], threshold: Optional[float]) -> Any:
    return PruningContentFilter(user_query=query, threshold=_numeric_threshold(threshold, 0.48))


def _bm25_filter(query: Optional[str], threshold: Optional[float]) -> Any:
    return BM25ContentFilter(user_query=query, bm25_threshold=_numeric_threshold(threshold, 1.0))


def _first_non_empty_string(*values: Optional[str]) -> Optional[str]:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value
    return None


def _resolved_llm_config(llm_provider: Optional[str]) -> LLMConfig:
    provider = (
        _first_non_empty_string(
            llm_provider,
            settings.effective_agentic_model_name,
            settings.effective_hybrid_model_name,
        )
        or "openai/gpt-4o"
    )
    api_token = _first_non_empty_string(
        settings.effective_agentic_api_key,
        settings.effective_hybrid_api_key,
    )
    base_url = _first_non_empty_string(
        settings.effective_agentic_base_url,
        settings.effective_hybrid_base_url,
    )
    return LLMConfig(provider=provider, api_token=api_token, base_url=base_url)


def _llm_filter(instruction: Optional[str], llm_provider: Optional[str]) -> Any:
    llm_cfg = _resolved_llm_config(llm_provider)
    return LLMContentFilter(
        llm_config=llm_cfg,
        instruction=instruction or "Keep only the most relevant content for the user query.",
    )


def _content_filter_object(
    normalized_filter: Optional[str],
    content_filter_query: Optional[str],
    content_filter_threshold: Optional[float],
    content_filter_instruction: Optional[str],
    llm_provider: Optional[str],
) -> Optional[Any]:
    builders = {
        "pruning": lambda: _pruning_filter(content_filter_query, content_filter_threshold),
        "bm25": lambda: _bm25_filter(content_filter_query, content_filter_threshold),
        "llm": lambda: _llm_filter(content_filter_instruction, llm_provider),
    }
    if not isinstance(normalized_filter, str):
        return None
    builder = builders.get(normalized_filter)
    return builder() if builder else None


def _is_default_markdown_generator(options: Dict[str, Any], normalized_source: str, filter_obj: Optional[Any]) -> bool:
    return not options and normalized_source == "cleaned_html" and filter_obj is None


def _build_markdown_generator(
    markdown_options: Optional[Dict[str, Any]] = None,
    content_source: str = "cleaned_html",
    content_filter: Optional[str] = None,
    content_filter_query: Optional[str] = None,
    content_filter_threshold: Optional[float] = None,
    content_filter_instruction: Optional[str] = None,
    llm_provider: Optional[str] = None,
) -> Optional[DefaultMarkdownGenerator]:
    """Build a markdown generator with optional content-source and filter controls."""
    options = dict(markdown_options or {})
    normalized_source = _normalized_content_source(content_source)
    normalized_filter = _normalized_content_filter(content_filter)
    filter_obj = _content_filter_object(
        normalized_filter,
        content_filter_query,
        content_filter_threshold,
        content_filter_instruction,
        llm_provider,
    )
    if _is_default_markdown_generator(options, normalized_source, filter_obj):
        return None
    return DefaultMarkdownGenerator(content_filter=filter_obj, options=options, content_source=normalized_source)


def _normalized_extraction_strategy_type(strategy_type: Optional[str]) -> Optional[str]:
    if not isinstance(strategy_type, str):
        return None
    normalized = strategy_type.lower()
    if normalized in _ALLOWED_EXTRACTION_STRATEGIES:
        return normalized
    return None


def _schema_required_strategy(strategy_type: str, schema: Optional[Dict[str, Any]]) -> bool:
    if strategy_type in {"css", "xpath"} and not schema:
        logger.warning(f"{strategy_type.upper()} extraction requires schema, skipping.")
        return True
    return False


def _schema_strategy(strategy_type: str, schema: Dict[str, Any]) -> Any:
    builders = {
        "css": JsonCssExtractionStrategy,
        "xpath": JsonXPathExtractionStrategy,
    }
    return builders[strategy_type](schema=schema)


def _regex_strategy(patterns: Optional[Dict[str, str]]) -> Any:
    return RegexExtractionStrategy(custom=patterns) if patterns else RegexExtractionStrategy()


def _llm_strategy(schema: Optional[Dict[str, Any]], instruction: Optional[str], llm_provider: Optional[str]) -> Any:
    llm_config = _resolved_llm_config(llm_provider)
    return LLMExtractionStrategy(
        llm_config=llm_config,
        instruction=instruction or "Extract structured data from the content.",
        schema=schema,
    )


def _schema_extraction_strategy(normalized_type: str, schema: Optional[Dict[str, Any]]) -> Optional[Any]:
    if normalized_type not in {"css", "xpath"}:
        return None
    if _schema_required_strategy(normalized_type, schema):
        return None
    return _schema_strategy(normalized_type, schema) if isinstance(schema, dict) else None


def _non_schema_extraction_strategy(
    normalized_type: str,
    patterns: Optional[Dict[str, str]],
    schema: Optional[Dict[str, Any]],
    instruction: Optional[str],
    llm_provider: Optional[str],
) -> Optional[Any]:
    strategy_builders = {
        "regex": lambda: _regex_strategy(patterns),
        "llm": lambda: _llm_strategy(schema, instruction, llm_provider),
    }
    builder = strategy_builders.get(normalized_type)
    return builder() if builder is not None else None


def _build_extraction_strategy(
    strategy_type: Optional[str] = None,
    schema: Optional[Dict[str, Any]] = None,
    patterns: Optional[Dict[str, str]] = None,
    instruction: Optional[str] = None,
    llm_provider: Optional[str] = None,
) -> Optional[Any]:
    """Build a Crawl4AI extraction strategy from safe parameters."""
    normalized_type = _normalized_extraction_strategy_type(strategy_type)
    if normalized_type is None:
        return None
    schema_strategy = _schema_extraction_strategy(normalized_type, schema)
    if schema_strategy is not None or normalized_type in {"css", "xpath"}:
        return schema_strategy
    return _non_schema_extraction_strategy(normalized_type, patterns, schema, instruction, llm_provider)


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


def _flatten_dict_content(value: Dict[Any, Any], prefix: str) -> List[str]:
    lines: List[str] = []
    for key in sorted(value.keys()):
        next_prefix = f"{prefix}.{key}" if prefix else str(key)
        lines.extend(_flatten_structured_content(value[key], next_prefix))
    return lines


def _flatten_list_content(value: List[Any], prefix: str) -> List[str]:
    item_lines: List[str] = []
    for idx, item in enumerate(value):
        next_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
        item_lines.extend(_flatten_structured_content(item, next_prefix))
    return item_lines


def _flatten_scalar_content(value: Any, prefix: str) -> List[str]:
    value_str = str(value).strip()
    if not value_str:
        return []
    return [f"{prefix}={value_str}" if prefix else value_str]


def _flatten_structured_content(value: Any, prefix: str = "") -> List[str]:
    """Flatten nested structured content to deterministic path=value lines."""
    if value is None:
        return []
    if isinstance(value, dict):
        return _flatten_dict_content(value, prefix)
    if isinstance(value, list):
        return _flatten_list_content(value, prefix)
    return _flatten_scalar_content(value, prefix)


def _normalize_list_record_item(item: Any) -> Optional[Dict[str, Any]]:
    if isinstance(item, dict):
        return item
    if item is None:
        return None
    return {"value": item}


def _normalized_list_records(extracted_content: List[Any]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for item in extracted_content:
        normalized = _normalize_list_record_item(item)
        if normalized is not None:
            records.append(normalized)
    return records


def _normalize_extraction_records(extracted_content: Any) -> List[Dict[str, Any]]:
    """Normalize extraction outputs to a stable records list contract."""
    if extracted_content is None:
        return []
    if isinstance(extracted_content, dict):
        return [extracted_content]
    if isinstance(extracted_content, list):
        return _normalized_list_records(extracted_content)
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

    fields = _sample_schema_fields(sample_html, selector_for, normalized_strategy)
    if not fields:
        fields.append(_fallback_sample_schema_field(selector_for, normalized_strategy))
    return _sample_schema_payload(fields, normalized_strategy)


def _sample_schema_fields(sample_html: str, selector_for: str, normalized_strategy: str) -> List[Dict[str, Any]]:
    candidates = [
        ("title", "title", "/html/head/title/text()"),
        ("heading", "h1", "//h1[1]/text()"),
        ("paragraph", "p", "//p[1]/text()"),
        ("link", "a", "//a[1]/@href"),
    ]
    return [
        _sample_schema_field(name, css_selector, xpath_selector, selector_for, normalized_strategy)
        for name, css_selector, xpath_selector in candidates
        if _sample_schema_candidate_present(sample_html, css_selector)
    ]


def _fallback_sample_schema_field(selector_for: str, normalized_strategy: str) -> Dict[str, Any]:
    fallback_field: Dict[str, Any] = {"name": "content", "type": "text"}
    fallback_field[selector_for] = "//body" if normalized_strategy == "xpath" else "body"
    return fallback_field


def _sample_schema_payload(fields: List[Dict[str, Any]], normalized_strategy: str) -> Dict[str, Any]:
    schema: Dict[str, Any] = {"fields": fields}
    if normalized_strategy == "css":
        schema["baseSelector"] = "body"
    return schema


def _sample_schema_candidate_present(sample_html: str, css_selector: str) -> bool:
    pattern = rf"<\s*{css_selector}\b" if css_selector != "title" else r"<\s*title\b"
    return bool(re.search(pattern, sample_html, flags=re.IGNORECASE))


def _sample_schema_field(
    name: str,
    css_selector: str,
    xpath_selector: str,
    selector_for: str,
    normalized_strategy: str,
) -> Dict[str, Any]:
    field: Dict[str, Any] = {"name": name, "type": "text"}
    field[selector_for] = xpath_selector if normalized_strategy == "xpath" else css_selector
    return field


def _schema_validation_result(
    valid: bool,
    errors: List[str],
    normalized_schema: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    return {"valid": valid, "errors": errors, "normalized_schema": normalized_schema}


def _normalize_strategy_for_schema(strategy: str) -> Optional[str]:
    normalized_strategy = (strategy or "css").strip().lower()
    if normalized_strategy in _ALLOWED_SCHEMA_STRATEGIES:
        return normalized_strategy
    return None


def _selector_key_for_strategy(normalized_strategy: str) -> str:
    return "xpath" if normalized_strategy == "xpath" else "selector"


def _missing_schema_field_errors(name: str, selector: str, selector_key: str) -> List[str]:
    errors: List[str] = []
    if not name:
        errors.append("name is required")
    if not selector:
        errors.append(f"{selector_key} is required")
    return errors


def _normalized_schema_field(field: Dict[str, Any], selector_key: str) -> tuple[Optional[Dict[str, Any]], List[str]]:
    name = str(field.get("name") or "").strip()
    selector = str(field.get(selector_key) or "").strip()
    errors = _missing_schema_field_errors(name, selector, selector_key)
    if errors:
        return None, errors
    return {
        "name": name,
        selector_key: selector,
        "type": str(field.get("type") or "text"),
    }, []


def _collect_normalized_fields(fields: List[Any], selector_key: str) -> tuple[List[Dict[str, Any]], List[str]]:
    normalized_fields: List[Dict[str, Any]] = []
    errors: List[str] = []
    for index, field in enumerate(fields):
        if not isinstance(field, dict):
            errors.append(f"fields[{index}] must be an object")
            continue
        normalized_field, field_errors = _normalized_schema_field(field, selector_key)
        if normalized_field is not None:
            normalized_fields.append(normalized_field)
        for error in field_errors:
            errors.append(f"fields[{index}].{error}")
    return normalized_fields, errors


def _validate_generated_schema(schema: Optional[Dict[str, Any]], strategy: str = "css") -> Dict[str, Any]:
    """Validate extraction schema structure for css/xpath strategies."""
    normalized_strategy = _normalize_strategy_for_schema(strategy)
    if normalized_strategy is None:
        return _schema_validation_result(False, ["Only css/xpath schemas are supported."], None)
    invalid_result = _invalid_schema_result(schema)
    if invalid_result is not None:
        return invalid_result
    typed_schema = cast(Dict[str, Any], schema)

    selector_key = _selector_key_for_strategy(normalized_strategy)
    normalized_fields, errors = _collect_normalized_fields(cast(List[Any], typed_schema.get("fields")), selector_key)
    if errors:
        return _schema_validation_result(False, errors, None)
    return _schema_validation_result(
        True,
        [],
        _validated_schema_payload(typed_schema, normalized_fields, normalized_strategy),
    )


def _invalid_schema_result(schema: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if isinstance(schema, dict):
        fields = schema.get("fields")
        if isinstance(fields, list) and fields:
            return None
        return _schema_validation_result(False, ["schema.fields must be a non-empty list."], None)
    return _schema_validation_result(False, ["schema must be a dictionary."], None)


def _validated_schema_payload(
    schema: Dict[str, Any],
    normalized_fields: List[Dict[str, Any]],
    normalized_strategy: str,
) -> Dict[str, Any]:
    normalized_schema: Dict[str, Any] = {"fields": normalized_fields}
    if normalized_strategy == "css":
        normalized_schema["baseSelector"] = str(schema.get("baseSelector") or "body")
    return normalized_schema


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


def _normalized_projection_mode(projection_mode: str) -> str:
    normalized_mode = projection_mode.strip().lower() if isinstance(projection_mode, str) else "hybrid"
    if normalized_mode in _ALLOWED_STRUCTURED_PROJECTION_MODES:
        return normalized_mode
    return "hybrid"


def _project_structured_content(
    structured_content: Any,
    projection_mode: str,
) -> tuple[str, Dict[str, Any]]:
    """Project structured content for vector indexing while preserving raw JSON."""
    normalized_mode = _normalized_projection_mode(projection_mode)
    raw_json_text = json.dumps(structured_content, ensure_ascii=False, indent=2)
    flattened_text = "\n".join(_flatten_structured_content(structured_content))

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

    records = [_adaptive_export_record(doc) for doc in crawled_docs]
    exporter = _adaptive_exporter(normalized_format)
    return exporter(records)


def _adaptive_exporter(format_name: str) -> Callable[[List[Dict[str, Any]]], Dict[str, Any]]:
    return {
        "jsonl": _adaptive_export_jsonl,
        "markdown": _adaptive_export_markdown,
    }.get(format_name, _adaptive_export_json)


def _adaptive_export_json(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "format": "json",
        "record_count": len(records),
        "data": records,
    }


def _adaptive_export_jsonl(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "format": "jsonl",
        "record_count": len(records),
        "data": "\n".join(json.dumps(row, ensure_ascii=False) for row in records),
    }


def _adaptive_export_markdown(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    sections: List[str] = []
    for idx, row in enumerate(records, start=1):
        sections.extend(_adaptive_export_markdown_section(idx, row))
    return {
        "format": "markdown",
        "record_count": len(records),
        "data": "\n".join(sections).strip(),
    }


def _adaptive_export_markdown_section(idx: int, row: Dict[str, Any]) -> List[str]:
    return [
        f"## {idx}. {row.get('url')}",
        f"- depth: {row.get('depth')}",
        f"- variant: {row.get('selected_variant')}",
        "",
        row.get("markdown") or "",
        "",
    ]


def _adaptive_export_record(doc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "url": doc.get("url"),
        "depth": doc.get("depth"),
        "selected_variant": doc.get("selected_variant"),
        "markdown": doc.get("markdown"),
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
    supporting: List[Dict[str, Any]] = [_adaptive_supporting_result(row) for row in rows]
    return {
        "query": query,
        "answer": answer,
        "supporting_results": supporting,
    }


def _adaptive_supporting_result(row: Dict[str, Any]) -> Dict[str, Any]:
    metadata = row.get("page_metadata") if isinstance(row.get("page_metadata"), dict) else {}
    return {
        "url": row.get("url"),
        "similarity": row.get("similarity_score"),
        "metadata": metadata,
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
    if _is_prefixed_url(url, "file://"):
        return "local_file"
    if _is_prefixed_url(url, "raw:"):
        return "raw_html"
    return "remote_url"


def _is_prefixed_url(url: Any, prefix: str) -> bool:
    return isinstance(url, str) and url.startswith(prefix)


def _json_safe_artifact(value: Any) -> Any:
    """Return a JSON-safe artifact value; coerce unknown objects to None."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool, dict, list)):
        return value
    return None


def _canonical_url_key(url: Any) -> str:
    """Return a deterministic canonical URL key for safeguard grouping."""
    normalized_url = _normalized_non_empty_string(url)
    if normalized_url is None:
        return ""
    parsed = urlparse(normalized_url)
    if parsed.scheme or parsed.netloc:
        cleaned = parsed._replace(fragment="")
        return cleaned.geturl()
    return parsed.path or normalized_url


def _normalized_non_empty_string(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _extract_source_change_id(crawl_result: Any) -> Optional[str]:
    """Extract optional ETag/Last-Modified identifier from crawl result headers."""
    headers = getattr(crawl_result, "response_headers", None)
    if not isinstance(headers, dict):
        headers = getattr(crawl_result, "headers", None)
    if not isinstance(headers, dict):
        return None

    normalized = _normalized_header_map(headers)
    return _source_change_id_from_headers(normalized)


def _normalized_header_map(headers: Dict[Any, Any]) -> Dict[str, str]:
    normalized: Dict[str, str] = {}
    for key, value in headers.items():
        if key is None or value is None:
            continue
        normalized[str(key).lower()] = str(value)
    return normalized


def _source_change_id_from_headers(normalized: Dict[str, str]) -> Optional[str]:
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
        return _datetime_to_utc(value)
    if isinstance(value, str):
        return _parse_datetime_string(value)
    return None


def _datetime_to_utc(value: datetime) -> datetime:
    if value.tzinfo:
        return value
    return value.replace(tzinfo=timezone.utc)


def _parse_datetime_string(value: str) -> Optional[datetime]:
    normalized = value.strip()
    if not normalized:
        return None
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    return _datetime_to_utc(parsed)


def _is_result_fresh(
    metadata: Dict[str, Any],
    staleness_threshold: float,
    as_of_dt: Optional[datetime],
    require_fresh: bool = True,
) -> bool:
    """Determine whether a result passes freshness criteria."""
    if not _passes_freshness_threshold(metadata, staleness_threshold, require_fresh):
        return False
    return _passes_as_of_filter(metadata, as_of_dt)


def _passes_freshness_threshold(
    metadata: Dict[str, Any],
    staleness_threshold: float,
    require_fresh: bool,
) -> bool:
    if not require_fresh:
        return True
    if _is_staler_than_threshold(metadata.get("staleness_score"), staleness_threshold):
        return False
    if _is_expired(metadata.get("expires_at")):
        return False
    return True


def _is_staler_than_threshold(staleness_value: Any, staleness_threshold: float) -> bool:
    if not isinstance(staleness_value, (int, float)):
        return False
    return float(staleness_value) > staleness_threshold


def _is_expired(expires_at_value: Any) -> bool:
    expires_at = _parse_datetime_utc(expires_at_value)
    if expires_at is None:
        return False
    return expires_at <= datetime.now(timezone.utc)


def _passes_as_of_filter(metadata: Dict[str, Any], as_of_dt: Optional[datetime]) -> bool:
    if as_of_dt is None:
        return True
    crawl_ts = _parse_datetime_utc(metadata.get("crawl_timestamp") or metadata.get("crawl_time"))
    if crawl_ts is None:
        return True
    return crawl_ts <= as_of_dt


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
    active_counts = _candidate_source_counts(candidates)

    selected_by_source: Dict[str, int] = {}
    filtered: List[Dict[str, Any]] = []
    for c in sorted(candidates, key=_eviction_sort_key):
        src = _candidate_source(c)
        min_docs = int(source_policy_map.get(src, 0))
        selected_count = selected_by_source.get(src, 0)
        if min_docs > 0 and (active_counts.get(src, 0) - selected_count) <= min_docs:
            continue
        filtered.append(c)
        selected_by_source[src] = selected_count + 1

    return filtered


def _candidate_source(candidate: Dict[str, Any]) -> str:
    return str(candidate.get("source") or "")


def _candidate_source_counts(candidates: List[Dict[str, Any]]) -> Dict[str, int]:
    active_counts: Dict[str, int] = {}
    for candidate in candidates:
        src = _candidate_source(candidate)
        active_counts[src] = active_counts.get(src, 0) + 1
    return active_counts


def _build_active_coverage_maps(session: Any) -> tuple[Dict[str, int], Dict[tuple[str, str], int]]:
    """Build active coverage maps for source and canonical URL safeguards."""
    source_counts: Dict[str, int] = {}
    canonical_counts: Dict[tuple[str, str], int] = {}

    for model_cls in (CrawledPage, CodeExample):
        for row in _active_coverage_rows(session, model_cls):
            _update_coverage_maps(source_counts, canonical_counts, row)

    return source_counts, canonical_counts


def _active_coverage_rows(session: Any, model_cls: Any) -> List[Any]:
    model_any = cast(Any, model_cls)
    return list(
        session.exec(
            select(model_cls).where(
                cast(Any, model_any.is_active).is_(True),
                model_any.tombstoned_at.is_(None),
            )
        ).all()
    )


def _update_coverage_maps(
    source_counts: Dict[str, int],
    canonical_counts: Dict[tuple[str, str], int],
    row: Any,
) -> None:
    metadata = _record_metadata(row)
    source = str(metadata.get("source") or "")
    canonical = _canonical_url_key(metadata.get("canonical_url") or row.url)
    source_counts[source] = source_counts.get(source, 0) + 1
    key = (source, canonical)
    canonical_counts[key] = canonical_counts.get(key, 0) + 1


def _candidate_scope(candidate: Dict[str, Any]) -> tuple[str, str, bool, tuple[str, str]]:
    source = str(candidate.get("source") or "")
    canonical_guard_enabled = bool(candidate.get("canonical_guard", False))
    canonical_key = _canonical_url_key(candidate.get("canonical_key") or candidate.get("url"))
    return source, canonical_key, canonical_guard_enabled, (source, canonical_key)


def _passes_source_safeguard(
    source: str,
    source_policy_map: Dict[str, int],
    projected_source_counts: Dict[str, int],
) -> tuple[bool, int]:
    min_docs = int(source_policy_map.get(source, 0))
    remaining_for_source = projected_source_counts.get(source, 0) - 1
    if min_docs > 0 and remaining_for_source < min_docs:
        return False, remaining_for_source
    return True, remaining_for_source


def _passes_canonical_safeguard(
    canonical_guard_enabled: bool,
    canonical_key: str,
    source_canonical_key: tuple[str, str],
    projected_canonical_counts: Dict[tuple[str, str], int],
) -> tuple[bool, int]:
    remaining_for_canonical = projected_canonical_counts.get(source_canonical_key, 0) - 1
    if canonical_guard_enabled and canonical_key and remaining_for_canonical < 1:
        return False, remaining_for_canonical
    return True, remaining_for_canonical


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
        if _select_eviction_candidate(
            candidate,
            source_policy_map,
            projected_source_counts,
            projected_canonical_counts,
            selected,
        ):
            continue

    return selected


def _select_eviction_candidate(
    candidate: Dict[str, Any],
    source_policy_map: Dict[str, int],
    projected_source_counts: Dict[str, int],
    projected_canonical_counts: Dict[tuple[str, str], int],
    selected: List[Dict[str, Any]],
) -> bool:
    source, canonical_key, canonical_guard_enabled, source_canonical_key = _candidate_scope(candidate)
    source_ok, remaining_for_source = _passes_source_safeguard(source, source_policy_map, projected_source_counts)
    if not source_ok:
        return False
    canonical_ok, remaining_for_canonical = _passes_canonical_safeguard(
        canonical_guard_enabled,
        canonical_key,
        source_canonical_key,
        projected_canonical_counts,
    )
    if not canonical_ok:
        return False
    selected.append(candidate)
    projected_source_counts[source] = max(0, remaining_for_source)
    if canonical_guard_enabled and canonical_key:
        projected_canonical_counts[source_canonical_key] = max(0, remaining_for_canonical)
    return True


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
    return "404" in lowered or "410" in lowered or "not found" in lowered or "gone" in lowered


def _compute_retry_backoff_hours(policy: Any) -> int:
    """Compute exponential retry backoff using per-source policy defaults."""
    failures = int(getattr(policy, "consecutive_failures", 0) or 0)
    base = max(1, int(getattr(policy, "retry_backoff_base_hours", 2) or 2))
    max_backoff = max(base, int(getattr(policy, "max_retry_backoff_hours", 168) or 168))
    if failures <= 0:
        return 0
    return min(max_backoff, base * (2 ** max(0, failures - 1)))


def _filter_rows_by_source(rows: List[Any], source: Optional[str]) -> List[Any]:
    """Filter rows by metadata source when a source filter is provided."""
    if not source:
        return rows
    return [
        row
        for row in rows
        if isinstance(getattr(row, "page_metadata", None), dict) and row.page_metadata.get("source") == source
    ]


def _classify_staleness_row_ids(
    active_rows: List[Any],
    minor_cut: float,
    major_cut: float,
) -> tuple[List[int], List[int], List[int]]:
    """Classify active rows into unchanged/minor/major buckets by staleness score."""
    id_scores = _row_id_scores(active_rows)
    major = _row_ids_by_score(id_scores, lambda score: score >= major_cut)
    minor = _row_ids_by_score(id_scores, lambda score: minor_cut <= score < major_cut)
    unchanged = _row_ids_by_score(id_scores, lambda score: score < minor_cut)
    return unchanged, minor, major


def _row_id_scores(active_rows: List[Any]) -> List[tuple[int, float]]:
    id_scores: List[tuple[int, float]] = []
    for row in active_rows:
        row_id = getattr(row, "id", None)
        if not isinstance(row_id, int):
            continue
        id_scores.append((row_id, float(getattr(row, "staleness_score", 0.0) or 0.0)))
    return id_scores


def _row_ids_by_score(id_scores: List[tuple[int, float]], predicate: Any) -> List[int]:
    return [row_id for row_id, score in id_scores if predicate(score)]


def _source_quota_candidate(row: Any, table_name: str) -> Optional[Dict[str, Any]]:
    row_id = getattr(row, "id", None)
    if not isinstance(row_id, int):
        return None
    return {
        "table": table_name,
        "id": row_id,
        "value_score": _row_float_value(row, "value_score"),
        "staleness_score": _row_float_value(row, "staleness_score"),
        "hit_count": _row_int_value(row, "hit_count"),
        "last_seen_at": _row_iso_timestamp(row, "last_seen_at"),
        "size_bytes": _estimate_record_size_bytes(row),
    }


def _row_float_value(row: Any, attr: str) -> float:
    return float(getattr(row, attr, 0.0) or 0.0)


def _row_int_value(row: Any, attr: str) -> int:
    return int(getattr(row, attr, 0) or 0)


def _row_iso_timestamp(row: Any, attr: str) -> Optional[str]:
    parsed_value = _parse_datetime_utc(getattr(row, attr, None))
    return parsed_value.isoformat() if parsed_value is not None else None


def _active_unpinned_rows(session: Any, model_cls: Any) -> List[Any]:
    model_any = cast(Any, model_cls)
    return session.exec(
        select(model_cls).where(
            cast(Any, model_any.is_active).is_(True),
            model_any.tombstoned_at.is_(None),
            cast(Any, model_any.is_pinned).is_(False),
        )
    ).all()


def _rows_by_source_for_quotas(session: Any, source_policies: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    rows_by_source: Dict[str, List[Dict[str, Any]]] = {}
    for model_cls, table_name in ((CrawledPage, "crawled_pages"), (CodeExample, "code_examples")):
        for row in _active_unpinned_rows(session, model_cls):
            _append_source_quota_candidate(rows_by_source, source_policies, row, table_name)
    return rows_by_source


def _append_source_quota_candidate(
    rows_by_source: Dict[str, List[Dict[str, Any]]],
    source_policies: Dict[str, Any],
    row: Any,
    table_name: str,
) -> None:
    source_name = str(_record_metadata(row).get("source") or "")
    if source_name not in source_policies:
        return
    candidate = _source_quota_candidate(row, table_name)
    if candidate is None:
        return
    rows_by_source.setdefault(source_name, []).append(candidate)


def _selected_quota_candidates(records: List[Dict[str, Any]], quota_bytes: int) -> List[Dict[str, Any]]:
    used = sum(record["size_bytes"] for record in records)
    if used <= quota_bytes:
        return []
    selected: List[Dict[str, Any]] = []
    for candidate in sorted(records, key=_eviction_sort_key):
        if used <= quota_bytes:
            break
        selected.append(candidate)
        used -= candidate["size_bytes"]
    return selected


def _tombstone_source_quota_candidates(session: Any, selected: List[Dict[str, Any]]) -> int:
    return _tombstone_candidate_groups(session, selected, "source_quota_prune")


def _tombstone_candidate_groups(session: Any, selected: List[Dict[str, Any]], reason: str) -> int:
    page_ids, code_ids = _candidate_ids_by_table(selected)
    return _tombstone_candidate_ids(session, page_ids, code_ids, reason)


def _enforce_source_quotas(session: Any, source_policies: Dict[str, Any]) -> Dict[str, Any]:
    """Enforce per-source quotas by tombstoning lowest-value rows first."""
    if not source_policies:
        return {"quota_evicted": 0, "sources_over_quota": []}

    rows_by_source = _rows_by_source_for_quotas(session, source_policies)
    over_quota_sources: List[str] = []
    total_evicted = 0
    for source_name, policy in source_policies.items():
        selected = _source_quota_selection(rows_by_source, source_name, policy)
        if not selected:
            continue
        over_quota_sources.append(source_name)
        total_evicted += _tombstone_source_quota_candidates(session, selected)

    return {"quota_evicted": total_evicted, "sources_over_quota": over_quota_sources}


def _source_quota_selection(
    rows_by_source: Dict[str, List[Dict[str, Any]]],
    source_name: str,
    policy: Any,
) -> List[Dict[str, Any]]:
    quota_mb = getattr(policy, "max_source_size_mb", None)
    if not isinstance(quota_mb, int) or quota_mb <= 0:
        return []
    return _selected_quota_candidates(rows_by_source.get(source_name, []), quota_mb * 1024 * 1024)


def _table_budget_rows(session: Any, model_cls: Any) -> List[Any]:
    model_any = cast(Any, model_cls)
    return list(
        session.exec(
            select(model_cls).where(
                cast(Any, model_any.is_active).is_(True),
                model_any.tombstoned_at.is_(None),
                cast(Any, model_any.is_pinned).is_(False),
            )
        ).all()
    )


def _table_budget_candidates(rows: List[Any]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for row in rows:
        row_id = getattr(row, "id", None)
        if not isinstance(row_id, int):
            continue
        candidates.append(_table_budget_candidate(row, row_id))
    return candidates


def _table_budget_candidate(row: Any, row_id: int) -> Dict[str, Any]:
    parsed_last_seen = _parse_datetime_utc(getattr(row, "last_seen_at", None))
    return {
        "id": row_id,
        "value_score": float(getattr(row, "value_score", 0.0) or 0.0),
        "staleness_score": float(getattr(row, "staleness_score", 0.0) or 0.0),
        "hit_count": int(getattr(row, "hit_count", 0) or 0),
        "last_seen_at": parsed_last_seen.isoformat() if parsed_last_seen is not None else None,
        "size_bytes": _estimate_record_size_bytes(row),
    }


def _table_budget_tombstone_ids(candidates: List[Dict[str, Any]], current_bytes: int, limit_bytes: int) -> List[int]:
    to_tombstone: List[int] = []
    remaining_bytes = current_bytes
    for candidate in sorted(candidates, key=_eviction_sort_key):
        if remaining_bytes <= limit_bytes:
            break
        to_tombstone.append(candidate["id"])
        remaining_bytes -= int(candidate.get("size_bytes") or 0)
    return to_tombstone


def _enforce_single_table_budget(
    session: Any,
    model_cls: Any,
    table_name: str,
    max_mb: Optional[int],
) -> int:
    if not isinstance(max_mb, int) or max_mb <= 0:
        return 0
    limit_bytes = max_mb * 1024 * 1024
    rows = _table_budget_rows(session, model_cls)
    current_bytes = sum(_estimate_record_size_bytes(row) for row in rows)
    if current_bytes <= limit_bytes:
        return 0
    candidates = _table_budget_candidates(rows)
    to_tombstone = _table_budget_tombstone_ids(candidates, current_bytes, limit_bytes)
    return _tombstone_table_budget_rows(session, to_tombstone, table_name)


def _tombstone_table_budget_rows(session: Any, row_ids: List[int], table_name: str) -> int:
    if not row_ids:
        return 0
    return tombstone_records(session, row_ids, table_name, f"table_budget_prune:{table_name}")


def _enforce_table_budgets(
    session: Any,
    max_crawled_pages_mb: Optional[int],
    max_code_examples_mb: Optional[int],
) -> Dict[str, int]:
    """Enforce per-table budgets using approximate content+metadata size."""
    return {
        "crawled_pages": _enforce_single_table_budget(session, CrawledPage, "crawled_pages", max_crawled_pages_mb),
        "code_examples": _enforce_single_table_budget(session, CodeExample, "code_examples", max_code_examples_mb),
    }


def _ttl_days_for_source(source_policies: Dict[str, Any], source: str) -> int:
    policy = source_policies.get(source)
    if policy is None:
        return 90
    return int(getattr(policy, "ttl_days", 90))


def _ttl_reference_time(row: Any, now: datetime) -> datetime:
    return (
        _parse_datetime_utc(getattr(row, "last_seen_at", None))
        or _parse_datetime_utc(getattr(row, "first_seen_at", None))
        or _parse_datetime_utc(getattr(row, "crawl_timestamp", None))
        or now
    )


def _row_exceeds_ttl(row: Any, source_policies: Dict[str, Any], now: datetime) -> bool:
    source = str(_record_metadata(row).get("source") or "")
    ttl_days = _ttl_days_for_source(source_policies, source)
    if ttl_days <= 0:
        return False
    return now - _ttl_reference_time(row, now) > timedelta(days=ttl_days)


def _ttl_rows(session: Any, model_cls: Any) -> List[Any]:
    model_any = cast(Any, model_cls)
    return list(
        session.exec(
            select(model_cls).where(
                cast(Any, model_any.is_active).is_(True),
                model_any.tombstoned_at.is_(None),
            )
        ).all()
    )


def _apply_hard_ttl_delete(session: Any, source_policies: Dict[str, Any]) -> Dict[str, int]:
    """Hard-delete active records that exceeded per-source TTL."""
    now = datetime.now(timezone.utc)
    deleted = {"crawled_pages": 0, "code_examples": 0}
    for model_cls, table_name in ((CrawledPage, "crawled_pages"), (CodeExample, "code_examples")):
        deleted[table_name] += _delete_expired_ttl_rows(session, model_cls, source_policies, now)

    if deleted["crawled_pages"] or deleted["code_examples"]:
        session.commit()
    return deleted


def _delete_expired_ttl_rows(session: Any, model_cls: Any, source_policies: Dict[str, Any], now: datetime) -> int:
    deleted = 0
    for row in _ttl_rows(session, model_cls):
        if not _row_exceeds_ttl(row, source_policies, now):
            continue
        session.delete(row)
        deleted += 1
    return deleted


_ScopedRow = tuple[Any, Dict[str, Any]]


def _active_rows(session: Any, model_cls: Any) -> List[Any]:
    model_any = cast(Any, model_cls)
    return session.exec(
        select(model_cls).where(
            cast(Any, model_any.is_active).is_(True),
            model_any.tombstoned_at.is_(None),
        )
    ).all()


def _scoped_active_rows(session: Any, model_cls: Any, source: str) -> List[_ScopedRow]:
    scoped: List[_ScopedRow] = []
    for row in _active_rows(session, model_cls):
        metadata = _record_metadata(row)
        if str(metadata.get("source") or "") != source:
            continue
        scoped.append((row, metadata))
    return scoped


def _group_rows_by_canonical(scoped_rows: List[_ScopedRow]) -> Dict[str, List[_ScopedRow]]:
    grouped: Dict[str, List[_ScopedRow]] = {}
    for row, metadata in scoped_rows:
        canonical = _canonical_url_key(metadata.get("canonical_url") or row.url)
        grouped.setdefault(canonical, []).append((row, metadata))
    return grouped


def _canonical_row_sort_key(pair: _ScopedRow) -> tuple[datetime, float, int]:
    row = pair[0]
    return (
        _parse_datetime_utc(getattr(row, "last_crawled_at", None)) or datetime(1970, 1, 1, tzinfo=timezone.utc),
        float(getattr(row, "value_score", 0.0) or 0.0),
        int(getattr(row, "hit_count", 0) or 0),
    )


def _classify_duplicate_and_superseded_ids(
    canonical_rows: List[_ScopedRow],
) -> tuple[List[int], List[int]]:
    sorted_rows = sorted(canonical_rows, key=_canonical_row_sort_key, reverse=True)
    if not sorted_rows:
        return [], []
    survivor = sorted_rows[0][0]
    seen_hashes = {str(getattr(survivor, "content_hash", "") or "")}
    duplicate_ids: List[int] = []
    superseded_ids: List[int] = []
    for row, _metadata in sorted_rows[1:]:
        _accumulate_duplicate_or_superseded(row, seen_hashes, duplicate_ids, superseded_ids)
    return duplicate_ids, superseded_ids


def _accumulate_duplicate_or_superseded(
    row: Any,
    seen_hashes: set[str],
    duplicate_ids: List[int],
    superseded_ids: List[int],
) -> None:
    classification = _duplicate_classification(row, seen_hashes)
    if classification is None:
        return
    row_id, row_hash, is_duplicate = classification
    _append_classified_row_id(row_id, duplicate_ids if is_duplicate else superseded_ids)
    _track_surviving_row_hash(seen_hashes, row_hash, is_duplicate)


def _is_duplicate_row_hash(row_hash: str, seen_hashes: set[str]) -> bool:
    return bool(row_hash) and row_hash in seen_hashes


def _append_classified_row_id(row_id: int, target_ids: List[int]) -> None:
    target_ids.append(row_id)


def _duplicate_classification(row: Any, seen_hashes: set[str]) -> Optional[tuple[int, str, bool]]:
    row_id = getattr(row, "id", None)
    if not isinstance(row_id, int):
        return None
    row_hash = str(getattr(row, "content_hash", "") or "")
    return row_id, row_hash, _is_duplicate_row_hash(row_hash, seen_hashes)


def _track_surviving_row_hash(seen_hashes: set[str], row_hash: str, is_duplicate: bool) -> None:
    if not is_duplicate and row_hash:
        seen_hashes.add(row_hash)


def _accumulate_retire_candidates(
    session: Any,
    model_cls: Any,
    source: str,
) -> tuple[List[int], List[int]]:
    duplicate_ids: List[int] = []
    superseded_ids: List[int] = []
    grouped = _group_rows_by_canonical(_scoped_active_rows(session, model_cls, source))
    for canonical_rows in grouped.values():
        dup_ids, sup_ids = _classify_duplicate_and_superseded_ids(canonical_rows)
        duplicate_ids.extend(dup_ids)
        superseded_ids.extend(sup_ids)
    return duplicate_ids, superseded_ids


def _retire_ids(session: Any, ids: List[int], table: str, reason: str) -> int:
    if not ids:
        return 0
    return tombstone_records(session, ids, table, reason)


def _retire_source_duplicates_and_superseded(session: Any, source: str) -> Dict[str, int]:
    """Retire duplicate/superseded active records for a source."""
    duplicate_page_ids, superseded_page_ids = _accumulate_retire_candidates(session, CrawledPage, source)
    duplicate_code_ids, superseded_code_ids = _accumulate_retire_candidates(session, CodeExample, source)

    retired_duplicates = _retire_ids(session, duplicate_page_ids, "crawled_pages", "duplicate_content_hash")
    retired_duplicates += _retire_ids(session, duplicate_code_ids, "code_examples", "duplicate_content_hash")
    retired_superseded = _retire_ids(session, superseded_page_ids, "crawled_pages", "superseded_canonical_url")
    retired_superseded += _retire_ids(session, superseded_code_ids, "code_examples", "superseded_canonical_url")
    return {"duplicate_retired": retired_duplicates, "superseded_retired": retired_superseded}


async def _run_selective_reembed(session: Any, row_ids: List[int]) -> int:
    """Re-embed selected crawled page records and refresh freshness metadata."""
    if not row_ids:
        return 0

    rows = session.exec(select(CrawledPage).where(cast(Any, CrawledPage.id).in_(row_ids))).all()
    updated = 0
    now = datetime.now(timezone.utc)
    for row in rows:
        updated += await _maybe_reembed_row(row, now)
    if updated:
        session.commit()
    return updated


async def _maybe_reembed_row(row: Any, now: datetime) -> int:
    content = getattr(row, "content", "")
    if not isinstance(content, str) or not content.strip():
        return 0
    try:
        row.embedding = await create_embedding(content)
        row.last_crawled_at = now
        row.staleness_score = 0.0
        return 1
    except Exception as exc:  # pragma: no cover - external provider errors are non-deterministic
        logger.warning(f"selective re-embed failed for row id={row.id}: {exc}")
        return 0


_DEEP_CRAWL_STRATEGIES = {"bfs", "dfs", "best_first"}


def _clamped_deep_limits(max_depth: int, max_pages: int) -> tuple[int, int]:
    return max(1, min(10, max_depth)), max(1, min(500, max_pages))


def _deep_crawl_filters(
    url_pattern: Optional[str],
    allowed_domains: Optional[List[str]],
    content_types: Optional[List[str]],
    relevance_query: Optional[str],
    relevance_threshold: Optional[float],
    seo_threshold: Optional[float],
    seo_keywords: Optional[List[str]],
) -> List[Any]:
    filters: List[Any] = []
    filters.extend(_optional_deep_filters(url_pattern, allowed_domains, content_types))
    relevance_filter = _relevance_filter(relevance_query, relevance_threshold)
    if relevance_filter is not None:
        filters.append(relevance_filter)
    seo_filter = _seo_filter(seo_threshold, seo_keywords)
    if seo_filter is not None:
        filters.append(seo_filter)
    return filters


def _optional_deep_filters(
    url_pattern: Optional[str],
    allowed_domains: Optional[List[str]],
    content_types: Optional[List[str]],
) -> List[Any]:
    filters: List[Any] = []
    if url_pattern:
        filters.append(URLPatternFilter(patterns=url_pattern, use_glob=True))
    if allowed_domains:
        filters.append(DomainFilter(allowed_domains=allowed_domains))
    if content_types:
        filters.append(ContentTypeFilter(allowed_types=content_types))
    return filters


def _relevance_filter(relevance_query: Optional[str], relevance_threshold: Optional[float]) -> Optional[Any]:
    if relevance_query and isinstance(relevance_threshold, (int, float)):
        return ContentRelevanceFilter(query=relevance_query, threshold=float(relevance_threshold))
    return None


def _seo_filter(seo_threshold: Optional[float], seo_keywords: Optional[List[str]]) -> Optional[Any]:
    if isinstance(seo_threshold, (int, float)):
        return SEOFilter(threshold=float(seo_threshold), keywords=seo_keywords)
    return None


def _deep_filter_chain(
    url_pattern: Optional[str],
    allowed_domains: Optional[List[str]],
    content_types: Optional[List[str]],
    relevance_query: Optional[str],
    relevance_threshold: Optional[float],
    seo_threshold: Optional[float],
    seo_keywords: Optional[List[str]],
) -> FilterChain:
    filters = _deep_crawl_filters(
        url_pattern,
        allowed_domains,
        content_types,
        relevance_query,
        relevance_threshold,
        seo_threshold,
        seo_keywords,
    )
    return FilterChain(filters=filters) if filters else FilterChain()


def _normalized_scorer_type(scorer_type: str) -> str:
    normalized = (scorer_type or "keyword").lower().strip()
    return normalized if normalized in _ALLOWED_SCORER_TYPES else "keyword"


def _deep_strategy_builder(strategy: str) -> Any:
    builders = {
        "bfs": BFSDeepCrawlStrategy,
        "dfs": DFSDeepCrawlStrategy,
        "best_first": BestFirstCrawlingStrategy,
    }
    return builders.get(strategy, BFSDeepCrawlStrategy)


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
    """Build a Crawl4AI deep crawl strategy with optional filter chain and scorer."""
    clamped_depth, clamped_pages = _clamped_deep_limits(max_depth, max_pages)
    filter_chain = _deep_filter_chain(
        url_pattern,
        allowed_domains,
        content_types,
        relevance_query,
        relevance_threshold,
        seo_threshold,
        seo_keywords,
    )
    scorer = build_url_scorer(_normalized_scorer_type(scorer_type), keywords=keywords)
    strategy_lower = (strategy or "bfs").lower().strip()
    builder = _deep_strategy_builder(strategy_lower)
    return builder(
        max_depth=clamped_depth,
        filter_chain=filter_chain,
        url_scorer=scorer,
        include_external=include_external,
        score_threshold=score_threshold,
        max_pages=clamped_pages,
    )


def _get_crawler(ctx: Context):
    """Retrieve the AsyncWebCrawler from lifespan context."""
    lc = ctx.lifespan_context
    if lc is None or not hasattr(lc, "crawler"):
        raise RuntimeError("Crawler not initialized — lifespan context missing.")
    return lc.crawler


def _markdown_variant_key(markdown_variant: str) -> str:
    variant_map = {
        "raw": "raw_markdown",
        "fit": "fit_markdown",
        "cited": "markdown_with_citations",
        "references": "references_markdown",
    }
    return variant_map.get(markdown_variant.lower(), "raw_markdown")


async def _crawl_results_for_markdown(
    crawler: Any,
    url: str,
    config: Any,
    follow_links: bool,
    max_depth: int,
    validated_link_filter: Optional[str],
) -> tuple[List[Any], int, Optional[str]]:
    if follow_links and max_depth > 1:
        return await _recursive_markdown_results(crawler, url, max_depth, validated_link_filter)

    result = await crawler.arun(url=url, config=config)
    return _single_markdown_result(result)


async def _recursive_markdown_results(
    crawler: Any,
    url: str,
    max_depth: int,
    validated_link_filter: Optional[str],
) -> tuple[List[Any], int, Optional[str]]:
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
    return results, len(results), None


def _single_markdown_result(result: Any) -> tuple[List[Any], int, Optional[str]]:
    if result.success and result.markdown:
        return [result], 1, None
    crawl_error = getattr(result, "error_message", None) or "No content."
    return [], 0, crawl_error


def _result_url(result: Any, fallback_url: str) -> str:
    result_url = getattr(result, "url", None)
    if isinstance(result_url, str):
        return result_url
    return fallback_url


def _result_to_doc_entry(
    result: Any,
    variant_key: str,
    fallback_url: str,
    strategy_applied: bool,
) -> Optional[Dict[str, Any]]:
    if not _result_has_markdown(result):
        return None
    variants = _result_markdown_variants(result)
    selected = _selected_variant_markdown(variants, variant_key)
    if not selected:
        return None
    doc_entry = _result_doc_entry(result, fallback_url, selected, variants)
    extracted_content = getattr(result, "extracted_content", None)
    if strategy_applied and extracted_content:
        doc_entry["extraction_result"] = extracted_content
    return doc_entry


def _result_has_markdown(result: Any) -> bool:
    return bool(getattr(result, "success", False)) and bool(getattr(result, "markdown", None))


def _result_markdown_variants(result: Any) -> Dict[str, str]:
    return _extract_markdown_variants(getattr(result, "markdown"))


def _result_doc_entry(
    result: Any,
    fallback_url: str,
    selected: str,
    variants: Dict[str, str],
) -> Dict[str, Any]:
    return {
        "url": _result_url(result, fallback_url),
        "markdown": selected,
        "variants": list(variants.keys()),
        "variant_values": variants,
        "depth": getattr(result, "depth", 0),
        "raw_result": result,
    }


def _results_to_crawled_docs(
    results: List[Any],
    variant_key: str,
    fallback_url: str,
    strategy_applied: bool,
) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for result in results:
        doc_entry = _result_to_doc_entry(result, variant_key, fallback_url, strategy_applied)
        if doc_entry is not None:
            docs.append(doc_entry)
    return docs


def _doc_index_metadata(
    *,
    doc: Dict[str, Any],
    chunk: str,
    chunk_index: int,
    variant_to_index: str,
    session_id: Optional[str],
    max_depth: int,
    follow_links: bool,
    effective_run_id: str,
    source_change_id: Optional[str],
    effective_policy: str,
    resolved_override: Optional[str],
    doc_link_graph: Dict[str, Any],
    doc_media_metadata: Dict[str, Any],
    extraction_strategy: Optional[str],
    reference_meta: Dict[str, Any],
) -> Dict[str, Any]:
    now_iso = datetime.now(timezone.utc).isoformat()
    meta = extract_section_info(chunk)
    meta.update(
        {
            "chunk_index": chunk_index,
            "url": doc["url"],
            "source": urlparse(doc["url"]).netloc,
            "source_type": _infer_source_type(doc["url"], session_id=session_id),
            "crawl_time": now_iso,
            "crawl_timestamp": now_iso,
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
        }
    )
    meta.update(reference_meta)
    if extraction_strategy:
        meta["extraction_strategy"] = extraction_strategy
    return meta


async def _variant_index_payload(
    *,
    doc: Dict[str, Any],
    variant_to_index: str,
    selected_variant_markdown: str,
    session_id: Optional[str],
    extraction_strategy: Optional[str],
    max_depth: int,
    follow_links: bool,
    effective_run_id: str,
    resolved_override: Optional[str],
    source_change_id: Optional[str],
    effective_policy: str,
    doc_link_graph: Dict[str, Any],
    doc_media_metadata: Dict[str, Any],
    reference_meta: Dict[str, Any],
) -> tuple[List[str], List[int], List[str], List[Dict[str, Any]], List[str]]:
    db_urls: List[str] = []
    db_chunks: List[int] = []
    db_contents: List[str] = []
    db_metas: List[Dict[str, Any]] = []
    db_fulldocs: List[str] = []
    chunks = await chunk_text_according_to_settings(selected_variant_markdown)
    for chunk_index, chunk in enumerate(chunks):
        db_urls.append(doc["url"])
        db_chunks.append(chunk_index)
        db_contents.append(chunk)
        db_metas.append(
            _doc_index_metadata(
                doc=doc,
                chunk=chunk,
                chunk_index=chunk_index,
                variant_to_index=variant_to_index,
                session_id=session_id,
                max_depth=max_depth,
                follow_links=follow_links,
                effective_run_id=effective_run_id,
                source_change_id=source_change_id,
                effective_policy=effective_policy,
                resolved_override=resolved_override,
                doc_link_graph=doc_link_graph,
                doc_media_metadata=doc_media_metadata,
                extraction_strategy=extraction_strategy,
                reference_meta=reference_meta,
            )
        )
        db_fulldocs.append(selected_variant_markdown)
    return db_urls, db_chunks, db_contents, db_metas, db_fulldocs


async def _collect_index_payload_for_doc(
    doc: Dict[str, Any],
    *,
    index_variants: Optional[str],
    session_id: Optional[str],
    extraction_strategy: Optional[str],
    max_depth: int,
    follow_links: bool,
    effective_run_id: str,
    resolved_override: Optional[str],
) -> tuple[List[str], List[int], List[str], List[Dict[str, Any]], List[str], set[str], List[str]]:
    db_urls: List[str] = []
    db_chunks: List[int] = []
    db_contents: List[str] = []
    db_metas: List[Dict[str, Any]] = []
    db_fulldocs: List[str] = []
    indexed_variant_keys: set[str] = set()

    variant_values: Dict[str, str] = doc.get("variant_values", {})
    doc_link_graph = extract_link_graph(doc.get("markdown", ""), base_url=doc.get("url"))
    doc_media_metadata = extract_media_metadata(doc.get("markdown", ""))
    source_change_id = _extract_source_change_id(doc.get("raw_result"))
    variant_keys_to_index, effective_policy, _, doc_fallback_notes = _resolve_variants_to_index(
        variants=variant_values,
        index_variants=index_variants,
        fallback_enabled=bool(settings.MARKDOWN_FALLBACK_ENABLED),
    )
    for variant_to_index in variant_keys_to_index:
        selected_variant_markdown = variant_values.get(variant_to_index) or ""
        if not selected_variant_markdown:
            continue
        indexed_variant_keys.add(variant_to_index)
        reference_meta = _build_reference_metadata(variant_values)
        variant_payload = await _variant_index_payload(
            doc=doc,
            variant_to_index=variant_to_index,
            selected_variant_markdown=selected_variant_markdown,
            session_id=session_id,
            extraction_strategy=extraction_strategy,
            max_depth=max_depth,
            follow_links=follow_links,
            effective_run_id=effective_run_id,
            resolved_override=resolved_override,
            source_change_id=source_change_id,
            effective_policy=effective_policy,
            doc_link_graph=doc_link_graph,
            doc_media_metadata=doc_media_metadata,
            reference_meta=reference_meta,
        )
        db_urls.extend(variant_payload[0])
        db_chunks.extend(variant_payload[1])
        db_contents.extend(variant_payload[2])
        db_metas.extend(variant_payload[3])
        db_fulldocs.extend(variant_payload[4])

    return db_urls, db_chunks, db_contents, db_metas, db_fulldocs, indexed_variant_keys, doc_fallback_notes


def _extend_index_payload(
    payload: Dict[str, Any],
    doc_payload: tuple[List[str], List[int], List[str], List[Dict[str, Any]], List[str], set[str], List[str]],
) -> None:
    db_urls, db_chunks, db_contents, db_metas, db_fulldocs, indexed_variants, fallback_notes = doc_payload
    payload["db_urls"].extend(db_urls)
    payload["db_chunks"].extend(db_chunks)
    payload["db_contents"].extend(db_contents)
    payload["db_metas"].extend(db_metas)
    payload["db_fulldocs"].extend(db_fulldocs)
    payload["indexed_variant_keys"].update(indexed_variants)
    payload["fallback_notes"].extend(fallback_notes)


async def _index_crawled_docs(
    crawled_docs: List[Dict[str, Any]],
    *,
    index_result: bool,
    index_variants: Optional[str],
    extraction_strategy: Optional[str],
    session_id: Optional[str],
    max_depth: int,
    follow_links: bool,
    effective_run_id: str,
    resolved_override: Optional[str],
) -> tuple[int, int, set[str], List[str]]:
    pages_indexed = 0
    chunks_stored = 0
    indexed_variant_keys: set[str] = set()
    fallback_notes: List[str] = []
    if not (index_result and crawled_docs):
        return pages_indexed, chunks_stored, indexed_variant_keys, fallback_notes

    payload: Dict[str, Any] = {
        "db_urls": [],
        "db_chunks": [],
        "db_contents": [],
        "db_metas": [],
        "db_fulldocs": [],
        "indexed_variant_keys": set(),
        "fallback_notes": [],
    }
    for doc in crawled_docs:
        doc_payload = await _collect_index_payload_for_doc(
            doc,
            index_variants=index_variants,
            session_id=session_id,
            extraction_strategy=extraction_strategy,
            max_depth=max_depth,
            follow_links=follow_links,
            effective_run_id=effective_run_id,
            resolved_override=resolved_override,
        )
        _extend_index_payload(payload, doc_payload)

    if payload["db_urls"]:
        with next(get_session()) as session:
            chunks_stored = await add_documents_to_db(
                session,
                payload["db_urls"],
                payload["db_contents"],
                payload["db_metas"],
                payload["db_chunks"],
                payload["db_fulldocs"],
            )
        pages_indexed = len(set(payload["db_urls"]))
    indexed_variant_keys = cast(set[str], payload["indexed_variant_keys"])
    fallback_notes = cast(List[str], payload["fallback_notes"])
    return pages_indexed, chunks_stored, indexed_variant_keys, fallback_notes


async def _crawl_many_results(
    crawler: Any,
    urls: List[str],
    config: Any,
    *,
    follow_links: bool,
    max_depth: int,
    validated_link_filter: Optional[str],
    max_concurrent: int,
) -> List[Any]:
    if follow_links and max_depth > 1:
        logger.warning(
            "Using compatibility helper crawl_recursive_internal_links in crawl_many_urls; "
            "prefer crawl_deep for native Crawl4AI deep crawling strategies."
        )
        all_results: List[Any] = []
        for start_url in urls:
            all_results.extend(
                await crawl_recursive_internal_links(
                    crawler=crawler,
                    start_urls=[start_url],
                    max_depth=max_depth,
                    url_pattern=validated_link_filter,
                )
            )
        return all_results

    dispatcher = MemoryAdaptiveDispatcher(max_session_permit=max_concurrent)
    return await crawler.arun_many(urls=urls, config=config, dispatcher=dispatcher)


def _batch_doc_from_result(
    result: Any,
    variant_key: str,
    effective_run_id: str,
    extraction_strategy: Optional[str],
) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, str]]]:
    result_url = getattr(result, "url", "unknown")
    if not _result_has_markdown(result):
        return _batch_doc_failure(result_url, getattr(result, "error_message", "No content."))

    variants = _result_markdown_variants(result)
    selected = _selected_variant_markdown(variants, variant_key)
    if not selected:
        return _batch_doc_failure(result_url, "Empty markdown variant: no content in any variant.")

    resolved_url = result_url if isinstance(result_url, str) and result_url else "unknown"
    doc = _batch_doc_payload(result, resolved_url, selected, variants, variant_key, effective_run_id)
    return _batch_doc_with_optional_extraction(doc, result, extraction_strategy), None


def _batch_result_has_content(result: Any) -> bool:
    return bool(getattr(result, "success", False)) and bool(getattr(result, "markdown", None))


def _batch_doc_failure(result_url: Any, error_message: str) -> tuple[None, Dict[str, str]]:
    return None, _batch_doc_error(result_url, error_message)


def _batch_doc_with_optional_extraction(
    doc: Dict[str, Any],
    result: Any,
    extraction_strategy: Optional[str],
) -> Dict[str, Any]:
    extracted_content = getattr(result, "extracted_content", None)
    if extraction_strategy and extracted_content:
        doc["extraction_result"] = extracted_content
    return doc


def _selected_variant_markdown(variants: Dict[str, str], variant_key: str) -> str:
    return variants.get(variant_key) or variants["raw_markdown"]


def _batch_doc_error(result_url: Any, error: str) -> Dict[str, str]:
    return {"url": result_url, "error": error}


def _batch_doc_payload(
    result: Any,
    resolved_url: str,
    selected: str,
    variants: Dict[str, str],
    variant_key: str,
    effective_run_id: str,
) -> Dict[str, Any]:
    return {
        "url": resolved_url,
        "markdown": selected,
        "depth": getattr(result, "depth", 0),
        "variant_values": variants,
        "selected_variant": variant_key,
        "link_graph": extract_link_graph(selected, base_url=resolved_url),
        "media_metadata": extract_media_metadata(selected),
        "source_change_id": _extract_source_change_id(result),
        "run_id": effective_run_id,
    }


def _batch_docs_and_errors(
    all_results: List[Any],
    variant_key: str,
    effective_run_id: str,
    extraction_strategy: Optional[str],
) -> tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    crawled_docs: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    for result in all_results:
        doc, error = _batch_doc_from_result(result, variant_key, effective_run_id, extraction_strategy)
        if doc is not None:
            crawled_docs.append(doc)
        if error is not None:
            errors.append(error)
    return crawled_docs, errors


async def _container_to_results(container: Any, stream_mode: bool) -> List[Any]:
    if not stream_mode:
        return list(container)
    results: List[Any] = []
    async for result in container:
        results.append(result)
    return results


def _crawl_mode_label(follow_links: bool, max_depth: int) -> str:
    return "compatibility_recursive" if (follow_links and max_depth > 1) else "single_page"


def _applied_content_source(content_source: str) -> str:
    return content_source if content_source in _ALLOWED_CONTENT_SOURCES else "cleaned_html"


def _applied_content_filter(content_filter: Optional[str]) -> Optional[str]:
    if not isinstance(content_filter, str):
        return None
    normalized = content_filter.lower().strip()
    if normalized in _ALLOWED_CONTENT_FILTERS:
        return normalized
    return None


def _crawl_artifacts(first_result: Any) -> Dict[str, Any]:
    return {
        "screenshot": _json_safe_artifact(getattr(first_result, "screenshot", None) if first_result else None),
        "pdf": _json_safe_artifact(getattr(first_result, "pdf", None) if first_result else None),
        "mhtml": _json_safe_artifact(getattr(first_result, "mhtml", None) if first_result else None),
    }


def _apply_extraction_result(response: Dict[str, Any], first_doc: Dict[str, Any]) -> Dict[str, Any]:
    extraction_result = first_doc.get("extraction_result")
    if extraction_result is not None:
        response["extraction_result"] = extraction_result
    return response


def _crawl_to_markdown_response(
    *,
    url: str,
    pages_crawled: int,
    pages_indexed: int,
    chunks_stored: int,
    variant_key: str,
    first_doc: Dict[str, Any],
    first_result: Any,
    index_result: bool,
    indexed_variant_keys: set[str],
    resolved_override: Optional[str],
    fallback_notes: List[str],
    extraction_strategy: Optional[str],
    session_id: Optional[str],
    effective_run_id: str,
    markdown_options: Optional[Dict[str, Any]],
    content_source: str,
    content_filter: Optional[str],
    max_depth: int,
    follow_links: bool,
    validated_link_filter: Optional[str],
) -> Dict[str, Any]:
    response = _crawl_to_markdown_response_base(
        url=url,
        pages_crawled=pages_crawled,
        pages_indexed=pages_indexed,
        chunks_stored=chunks_stored,
        variant_key=variant_key,
        first_doc=first_doc,
        index_result=index_result,
        indexed_variant_keys=indexed_variant_keys,
        resolved_override=resolved_override,
        fallback_notes=fallback_notes,
        extraction_strategy=extraction_strategy,
        session_id=session_id,
        effective_run_id=effective_run_id,
        markdown_options=markdown_options,
        content_source=content_source,
        content_filter=content_filter,
        max_depth=max_depth,
        follow_links=follow_links,
        validated_link_filter=validated_link_filter,
    )
    first_markdown = first_doc.get("markdown", "")
    response.update(_crawl_markdown_artifacts(first_result, first_markdown, url))
    return _apply_extraction_result(response, first_doc)


def _crawl_to_markdown_response_base(
    *,
    url: str,
    pages_crawled: int,
    pages_indexed: int,
    chunks_stored: int,
    variant_key: str,
    first_doc: Dict[str, Any],
    index_result: bool,
    indexed_variant_keys: set[str],
    resolved_override: Optional[str],
    fallback_notes: List[str],
    extraction_strategy: Optional[str],
    session_id: Optional[str],
    effective_run_id: str,
    markdown_options: Optional[Dict[str, Any]],
    content_source: str,
    content_filter: Optional[str],
    max_depth: int,
    follow_links: bool,
    validated_link_filter: Optional[str],
) -> Dict[str, Any]:
    first_markdown = first_doc.get("markdown", "")
    return {
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
        "content_source_applied": _applied_content_source(content_source),
        "content_filter_applied": _applied_content_filter(content_filter),
        "max_depth_configured": max_depth,
        "follow_links_enabled": follow_links,
        "deep_crawl_mode": _crawl_mode_label(follow_links, max_depth),
        "compatibility_helper_used": bool(follow_links and max_depth > 1),
        "link_filter_applied": bool(validated_link_filter),
    }


def _crawl_markdown_artifacts(first_result: Any, first_markdown: str, url: str) -> Dict[str, Any]:
    return {
        "artifacts": _crawl_artifacts(first_result),
        "link_graph": extract_link_graph(first_markdown, base_url=url),
        "media_metadata": extract_media_metadata(first_markdown),
    }


def _crawl_to_markdown_strategy(
    extraction_strategy: Optional[str],
    extraction_schema: Optional[Dict[str, Any]],
    extraction_patterns: Optional[Dict[str, str]],
    extraction_instruction: Optional[str],
    llm_provider: Optional[str],
) -> Optional[Any]:
    return _build_extraction_strategy(
        strategy_type=extraction_strategy,
        schema=extraction_schema,
        patterns=extraction_patterns,
        instruction=extraction_instruction,
        llm_provider=llm_provider,
    )


def _crawl_to_markdown_config(
    run_config: Optional[Dict[str, Any]],
    session_id: Optional[str],
    markdown_options: Optional[Dict[str, Any]],
    content_source: str,
    content_filter: Optional[str],
    content_filter_query: Optional[str],
    content_filter_threshold: Optional[float],
    content_filter_instruction: Optional[str],
    llm_provider: Optional[str],
    strategy: Optional[Any],
) -> Any:
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
    return config


async def _crawl_to_markdown_indexing(
    crawled_docs: List[Dict[str, Any]],
    *,
    index_result: bool,
    index_variants: Optional[str],
    extraction_strategy: Optional[str],
    session_id: Optional[str],
    max_depth: int,
    follow_links: bool,
    effective_run_id: str,
) -> tuple[int, int, set[str], Optional[str], List[str]]:
    resolved_override = _normalize_index_variants_override(index_variants)
    fallback_notes = _index_override_warning(index_result, index_variants, resolved_override)
    pages_indexed, chunks_stored, indexed_variant_keys, indexing_fallback_notes = await _index_crawled_docs(
        crawled_docs,
        index_result=index_result,
        index_variants=index_variants,
        extraction_strategy=extraction_strategy,
        session_id=session_id,
        max_depth=max_depth,
        follow_links=follow_links,
        effective_run_id=effective_run_id,
        resolved_override=resolved_override,
    )
    fallback_notes.extend(indexing_fallback_notes)
    return pages_indexed, chunks_stored, indexed_variant_keys, resolved_override, fallback_notes


def _crawl_to_markdown_setup(
    ctx: Context,
    url: str,
    run_id: Optional[str],
    run_config: Optional[Dict[str, Any]],
    session_id: Optional[str],
    markdown_options: Optional[Dict[str, Any]],
    content_source: str,
    content_filter: Optional[str],
    content_filter_query: Optional[str],
    content_filter_threshold: Optional[float],
    content_filter_instruction: Optional[str],
    llm_provider: Optional[str],
    extraction_strategy: Optional[str],
    extraction_schema: Optional[Dict[str, Any]],
    extraction_patterns: Optional[Dict[str, str]],
    extraction_instruction: Optional[str],
    markdown_variant: str,
    max_depth: int,
    follow_links: bool,
    link_filter: Optional[str],
) -> tuple[int, Optional[str], Any, str, Optional[Any], Any, str]:
    normalized_max_depth = max(1, min(10, max_depth))
    validated_link_filter = _validate_link_filter(link_filter) if follow_links else None
    crawler = _get_crawler(ctx)
    effective_run_id = _normalize_run_id(run_id) or _generate_run_id("crawl")
    strategy = _crawl_to_markdown_strategy(
        extraction_strategy,
        extraction_schema,
        extraction_patterns,
        extraction_instruction,
        llm_provider,
    )
    config = _crawl_to_markdown_config(
        run_config,
        session_id,
        markdown_options,
        content_source,
        content_filter,
        content_filter_query,
        content_filter_threshold,
        content_filter_instruction,
        llm_provider,
        strategy,
    )
    variant_key = _markdown_variant_key(markdown_variant)
    return normalized_max_depth, validated_link_filter, crawler, effective_run_id, strategy, config, variant_key


async def _crawl_to_markdown_docs(
    crawler: Any,
    url: str,
    config: Any,
    follow_links: bool,
    max_depth: int,
    validated_link_filter: Optional[str],
    variant_key: str,
    strategy: Optional[Any],
) -> tuple[List[Any], int, Optional[str], List[Dict[str, Any]]]:
    results, pages_crawled, crawl_error = await _crawl_results_for_markdown(
        crawler,
        url,
        config,
        follow_links,
        max_depth,
        validated_link_filter,
    )
    return results, pages_crawled, crawl_error, _results_to_crawled_docs(results, variant_key, url, bool(strategy))


def _crawl_to_markdown_empty_payload(url: str, crawl_error: Optional[str]) -> str:
    return json.dumps(
        {"success": False, "url": url, "error": crawl_error or "No content crawled.", "pages_crawled": 0},
        indent=2,
    )


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
        (
            max_depth,
            validated_link_filter,
            crawler,
            effective_run_id,
            strategy,
            config,
            variant_key,
        ) = _crawl_to_markdown_setup(
            ctx,
            url,
            run_id,
            run_config,
            session_id,
            markdown_options,
            content_source,
            content_filter,
            content_filter_query,
            content_filter_threshold,
            content_filter_instruction,
            llm_provider,
            extraction_strategy,
            extraction_schema,
            extraction_patterns,
            extraction_instruction,
            markdown_variant,
            max_depth,
            follow_links,
            link_filter,
        )
        results, pages_crawled, crawl_error, crawled_docs = await _crawl_to_markdown_docs(
            crawler,
            url,
            config,
            follow_links,
            max_depth,
            validated_link_filter,
            variant_key,
            strategy,
        )
        if not results:
            return _crawl_to_markdown_empty_payload(url, crawl_error)

        (
            pages_indexed,
            chunks_stored,
            indexed_variant_keys,
            resolved_override,
            fallback_notes,
        ) = await _crawl_to_markdown_indexing(
            crawled_docs,
            index_result=index_result,
            index_variants=index_variants,
            extraction_strategy=extraction_strategy,
            session_id=session_id,
            max_depth=max_depth,
            follow_links=follow_links,
            effective_run_id=effective_run_id,
        )

        response = _crawl_to_markdown_response(
            url=url,
            pages_crawled=pages_crawled,
            pages_indexed=pages_indexed,
            chunks_stored=chunks_stored,
            variant_key=variant_key,
            first_doc=crawled_docs[0] if crawled_docs else {},
            first_result=results[0] if results else None,
            index_result=index_result,
            indexed_variant_keys=indexed_variant_keys,
            resolved_override=resolved_override,
            fallback_notes=fallback_notes,
            extraction_strategy=extraction_strategy,
            session_id=session_id,
            effective_run_id=effective_run_id,
            markdown_options=markdown_options,
            content_source=content_source,
            content_filter=content_filter,
            max_depth=max_depth,
            follow_links=follow_links,
            validated_link_filter=validated_link_filter,
        )
        return json.dumps(response, indent=2)
    except Exception as exc:
        logger.error(f"crawl_to_markdown {url}: {exc}", exc_info=True)
        return json.dumps({"success": False, "url": url, "error": str(exc), "pages_crawled": 0}, indent=2)


def _crawl_many_config(
    run_config: Optional[Dict[str, Any]],
    session_id: Optional[str],
    markdown_options: Optional[Dict[str, Any]],
    content_source: str,
    content_filter: Optional[str],
    content_filter_query: Optional[str],
    content_filter_threshold: Optional[float],
    content_filter_instruction: Optional[str],
    extraction_strategy: Optional[str],
    extraction_schema: Optional[Dict[str, Any]],
    extraction_patterns: Optional[Dict[str, str]],
    extraction_instruction: Optional[str],
) -> Any:
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
    strategy = _build_extraction_strategy(
        strategy_type=extraction_strategy,
        schema=extraction_schema,
        patterns=extraction_patterns,
        instruction=extraction_instruction,
    )
    if strategy:
        config.extraction_strategy = strategy
    return config


def _crawl_many_response(
    *,
    urls: List[str],
    crawled_docs: List[Dict[str, Any]],
    errors: List[Dict[str, str]],
    index_result: bool,
    pages_processed: int,
    chunks_stored: int,
    variant_key: str,
    extraction_strategy: Optional[str],
    session_id: Optional[str],
    effective_run_id: str,
    markdown_options: Optional[Dict[str, Any]],
    content_source: str,
    content_filter: Optional[str],
    max_depth: int,
    follow_links: bool,
    validated_link_filter: Optional[str],
) -> Dict[str, Any]:
    first_doc = crawled_docs[0] if crawled_docs else {}
    response = {
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
        "content_source_applied": _applied_content_source(content_source),
        "content_filter_applied": _applied_content_filter(content_filter),
        "max_depth_configured": max_depth,
        "follow_links_enabled": follow_links,
        "link_filter_applied": bool(validated_link_filter),
        "compatibility_helper_used": bool(follow_links and max_depth > 1),
        "errors": errors,
    }
    response.update(_crawl_many_artifacts(first_doc))
    return response


def _crawl_many_artifacts(first_doc: Dict[str, Any]) -> Dict[str, Any]:
    if not first_doc:
        return {"link_graph": {}, "media_metadata": {}}
    first_markdown = first_doc.get("markdown", "")
    first_url = first_doc.get("url")
    return {
        "link_graph": extract_link_graph(first_markdown, base_url=first_url),
        "media_metadata": extract_media_metadata(first_markdown),
    }


def _crawl_many_empty_payload() -> str:
    return json.dumps({"success": False, "error": "No URLs provided.", "pages_crawled": 0}, indent=2)


async def _crawl_many_indexing(
    crawled_docs: List[Dict[str, Any]],
    index_result: bool,
    variant_key: str,
) -> tuple[int, int]:
    pages_processed = len(crawled_docs)
    if not (index_result and crawled_docs):
        return pages_processed, 0
    with next(get_session()) as session:
        return await store_crawled_documents(session, crawled_docs, f"batch_{variant_key}")


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
            return _crawl_many_empty_payload()

        max_depth = max(1, min(10, max_depth))
        validated_link_filter = _validate_link_filter(link_filter) if follow_links else None
        crawler = _get_crawler(ctx)
        effective_run_id = _normalize_run_id(run_id) or _generate_run_id("crawl-many")
        config = _crawl_many_config(
            run_config,
            session_id,
            markdown_options,
            content_source,
            content_filter,
            content_filter_query,
            content_filter_threshold,
            content_filter_instruction,
            extraction_strategy,
            extraction_schema,
            extraction_patterns,
            extraction_instruction,
        )
        all_results = await _crawl_many_results(
            crawler,
            urls,
            config,
            follow_links=follow_links,
            max_depth=max_depth,
            validated_link_filter=validated_link_filter,
            max_concurrent=max_concurrent,
        )

        variant_key = _markdown_variant_key(markdown_variant)
        crawled_docs, errors = _batch_docs_and_errors(all_results, variant_key, effective_run_id, extraction_strategy)
        pages_processed, chunks_stored = await _crawl_many_indexing(crawled_docs, index_result, variant_key)

        payload = _crawl_many_response(
            urls=urls,
            crawled_docs=crawled_docs,
            errors=errors,
            index_result=index_result,
            pages_processed=pages_processed,
            chunks_stored=chunks_stored,
            variant_key=variant_key,
            extraction_strategy=extraction_strategy,
            session_id=session_id,
            effective_run_id=effective_run_id,
            markdown_options=markdown_options,
            content_source=content_source,
            content_filter=content_filter,
            max_depth=max_depth,
            follow_links=follow_links,
            validated_link_filter=validated_link_filter,
        )
        return json.dumps(payload, indent=2)
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


def _ingest_patterns(include_patterns: Optional[List[str]]) -> List[str]:
    default_patterns = ["**/*.md", "**/*.markdown", "**/*.html", "**/*.htm"]
    if isinstance(include_patterns, list) and include_patterns:
        return include_patterns
    return default_patterns


def _discover_ingest_files(root: Path, patterns: List[str]) -> List[Path]:
    discovered: List[Path] = []
    for pattern in _valid_ingest_patterns(patterns):
        discovered.extend(root.glob(pattern))
    return sorted({path for path in discovered if path.is_file()})


def _valid_ingest_patterns(patterns: List[str]) -> List[str]:
    return [pattern.strip() for pattern in patterns if isinstance(pattern, str) and pattern.strip()]


async def _ingest_html_file(ctx: Context, path: Path, markdown_variant: str, index_result: bool) -> Dict[str, Any]:
    html = path.read_text(encoding="utf-8", errors="ignore")
    result = await crawl_raw_html(
        ctx=ctx,
        html=html,
        markdown_variant=markdown_variant,
        index_result=index_result,
    )
    return cast(Dict[str, Any], json.loads(result))


async def _ingest_markdown_file(ctx: Context, path: Path, index_result: bool) -> Dict[str, Any]:
    if not index_result:
        return {"success": True, "pages_indexed": 0}
    markdown_text = path.read_text(encoding="utf-8", errors="ignore")
    file_url = f"file://{path}"
    result = await index_markdown(
        ctx=ctx,
        url=file_url,
        markdown=markdown_text,
        metadata={"source_type": "local_file", "ingestion_wrapper": "directory"},
    )
    return cast(Dict[str, Any], json.loads(result))


async def _ingest_file(
    ctx: Context,
    path: Path,
    markdown_variant: str,
    index_result: bool,
) -> Dict[str, Any]:
    if path.suffix.lower() in {".html", ".htm"}:
        return await _ingest_html_file(ctx, path, markdown_variant, index_result)
    return await _ingest_markdown_file(ctx, path, index_result)


def _ingest_success_index_delta(path: Path, data: Dict[str, Any], index_result: bool) -> int:
    if path.suffix.lower() in {".html", ".htm"}:
        return int(data.get("pages_indexed", 0) or 0)
    return 1 if index_result else 0


async def _ingest_files_summary(
    ctx: Context,
    files: List[Path],
    markdown_variant: str,
    index_result: bool,
) -> Dict[str, Any]:
    processed = 0
    indexed_count = 0
    errors: List[Dict[str, str]] = []
    for path in files:
        try:
            data = await _ingest_file(ctx, path, markdown_variant, index_result)
            if not data.get("success"):
                errors.append({"file": str(path), "error": str(data.get("error", "crawl/index failed"))})
                continue
            processed += 1
            indexed_count += _ingest_success_index_delta(path, data, index_result)
        except Exception as file_exc:  # pragma: no cover - defensive branch
            errors.append({"file": str(path), "error": str(file_exc)})
    return {"files_processed": processed, "indexed_count": indexed_count, "errors": errors}


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
            return json.dumps(
                {"success": False, "error": "directory_path must point to an existing directory."},
                indent=2,
            )

        files = _discover_ingest_files(root, _ingest_patterns(include_patterns))
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

        summary = await _ingest_files_summary(ctx, files, markdown_variant, index_result)
        return json.dumps(
            {
                "success": True,
                "directory": str(root),
                "files_discovered": len(files),
                "files_processed": summary["files_processed"],
                "indexed_count": summary["indexed_count"],
                "errors": summary["errors"],
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

        payload: Dict[str, Any] = {
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
                select(CrawledPage).where(CrawledPage.url == url).order_by(cast(Any, CrawledPage.chunk_number))
            ).all()

        if not rows:
            return json.dumps({"success": False, "url": url, "error": "No stored chunks found."}, indent=2)

        selected_rows = _select_rows_for_variant(list(rows), preferred_variant="raw_markdown")
        return _markdown_rows_payload(
            url=url,
            rows=selected_rows,
            content_key="markdown",
            variant="raw_markdown",
            include_provenance=include_provenance,
        )
    except Exception as exc:
        logger.error(f"get_markdown_by_url {url}: {exc}", exc_info=True)
        return json.dumps({"success": False, "url": url, "error": str(exc)}, indent=2)


def _selected_row_variant(rows: List[Any]) -> Optional[str]:
    if not rows:
        return None
    metadata = rows[0].page_metadata if isinstance(rows[0].page_metadata, dict) else {}
    return metadata.get("markdown_variant") if isinstance(metadata, dict) else None


def _markdown_rows_payload(
    *,
    url: str,
    rows: List[Any],
    content_key: str,
    variant: str,
    include_provenance: bool,
) -> str:
    payload = _markdown_rows_payload_base(url, rows, content_key, variant)
    if include_provenance and rows:
        payload["provenance"] = _build_requested_provenance(rows[0].page_metadata)
    return json.dumps(payload, indent=2)


def _markdown_rows_payload_base(url: str, rows: List[Any], content_key: str, variant: str) -> Dict[str, Any]:
    return {
        "success": True,
        "url": url,
        "chunk_count": len(rows),
        content_key: "\n\n".join(r.content for r in rows if r.content),
        "selected_variant": _selected_row_variant(rows) or variant,
    }


def _rows_matching_variant(rows: List[Any], variant: str) -> List[Any]:
    return [
        row
        for row in rows
        if isinstance(row.page_metadata, dict) and row.page_metadata.get("markdown_variant") == variant
    ]


def _removed_mode_payload(url: str, selected_mode: str) -> str:
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


def _invalid_mode_payload(url: str, selected_mode: str) -> str:
    return json.dumps(
        {
            "success": False,
            "url": url,
            "error": "Invalid mode. Use one of: markdown, deep.",
            "mode": selected_mode,
        },
        indent=2,
    )


def _is_markdown_mode(selected_mode: str) -> bool:
    return selected_mode == "markdown"


def _is_removed_mode(selected_mode: str) -> bool:
    return selected_mode in {"smart", "legacy", "single", "single_legacy"}


def _is_deep_mode(selected_mode: str) -> bool:
    return selected_mode == "deep"


async def _crawl_url_markdown(
    ctx: Context,
    url: str,
    markdown_variant: str,
    run_config: Optional[Dict[str, Any]],
    index_result: bool,
    index_variants: Optional[str],
    extraction_strategy: Optional[str],
    extraction_schema: Optional[Dict[str, Any]],
    extraction_patterns: Optional[Dict[str, str]],
    extraction_instruction: Optional[str],
    llm_provider: Optional[str],
    session_id: Optional[str],
    markdown_options: Optional[Dict[str, Any]],
    content_source: str,
    content_filter: Optional[str],
    content_filter_query: Optional[str],
    content_filter_threshold: Optional[float],
    content_filter_instruction: Optional[str],
    max_depth: int,
    follow_links: bool,
    link_filter: Optional[str],
) -> str:
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


async def _crawl_url_deep(
    ctx: Context,
    url: str,
    max_depth: int,
    link_filter: Optional[str],
    url_pattern: Optional[str],
    content_types: Optional[List[str]],
    relevance_query: Optional[str],
    relevance_threshold: Optional[float],
    seo_threshold: Optional[float],
    seo_keywords: Optional[List[str]],
    scorer_type: str,
    markdown_variant: str,
    run_config: Optional[Dict[str, Any]],
    index_result: bool,
) -> str:
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
    if _is_removed_mode(selected_mode):
        return _removed_mode_payload(url, selected_mode)
    if _is_markdown_mode(selected_mode):
        return await _crawl_url_markdown(
            ctx,
            url,
            markdown_variant,
            run_config,
            index_result,
            index_variants,
            extraction_strategy,
            extraction_schema,
            extraction_patterns,
            extraction_instruction,
            llm_provider,
            session_id,
            markdown_options,
            content_source,
            content_filter,
            content_filter_query,
            content_filter_threshold,
            content_filter_instruction,
            max_depth,
            follow_links,
            link_filter,
        )
    if not _is_deep_mode(selected_mode):
        return _invalid_mode_payload(url, selected_mode)
    return await _crawl_url_deep(
        ctx,
        url,
        max_depth,
        link_filter,
        url_pattern,
        content_types,
        relevance_query,
        relevance_threshold,
        seo_threshold,
        seo_keywords,
        scorer_type,
        markdown_variant,
        run_config,
        index_result,
    )


def _session_target_error(normalized_session: Optional[str], action_lower: str) -> Optional[str]:
    if not normalized_session:
        return json.dumps({"success": False, "error": "session_id must be a non-empty string."}, indent=2)
    if action_lower not in {"create", "reuse", "kill"}:
        return _session_action_error_payload()
    return None


def _session_target_kind(url: Optional[str], urls: Optional[List[str]]) -> str:
    if urls:
        return "urls"
    if url:
        return "url"
    return "missing"


async def _crawl_with_session_target(
    ctx: Context,
    target_kind: str,
    url: Optional[str],
    urls: Optional[List[str]],
    markdown_variant: str,
    run_config: Optional[Dict[str, Any]],
    index_result: bool,
    normalized_session: str,
    run_id: Optional[str],
) -> str:
    if target_kind == "urls":
        return await _crawl_with_session_urls(
            ctx,
            cast(List[str], urls),
            markdown_variant,
            run_config,
            index_result,
            normalized_session,
            run_id,
        )
    if target_kind == "url":
        return await _crawl_with_session_url(
            ctx,
            cast(str, url),
            markdown_variant,
            run_config,
            index_result,
            normalized_session,
            run_id,
        )
    return _session_missing_target_payload()


def _validate_deep_strategy(strategy: str, url: str) -> tuple[Optional[str], Optional[str]]:
    strategy_lower = (strategy or "bfs").lower().strip()
    if strategy_lower in _DEEP_CRAWL_STRATEGIES:
        return strategy_lower, None
    return None, json.dumps(
        {
            "success": False,
            "url": url,
            "error": f"Invalid strategy '{strategy}'. Use one of: bfs, dfs, best_first.",
        },
        indent=2,
    )


def _deep_variant_key(markdown_variant: str) -> str:
    variant_map = {
        "raw": "raw_markdown",
        "fit": "fit_markdown",
        "cited": "markdown_with_citations",
        "references": "references_markdown",
    }
    return variant_map.get((markdown_variant or "raw").lower(), "raw_markdown")


def _deep_response_payload(
    *,
    url: str,
    strategy_lower: str,
    max_depth: int,
    max_pages: int,
    crawled_docs: List[Dict[str, Any]],
    pages_indexed: int,
    chunks_stored: int,
    variant_key: str,
    effective_index_result: bool,
    prefetch_only: bool,
    effective_run_id: str,
    stream_mode: bool,
    content_types: Optional[List[str]],
    relevance_query: Optional[str],
    relevance_threshold: Optional[float],
    seo_threshold: Optional[float],
    scorer_type: str,
    errors: List[Dict[str, str]],
) -> Dict[str, Any]:
    clamped_depth, clamped_pages = _clamped_deep_limits(max_depth, max_pages)
    return {
        "success": True,
        "url": url,
        "strategy": strategy_lower,
        "max_depth_configured": clamped_depth,
        "max_pages_configured": clamped_pages,
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
        "scorer_type_applied": _normalized_scorer_type(scorer_type),
        "errors": errors,
        "urls_crawled_sample": _urls_crawled_sample(crawled_docs),
    }


def _deep_empty_payload(url: str, errors: List[Dict[str, str]]) -> str:
    return json.dumps(
        {
            "success": False,
            "url": url,
            "error": "No pages crawled successfully.",
            "errors": errors,
            "pages_crawled": 0,
        },
        indent=2,
    )


def _deep_validated_strategy(strategy: str, url: str) -> tuple[Optional[str], Optional[str]]:
    strategy_lower, strategy_error = _validate_deep_strategy(strategy, url)
    if strategy_error is not None:
        return None, strategy_error
    return strategy_lower or "bfs", None


async def _deep_index_docs(
    crawled_docs: List[Dict[str, Any]],
    effective_index_result: bool,
    strategy_lower: str,
    variant_key: str,
) -> tuple[int, int]:
    if not effective_index_result:
        return 0, 0
    with next(get_session()) as session:
        return await store_crawled_documents(session, crawled_docs, f"deep_{strategy_lower}_{variant_key}")


async def _deep_crawl_docs(
    crawler: Any,
    url: str,
    config: Any,
    variant_key: str,
    effective_run_id: str,
) -> tuple[bool, List[Dict[str, Any]], List[Dict[str, str]]]:
    container = await crawler.arun(url=url, config=config)
    stream_mode = bool(getattr(config, "stream", False))
    container_results = await _container_to_results(container, stream_mode)
    crawled_docs, errors = _batch_docs_and_errors(
        container_results,
        variant_key,
        effective_run_id,
        extraction_strategy=None,
    )
    return stream_mode, crawled_docs, errors


async def _deep_runtime_result(
    crawler: Any,
    url: str,
    config: Any,
    variant_key: str,
    effective_run_id: str,
) -> tuple[Optional[str], Optional[bool], Optional[List[Dict[str, Any]]], Optional[List[Dict[str, str]]]]:
    stream_mode, crawled_docs, errors = await _deep_crawl_docs(crawler, url, config, variant_key, effective_run_id)
    if not crawled_docs:
        return _deep_empty_payload(url, errors), None, None, None
    return None, stream_mode, crawled_docs, errors


async def _crawl_deep_payload(
    ctx: Context,
    url: str,
    strategy_lower: str,
    max_depth: int,
    max_pages: int,
    include_external: bool,
    score_threshold: float,
    url_pattern: Optional[str],
    allowed_domains: Optional[List[str]],
    keywords: Optional[List[str]],
    content_types: Optional[List[str]],
    relevance_query: Optional[str],
    relevance_threshold: Optional[float],
    seo_threshold: Optional[float],
    seo_keywords: Optional[List[str]],
    scorer_type: str,
    markdown_variant: str,
    run_config: Optional[Dict[str, Any]],
    index_result: bool,
    prefetch_only: bool,
    run_id: Optional[str],
) -> str:
    crawler = _get_crawler(ctx)
    effective_run_id = _normalize_run_id(run_id) or _generate_run_id("crawl-deep")
    config = _deep_run_config(
        run_config=run_config,
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
    variant_key = _deep_variant_key(markdown_variant)
    runtime_error, stream_mode, crawled_docs, errors = await _deep_runtime_result(
        crawler,
        url,
        config,
        variant_key,
        effective_run_id,
    )
    if runtime_error is not None:
        return runtime_error
    effective_index_result = bool(index_result and not prefetch_only)
    pages_indexed, chunks_stored = await _deep_index_docs(
        cast(List[Dict[str, Any]], crawled_docs),
        effective_index_result,
        strategy_lower,
        variant_key,
    )
    payload = _deep_response_payload(
        url=url,
        strategy_lower=strategy_lower,
        max_depth=max_depth,
        max_pages=max_pages,
        crawled_docs=cast(List[Dict[str, Any]], crawled_docs),
        pages_indexed=pages_indexed,
        chunks_stored=chunks_stored,
        variant_key=variant_key,
        effective_index_result=effective_index_result,
        prefetch_only=prefetch_only,
        effective_run_id=effective_run_id,
        stream_mode=cast(bool, stream_mode),
        content_types=content_types,
        relevance_query=relevance_query,
        relevance_threshold=relevance_threshold,
        seo_threshold=seo_threshold,
        scorer_type=scorer_type,
        errors=cast(List[Dict[str, str]], errors),
    )
    return json.dumps(payload, indent=2)


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
    """Deep-crawl a site using Crawl4AI native BFS/DFS/Best-first strategies."""
    strategy_lower, validation_error = _deep_validated_strategy(strategy, url)
    if validation_error is not None:
        return validation_error

    try:
        return await _crawl_deep_payload(
            ctx,
            url,
            cast(str, strategy_lower),
            max_depth,
            max_pages,
            include_external,
            score_threshold,
            url_pattern,
            allowed_domains,
            keywords,
            content_types,
            relevance_query,
            relevance_threshold,
            seo_threshold,
            seo_keywords,
            scorer_type,
            markdown_variant,
            run_config,
            index_result,
            prefetch_only,
            run_id,
        )

    except Exception as exc:
        logger.error(f"crawl_deep {url}: {exc}", exc_info=True)
        return json.dumps({"success": False, "url": url, "error": str(exc), "pages_crawled": 0}, indent=2)


def _deep_run_config(
    *,
    run_config: Optional[Dict[str, Any]],
    strategy: str,
    max_depth: int,
    max_pages: int,
    include_external: bool,
    score_threshold: float,
    url_pattern: Optional[str],
    allowed_domains: Optional[List[str]],
    keywords: Optional[List[str]],
    content_types: Optional[List[str]],
    relevance_query: Optional[str],
    relevance_threshold: Optional[float],
    seo_threshold: Optional[float],
    seo_keywords: Optional[List[str]],
    scorer_type: str,
) -> Any:
    config = _build_run_config(run_config)
    config.deep_crawl_strategy = _build_deep_crawl_strategy(
        strategy=strategy,
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
    return config


def _validate_adaptive_strategy(strategy: str, url: str) -> tuple[Optional[str], Optional[str]]:
    strategy_lower = (strategy or "statistical").lower().strip()
    if strategy_lower in {"statistical", "embedding"}:
        return strategy_lower, None
    return None, json.dumps(
        {
            "success": False,
            "url": url,
            "error": f"Invalid strategy '{strategy}'. Use one of: statistical, embedding.",
        },
        indent=2,
    )


def _validate_adaptive_query(query: str, url: str) -> Optional[str]:
    if isinstance(query, str) and query.strip():
        return None
    return json.dumps(
        {
            "success": False,
            "url": url,
            "error": "query must be a non-empty string.",
        },
        indent=2,
    )


def _adaptive_confidence_threshold(confidence_threshold: float) -> float:
    return max(0.0, min(1.0, confidence_threshold))


def _adaptive_config(
    confidence_threshold: float,
    max_depth: int,
    max_pages: int,
    top_k_links: int,
    min_gain_threshold: float,
    strategy_name: str,
) -> AdaptiveConfig:
    return AdaptiveConfig(
        confidence_threshold=confidence_threshold,
        max_depth=max(1, min(10, max_depth)),
        max_pages=max(1, min(200, max_pages)),
        top_k_links=max(1, min(20, top_k_links)),
        min_gain_threshold=max(0.0, min(1.0, min_gain_threshold)),
        strategy=strategy_name,
    )


def _adaptive_empty_payload(url: str, query: str, errors: List[Dict[str, str]]) -> str:
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


def _adaptive_validated_strategy(strategy: str, query: str, url: str) -> tuple[Optional[str], Optional[str]]:
    strategy_lower, strategy_error = _validate_adaptive_strategy(strategy, url)
    if strategy_error is not None:
        return None, strategy_error
    query_error = _validate_adaptive_query(query, url)
    if query_error is not None:
        return None, query_error
    return strategy_lower or "statistical", None


def _adaptive_pseudo_rows(
    crawled_docs: List[Dict[str, Any]],
    bounded_count: int,
    strategy_lower: str,
    variant_key: str,
) -> List[Dict[str, Any]]:
    return [
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
        for doc in crawled_docs[:bounded_count]
    ]


def _urls_crawled_sample(crawled_docs: List[Dict[str, Any]]) -> List[str]:
    sample = [str(doc.get("url") or "") for doc in crawled_docs[:5] if doc.get("url")]
    return sample + (["..."] if len(crawled_docs) > 5 else [])


def _with_optional_payload_fields(
    payload: Dict[str, Any],
    kb_export: Any,
    adaptive_answer: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if kb_export is not None:
        payload["knowledge_base_export"] = kb_export
    if adaptive_answer is not None:
        payload["adaptive_answer"] = adaptive_answer
    return payload


def _adaptive_kb_export(
    export_knowledge_base: bool,
    crawled_docs: List[Dict[str, Any]],
    knowledge_base_format: str,
) -> Any:
    if not export_knowledge_base:
        return None
    return _build_adaptive_knowledge_base_export(crawled_docs, format_name=knowledge_base_format)


def _adaptive_relevant_content(adaptive: Any, top_k_relevant: int) -> Any:
    return adaptive.get_relevant_content(top_k=max(1, min(20, top_k_relevant)))


def _adaptive_runtime(
    ctx: Context,
    run_id: Optional[str],
    confidence_threshold: float,
    max_depth: int,
    max_pages: int,
    top_k_links: int,
    min_gain_threshold: float,
    strategy_name: str,
) -> tuple[str, float, AdaptiveCrawler]:
    effective_run_id = _normalize_run_id(run_id) or _generate_run_id("crawl-adaptive")
    threshold = _adaptive_confidence_threshold(confidence_threshold)
    adaptive = _adaptive_crawler(
        ctx,
        threshold,
        max_depth,
        max_pages,
        top_k_links,
        min_gain_threshold,
        strategy_name,
    )
    return effective_run_id, threshold, adaptive


async def _adaptive_response_data(
    adaptive: Any,
    index_result: bool,
    strategy_name: str,
    variant_key: str,
    top_k_relevant: int,
    answer_query: Optional[str],
    pages_indexed: int,
    answer_match_count: int,
    crawled_docs: List[Dict[str, Any]],
    export_knowledge_base: bool,
    knowledge_base_format: str,
) -> tuple[Any, Optional[Dict[str, Any]], Any]:
    relevant = _adaptive_relevant_content(adaptive, top_k_relevant)
    adaptive_answer = await _adaptive_answer_payload(
        answer_query,
        index_result,
        pages_indexed,
        answer_match_count,
        strategy_name,
        variant_key,
        crawled_docs,
    )
    kb_export = _adaptive_kb_export(export_knowledge_base, crawled_docs, knowledge_base_format)
    return relevant, adaptive_answer, kb_export


async def _adaptive_digest_docs(
    adaptive: Any,
    url: str,
    query: str,
    variant_key: str,
    effective_run_id: str,
) -> tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    state = await adaptive.digest(start_url=url, query=query.strip())
    knowledge_base = list(getattr(state, "knowledge_base", []) or [])
    return _batch_docs_and_errors(
        knowledge_base,
        variant_key,
        effective_run_id,
        extraction_strategy=None,
    )


async def _adaptive_crawled_docs_or_error(
    adaptive: Any,
    url: str,
    query: str,
    variant_key: str,
    effective_run_id: str,
) -> tuple[Optional[str], Optional[List[Dict[str, Any]]], Optional[List[Dict[str, str]]]]:
    crawled_docs, errors = await _adaptive_digest_docs(adaptive, url, query, variant_key, effective_run_id)
    if not crawled_docs:
        return _adaptive_empty_payload(url, query, errors), None, None
    return None, crawled_docs, errors


async def _adaptive_index_docs(
    crawled_docs: List[Dict[str, Any]],
    index_result: bool,
    strategy_lower: str,
    variant_key: str,
) -> tuple[int, int]:
    if not (index_result and crawled_docs):
        return 0, 0
    with next(get_session()) as session:
        return await store_crawled_documents(session, crawled_docs, f"adaptive_{strategy_lower}_{variant_key}")


async def _adaptive_answer_payload(
    answer_query: Optional[str],
    index_result: bool,
    pages_indexed: int,
    answer_match_count: int,
    strategy_lower: str,
    variant_key: str,
    crawled_docs: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not (isinstance(answer_query, str) and answer_query.strip()):
        return None
    normalized_query = answer_query.strip()
    bounded_count = max(1, min(10, answer_match_count))
    if index_result and pages_indexed > 0:
        with next(get_session()) as session:
            scoped_rows = await search_documents(
                session,
                normalized_query,
                match_count=bounded_count,
                filter_metadata={"crawl_type": f"adaptive_{strategy_lower}_{variant_key}"},
            )
        return _build_adaptive_answer(normalized_query, scoped_rows)
    pseudo_rows = _adaptive_pseudo_rows(crawled_docs, bounded_count, strategy_lower, variant_key)
    return _build_adaptive_answer(normalized_query, pseudo_rows)


def _adaptive_response_payload(
    *,
    url: str,
    query: str,
    strategy_lower: str,
    crawled_docs: List[Dict[str, Any]],
    index_result: bool,
    pages_indexed: int,
    chunks_stored: int,
    variant_key: str,
    confidence_threshold: float,
    adaptive: Any,
    relevant: Any,
    errors: List[Dict[str, str]],
    effective_run_id: str,
    kb_export: Any,
    adaptive_answer: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
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
        "confidence_threshold": confidence_threshold,
        "stopped_when_confident": bool(adaptive.confidence >= confidence_threshold),
        "coverage_stats": adaptive.coverage_stats,
        "relevant_content": relevant,
        "errors": errors,
        "run_id": effective_run_id,
        "urls_crawled_sample": _urls_crawled_sample(crawled_docs),
    }
    return _with_optional_payload_fields(payload, kb_export, adaptive_answer)


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
    """Adaptive, query-guided crawl using Crawl4AI AdaptiveCrawler."""
    strategy_name, validation_error = _adaptive_validated_strategy(strategy, query, url)
    if validation_error is not None:
        return validation_error

    try:
        effective_run_id, threshold, adaptive = _adaptive_runtime(
            ctx,
            run_id,
            confidence_threshold,
            max_depth,
            max_pages,
            top_k_links,
            min_gain_threshold,
            cast(str, strategy_name),
        )
        variant_key = _markdown_variant_key(markdown_variant or "raw")
        crawl_error, crawled_docs, errors = await _adaptive_crawled_docs_or_error(
            adaptive,
            url,
            query,
            variant_key,
            effective_run_id,
        )
        if crawl_error is not None:
            return crawl_error

        pages_indexed, chunks_stored = await _adaptive_index_docs(
            cast(List[Dict[str, Any]], crawled_docs),
            index_result,
            cast(str, strategy_name),
            variant_key,
        )
        relevant, adaptive_answer, kb_export = await _adaptive_response_data(
            adaptive,
            index_result,
            cast(str, strategy_name),
            variant_key,
            top_k_relevant,
            answer_query,
            pages_indexed,
            answer_match_count,
            cast(List[Dict[str, Any]], crawled_docs),
            export_knowledge_base,
            knowledge_base_format,
        )
        payload = _adaptive_response_payload(
            url=url,
            query=query,
            strategy_lower=cast(str, strategy_name),
            crawled_docs=cast(List[Dict[str, Any]], crawled_docs),
            index_result=index_result,
            pages_indexed=pages_indexed,
            chunks_stored=chunks_stored,
            variant_key=variant_key,
            confidence_threshold=threshold,
            adaptive=adaptive,
            relevant=relevant,
            errors=cast(List[Dict[str, str]], errors),
            effective_run_id=effective_run_id,
            kb_export=kb_export,
            adaptive_answer=adaptive_answer,
        )
        return json.dumps(payload, indent=2)

    except Exception as exc:
        logger.error(f"crawl_adaptive {url}: {exc}", exc_info=True)
        return json.dumps(
            {"success": False, "url": url, "query": query, "error": str(exc), "pages_crawled": 0},
            indent=2,
        )


def _adaptive_crawler(
    ctx: Context,
    threshold: float,
    max_depth: int,
    max_pages: int,
    top_k_links: int,
    min_gain_threshold: float,
    strategy_name: str,
) -> AdaptiveCrawler:
    crawler = _get_crawler(ctx)
    adaptive_config = _adaptive_config(
        threshold,
        max_depth,
        max_pages,
        top_k_links,
        min_gain_threshold,
        strategy_name,
    )
    return AdaptiveCrawler(crawler=crawler, config=adaptive_config)


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
    action_lower = (action or "reuse").lower().strip()
    validation_error = _session_target_error(normalized_session, action_lower)
    if validation_error is not None:
        return validation_error

    if action_lower == "kill":
        return _session_kill_payload(cast(str, normalized_session))

    target_kind = _session_target_kind(url, urls)
    return await _crawl_with_session_target(
        ctx,
        target_kind,
        url,
        urls,
        markdown_variant,
        run_config,
        index_result,
        cast(str, normalized_session),
        run_id,
    )


async def _crawl_with_session_urls(
    ctx: Context,
    urls: List[str],
    markdown_variant: str,
    run_config: Optional[Dict[str, Any]],
    index_result: bool,
    normalized_session: str,
    run_id: Optional[str],
) -> str:
    return await crawl_many_urls(
        ctx=ctx,
        urls=urls,
        markdown_variant=markdown_variant,
        run_config=run_config,
        index_result=index_result,
        session_id=normalized_session,
        run_id=run_id,
    )


async def _crawl_with_session_url(
    ctx: Context,
    url: str,
    markdown_variant: str,
    run_config: Optional[Dict[str, Any]],
    index_result: bool,
    normalized_session: str,
    run_id: Optional[str],
) -> str:
    return await crawl_to_markdown(
        ctx=ctx,
        url=url,
        markdown_variant=markdown_variant,
        run_config=run_config,
        index_result=index_result,
        session_id=normalized_session,
        run_id=run_id,
    )


def _safe_auth_headers(custom_headers: Optional[Dict[str, str]]) -> Dict[str, str]:
    return {str(k): str(v) for k, v in (custom_headers or {}).items() if isinstance(k, str) and isinstance(v, str)}


def _safe_auth_cookies(cookies: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    return cookies if isinstance(cookies, list) else []


def _hook_script_parts(
    pre_navigation_js: Optional[str],
    local_storage: Optional[Dict[str, str]],
    final_scroll: bool,
    post_navigation_js: Optional[str],
) -> List[str]:
    hook_script_parts: List[str] = []
    _append_hook_script(hook_script_parts, pre_navigation_js)
    hook_script_parts.extend(_local_storage_hook_scripts(local_storage))
    if final_scroll:
        hook_script_parts.append("window.scrollTo(0, document.body.scrollHeight);")
    _append_hook_script(hook_script_parts, post_navigation_js)
    return hook_script_parts


def _append_hook_script(parts: List[str], script: Optional[str]) -> None:
    if isinstance(script, str) and script.strip():
        parts.append(script.strip())


def _local_storage_hook_scripts(local_storage: Optional[Dict[str, str]]) -> List[str]:
    if not isinstance(local_storage, dict) or not local_storage:
        return []
    return [
        f"window.localStorage.setItem({json.dumps(str(key))}, {json.dumps(str(value))});"
        for key, value in local_storage.items()
    ]


def _session_action_error_payload() -> str:
    return json.dumps({"success": False, "error": "action must be one of: create, reuse, kill."}, indent=2)


def _session_kill_payload(session_id: str) -> str:
    return json.dumps(
        {
            "success": True,
            "action": "kill",
            "session_id": session_id,
            "message": "Session marked for termination on next crawl lifecycle boundary.",
        },
        indent=2,
    )


def _session_missing_target_payload() -> str:
    return json.dumps({"success": False, "error": "Provide url or urls."}, indent=2)


def _normalized_route_block_patterns(route_block_patterns: Optional[List[str]]) -> List[str]:
    return [pattern for pattern in (route_block_patterns or []) if isinstance(pattern, str) and pattern.strip()]


def _auth_run_config(
    run_config: Optional[Dict[str, Any]],
    hook_script_parts: List[str],
    route_block_patterns: Optional[List[str]],
) -> Dict[str, Any]:
    merged_run_config = dict(run_config or {})
    if hook_script_parts:
        merged_run_config["js_code_before_wait"] = "\n".join(hook_script_parts)
    normalized_patterns = _normalized_route_block_patterns(route_block_patterns)
    if normalized_patterns:
        merged_run_config["c4a_script"] = {"route_block_patterns": normalized_patterns}
    return merged_run_config


def _auth_browser_override(
    safe_headers: Dict[str, str],
    safe_cookies: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    browser_override: Dict[str, Any] = {}
    if safe_headers:
        browser_override["headers"] = safe_headers
    if safe_cookies:
        browser_override["cookies"] = safe_cookies
    return browser_override or None


def _auth_hooks_payload(
    payload: Dict[str, Any],
    normalized_session: str,
    safe_headers: Dict[str, str],
    safe_cookies: List[Dict[str, Any]],
    local_storage: Optional[Dict[str, str]],
    route_block_patterns: Optional[List[str]],
    final_scroll: bool,
    pre_navigation_js: Optional[str],
    post_navigation_js: Optional[str],
) -> Dict[str, Any]:
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
    return payload


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
    """Auth-capable crawl wrapper with safe hook-like controls."""
    normalized_session = _normalize_session_id(session_id)
    if not normalized_session:
        return json.dumps({"success": False, "error": "session_id must be a non-empty string."}, indent=2)

    safe_headers = _safe_auth_headers(custom_headers)
    safe_cookies = _safe_auth_cookies(cookies)
    hooks = _hook_script_parts(pre_navigation_js, local_storage, final_scroll, post_navigation_js)
    merged_run_config = _auth_run_config(run_config, hooks, route_block_patterns)
    crawl_result = await crawl_with_browser_config(
        ctx=ctx,
        url=url,
        browser_config=_auth_browser_override(safe_headers, safe_cookies),
        markdown_variant=markdown_variant,
        index_result=index_result,
        run_config=_merge_run_config_with_session(merged_run_config, normalized_session),
    )
    payload = _auth_hooks_payload(
        json.loads(crawl_result),
        normalized_session,
        safe_headers,
        safe_cookies,
        local_storage,
        route_block_patterns,
        final_scroll,
        pre_navigation_js,
        post_navigation_js,
    )
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
    urls = _paginated_urls(start_url, additional_urls)

    result = await crawl_many_urls(
        ctx=ctx,
        urls=urls,
        max_concurrent=max(1, min(20, max_concurrent)),
        markdown_variant=markdown_variant,
        index_result=index_result,
        session_id=normalized_session,
    )
    return _paginated_payload(result, start_url, normalized_session)


def _paginated_urls(start_url: str, additional_urls: Optional[List[str]]) -> List[str]:
    urls = [start_url]
    for candidate in additional_urls or []:
        _append_paginated_url(urls, candidate)
    return urls


def _append_paginated_url(urls: List[str], candidate: Any) -> None:
    if not isinstance(candidate, str):
        return
    normalized = candidate.strip()
    if normalized and normalized not in urls:
        urls.append(normalized)


def _structured_failure_payload(error: str) -> str:
    return json.dumps({"success": False, "error": error}, indent=2)


def _structured_normalized_payload(
    payload: Dict[str, Any],
    strategy: str,
    source_type: str,
    target_url: str,
    extraction_schema: Optional[Dict[str, Any]],
    fit_source: bool,
) -> str:
    payload["normalized_output"] = _normalize_structured_contract(
        strategy=strategy,
        extracted_content=payload.get("extraction_result"),
        source_type=source_type,
        source_value=target_url,
        schema=extraction_schema,
        fit_source_used=bool(fit_source),
    )
    return json.dumps(payload, indent=2)


def _paginated_payload(result: str, start_url: str, normalized_session: str) -> str:
    payload = json.loads(result)
    payload["workflow_mode"] = "paginated_preset"
    payload["start_url"] = start_url
    payload["session_id_applied"] = normalized_session
    return json.dumps(payload, indent=2)


def _browser_variant_key(markdown_variant: str) -> str:
    variant_map = {
        "raw": "raw_markdown",
        "fit": "fit_markdown",
        "cited": "markdown_with_citations",
        "references": "references_markdown",
    }
    return variant_map.get(markdown_variant.lower(), "raw_markdown")


async def _browser_variant_index_count(
    ctx: Context,
    url: str,
    selected: str,
    variant_key: str,
    index_result: bool,
) -> int:
    if not (index_result and selected):
        return 0
    indexed = await index_markdown(
        ctx=ctx,
        url=url,
        markdown=selected,
        metadata={"markdown_variant": variant_key, "browser_override": True},
    )
    return int(json.loads(indexed).get("chunks_stored", 0) or 0)


def _browser_failure_payload(url: str, result: Any) -> str:
    return json.dumps(
        {
            "success": False,
            "url": url,
            "error": getattr(result, "error_message", None) or "No content.",
        },
        indent=2,
    )


def _browser_success_payload(
    *,
    url: str,
    variant_key: str,
    selected: str,
    index_result: bool,
    chunks_stored: int,
    browser_config: Optional[Dict[str, Any]],
) -> str:
    return json.dumps(
        {
            "success": True,
            "url": url,
            "selected_variant": variant_key,
            "selected_markdown": selected,
            "index_result": index_result,
            "chunks_stored": chunks_stored,
            "browser_config_applied": _applied_browser_config(browser_config),
        },
        indent=2,
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
            result = await crawler.arun(url=url, config=_build_run_config(run_config))
            if not (result.success and result.markdown):
                return _browser_failure_payload(url, result)

            variants = _extract_markdown_variants(result.markdown)
            variant_key = _browser_variant_key(markdown_variant)
            selected = variants.get(variant_key) or variants["raw_markdown"]
            chunks_stored = await _browser_variant_index_count(ctx, url, selected, variant_key, index_result)
            return _browser_success_payload(
                url=url,
                variant_key=variant_key,
                selected=selected,
                index_result=index_result,
                chunks_stored=chunks_stored,
                browser_config=browser_config,
            )
    except Exception as exc:
        logger.error(f"crawl_with_browser_config {url}: {exc}", exc_info=True)
        return json.dumps({"success": False, "url": url, "error": str(exc)}, indent=2)


def _applied_browser_config(browser_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return {k: v for k, v in (browser_config or {}).items() if k in _ALLOWED_BROWSER_CONFIG_FIELDS}


async def inspect_session(
    ctx: Context,
    session_id: str,
) -> str:
    """Inspect a session identifier and report normalized status."""
    normalized = _normalize_session_id(session_id)
    if not normalized:
        return json.dumps({"success": False, "error": "session_id must be a non-empty string."}, indent=2)
    return json.dumps(
        {
            "success": True,
            "session_id": normalized,
            "active": True,
            "message": "Session is available for reuse in crawl tools that accept session_id.",
        },
        indent=2,
    )


async def create_session(
    ctx: Context,
    session_id: str,
) -> str:
    """Create/register a reusable crawl session identifier."""
    normalized = _normalize_session_id(session_id)
    if not normalized:
        return json.dumps({"success": False, "error": "session_id must be a non-empty string."}, indent=2)
    return json.dumps(
        {
            "success": True,
            "session_id": normalized,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "message": "Session is ready to use in crawl tools that accept session_id.",
        },
        indent=2,
    )


async def kill_session(
    ctx: Context,
    session_id: str,
) -> str:
    """Terminate/deactivate a reusable crawl session identifier."""
    normalized = _normalize_session_id(session_id)
    if not normalized:
        return json.dumps({"success": False, "error": "session_id must be a non-empty string."}, indent=2)
    return json.dumps(
        {
            "success": True,
            "session_id": normalized,
            "terminated_at": datetime.now(timezone.utc).isoformat(),
            "message": "Session marked as terminated.",
        },
        indent=2,
    )


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


def _index_override_warning(
    index_result: bool,
    index_variants: Optional[str],
    resolved_override: Optional[str],
) -> List[str]:
    if index_result and isinstance(index_variants, str) and resolved_override is None:
        return ["invalid index_variants override; using global markdown index policy"]
    return []


async def _append_variant_chunks(
    *,
    variants: Dict[str, str],
    variant_key: str,
    url: str,
    effective_policy: str,
    resolved_override: Optional[str],
    payload: Dict[str, Any],
) -> None:
    markdown_value = variants.get(variant_key) or ""
    if not markdown_value:
        return
    payload["indexed_variants"].append(variant_key)
    chunks = await chunk_text_according_to_settings(markdown_value)
    reference_meta = _build_reference_metadata(variants)
    now_iso = datetime.now(timezone.utc).isoformat()
    for chunk_index, chunk in enumerate(chunks):
        meta = extract_section_info(chunk)
        meta.update(
            {
                "chunk_index": chunk_index,
                "url": url,
                "source": urlparse(url).netloc,
                "source_type": _infer_source_type(url),
                "crawl_time": now_iso,
                "crawl_timestamp": now_iso,
                "markdown_variant": variant_key,
                "content_class": "text",
                "is_active": True,
                "content_hash": hashlib.sha256(chunk.encode("utf-8")).hexdigest(),
                "markdown_index_policy": effective_policy,
                "index_variants_override": resolved_override,
            }
        )
        meta.update(reference_meta)
        payload["db_urls"].append(url)
        payload["db_chunks"].append(chunk_index)
        payload["db_contents"].append(chunk)
        payload["db_metas"].append(meta)
        payload["db_fulldocs"].append(markdown_value)


async def _index_markdown_variants_payload(
    *,
    url: str,
    variants: Dict[str, str],
    index_variants: Optional[str],
    resolved_override: Optional[str],
) -> tuple[int, List[str], List[str]]:
    variant_keys_to_index, effective_policy, _, resolved_fallback_notes = _resolve_variants_to_index(
        variants=variants,
        index_variants=index_variants,
        fallback_enabled=bool(settings.MARKDOWN_FALLBACK_ENABLED),
    )
    payload: Dict[str, Any] = {
        "db_urls": [],
        "db_chunks": [],
        "db_contents": [],
        "db_metas": [],
        "db_fulldocs": [],
        "indexed_variants": [],
    }
    for variant_key in variant_keys_to_index:
        await _append_variant_chunks(
            variants=variants,
            variant_key=variant_key,
            url=url,
            effective_policy=effective_policy,
            resolved_override=resolved_override,
            payload=payload,
        )
    if payload["db_urls"]:
        with next(get_session()) as session:
            chunks_stored = await add_documents_to_db(
                session,
                payload["db_urls"],
                payload["db_contents"],
                payload["db_metas"],
                payload["db_chunks"],
                payload["db_fulldocs"],
            )
    else:
        chunks_stored = 0
    return chunks_stored, cast(List[str], payload["indexed_variants"]), resolved_fallback_notes


def _markdown_variant_failure(url: str, result: Any) -> str:
    return json.dumps(
        {
            "success": False,
            "url": url,
            "error": getattr(result, "error_message", None) or "No content.",
        },
        indent=2,
    )


async def _indexed_variant_result(
    *,
    url: str,
    variants: Dict[str, str],
    index_result: bool,
    index_variants: Optional[str],
    resolved_override: Optional[str],
    fallback_notes: List[str],
) -> tuple[int, List[str], List[str]]:
    if not index_result:
        return 0, [], fallback_notes
    chunks_stored, indexed_variants, resolved_fallback_notes = await _index_markdown_variants_payload(
        url=url,
        variants=variants,
        index_variants=index_variants,
        resolved_override=resolved_override,
    )
    fallback_notes.extend(resolved_fallback_notes)
    return chunks_stored, indexed_variants, fallback_notes


def _markdown_variants_payload(
    *,
    url: str,
    variants: Dict[str, str],
    index_result: bool,
    chunks_stored: int,
    indexed_variants: List[str],
    resolved_override: Optional[str],
    fallback_notes: List[str],
) -> str:
    return json.dumps(
        {
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
        },
        indent=2,
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
        result = await crawler.arun(url=url, config=_build_run_config(run_config))
        if not (result.success and result.markdown):
            return _markdown_variant_failure(url, result)

        variants = _extract_markdown_variants(result.markdown)
        resolved_override = _normalize_index_variants_override(index_variants)
        fallback_notes = _index_override_warning(index_result, index_variants, resolved_override)
        chunks_stored, indexed_variants, fallback_notes = await _indexed_variant_result(
            url=url,
            variants=variants,
            index_result=index_result,
            index_variants=index_variants,
            resolved_override=resolved_override,
            fallback_notes=fallback_notes,
        )
        return _markdown_variants_payload(
            url=url,
            variants=variants,
            index_result=index_result,
            chunks_stored=chunks_stored,
            indexed_variants=indexed_variants,
            resolved_override=resolved_override,
            fallback_notes=fallback_notes,
        )
    except Exception as exc:
        logger.error(f"extract_markdown_variants {url}: {exc}", exc_info=True)
        return json.dumps({"success": False, "url": url, "error": str(exc)}, indent=2)


def _extract_regex_from_fit_markdown(
    fit_markdown: str,
    extraction_schema: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    regex_patterns = extraction_schema if isinstance(extraction_schema, dict) else {}
    extracted_regex: Dict[str, Any] = {}
    for key, pattern in regex_patterns.items():
        try:
            extracted_regex[key] = re.findall(str(pattern), fit_markdown)
        except re.error:
            extracted_regex[key] = []
    return extracted_regex


def _fit_markdown_structured_response(
    fit_markdown: str,
    strategy: str,
    extraction_schema: Optional[Dict[str, Any]],
) -> str:
    if strategy != "regex":
        return json.dumps(
            {
                "success": False,
                "error": "fit_markdown extraction is currently supported only for regex strategy.",
            },
            indent=2,
        )
    regex_patterns = extraction_schema if isinstance(extraction_schema, dict) else {}
    extracted_regex = _extract_regex_from_fit_markdown(fit_markdown, extraction_schema)
    normalized = _normalize_structured_contract(
        strategy="regex",
        extracted_content=extracted_regex,
        source_type="fit_markdown",
        source_value="inline",
        schema=regex_patterns,
        fit_source_used=True,
    )
    return json.dumps({"success": True, "normalized_output": normalized}, indent=2)


def _structured_source_target(
    url: Optional[str],
    file_path: Optional[str],
    html: Optional[str],
) -> tuple[Optional[str], str, Optional[str]]:
    html_target = _html_source_target(html)
    if html_target is not None:
        return html_target
    file_target = _file_source_target(file_path)
    if file_target is not None:
        return file_target
    if url:
        return url, "url", None
    return None, "url", "Provide one of: url, file_path, html."


def _html_source_target(html: Optional[str]) -> Optional[tuple[Optional[str], str, Optional[str]]]:
    if html is None:
        return None
    if not html.strip():
        return None, "raw_html", "html must be non-empty when provided."
    return f"raw:{html}", "raw_html", None


def _file_source_target(file_path: Optional[str]) -> Optional[tuple[Optional[str], str, Optional[str]]]:
    if file_path is None:
        return None
    normalized_file_path = file_path.replace("file://", "", 1) if file_path.startswith("file://") else file_path
    return f"file://{Path(normalized_file_path).resolve()}", "file", None


def _structured_fit_response(
    fit_markdown: Optional[str],
    strategy: str,
    extraction_schema: Optional[Dict[str, Any]],
) -> Optional[str]:
    if fit_markdown is None:
        return None
    return _fit_markdown_structured_response(fit_markdown, strategy, extraction_schema)


def _structured_target_result(
    url: Optional[str],
    file_path: Optional[str],
    html: Optional[str],
) -> tuple[Optional[tuple[Optional[str], str]], Optional[str]]:
    target_url, source_type, source_error = _structured_source_target(url, file_path, html)
    if source_error is not None:
        return None, _structured_failure_payload(source_error)
    return (target_url, source_type), None


def _structured_response_payload(
    extracted_response: str,
    strategy: str,
    source_type: str,
    target_url: str,
    extraction_schema: Optional[Dict[str, Any]],
    fit_source: bool,
) -> str:
    payload = json.loads(extracted_response)
    if not payload.get("success"):
        return json.dumps(payload, indent=2)
    return _structured_normalized_payload(
        payload,
        strategy,
        source_type,
        target_url,
        extraction_schema,
        fit_source,
    )


def _structured_preflight(
    url: Optional[str],
    file_path: Optional[str],
    html: Optional[str],
    fit_markdown: Optional[str],
    strategy: str,
    extraction_schema: Optional[Dict[str, Any]],
) -> tuple[Optional[str], Optional[tuple[Optional[str], str]]]:
    fit_response = _structured_fit_response(fit_markdown, strategy, extraction_schema)
    if fit_response is not None:
        return fit_response, None
    target_result, target_error = _structured_target_result(url, file_path, html)
    if target_error is not None:
        return target_error, None
    return None, target_result


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
    preflight_response, target_result = _structured_preflight(
        url,
        file_path,
        html,
        fit_markdown,
        strategy,
        extraction_schema,
    )
    if preflight_response is not None:
        return preflight_response
    target_url, source_type = cast(tuple[Optional[str], str], target_result)

    extracted_response = await _structured_crawl_response(
        ctx,
        target_url or "",
        run_config,
        strategy,
        extraction_schema,
        extraction_instruction,
        llm_provider,
        content_source,
        fit_source,
    )
    return _structured_response_payload(
        extracted_response,
        strategy,
        source_type,
        target_url or "",
        extraction_schema,
        fit_source,
    )


async def _structured_crawl_response(
    ctx: Context,
    target_url: str,
    run_config: Optional[Dict[str, Any]],
    strategy: str,
    extraction_schema: Optional[Dict[str, Any]],
    extraction_instruction: Optional[str],
    llm_provider: Optional[str],
    content_source: str,
    fit_source: bool,
) -> str:
    return await crawl_to_markdown(
        ctx=ctx,
        url=target_url,
        markdown_variant="raw",
        run_config=run_config,
        index_result=False,
        extraction_strategy=strategy,
        extraction_schema=extraction_schema,
        extraction_instruction=extraction_instruction,
        llm_provider=llm_provider,
        content_source=("fit_html" if fit_source else content_source),
    )


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
            return json.dumps(
                {
                    "success": False,
                    "url": url,
                    "error": getattr(result, "error_message", "No content."),
                },
                indent=2,
            )

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
        return json.dumps(
            {
                "success": True,
                "url": url,
                "selected_variant": variant_key,
                "code_examples": blocks,
                "count": len(blocks),
            },
            indent=2,
        )
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
        validation_error = _index_markdown_validation_error(url, markdown)
        if validation_error is not None:
            return validation_error

        chunks = await chunk_text_according_to_settings(markdown, strategy=chunking_strategy)
        source_type = _infer_source_type(url)
        effective_run_id = _normalize_run_id(run_id) or _generate_run_id("index")
        payload = _index_markdown_payload(
            url=url,
            markdown=markdown,
            chunks=chunks,
            source_type=source_type,
            effective_run_id=effective_run_id,
            metadata=metadata,
        )

        first_chunk_id: Optional[int] = None
        with next(get_session()) as session:
            chunks_stored, first_chunk_id = await _store_index_markdown_payload(session, url, payload)

        return json.dumps(
            {
                "success": True,
                "url": url,
                "chunks_stored": chunks_stored,
                "pages_indexed": 1,
                "run_id": effective_run_id,
                "first_chunk_id": first_chunk_id,
                "chunking_strategy_applied": chunking_strategy or None,
            },
            indent=2,
        )
    except Exception as exc:
        logger.error(f"index_markdown {url}: {exc}", exc_info=True)
        return json.dumps({"success": False, "url": url, "error": str(exc)}, indent=2)


def _index_markdown_validation_error(url: str, markdown: str) -> Optional[str]:
    if markdown and markdown.strip():
        return None
    return json.dumps({"success": False, "url": url, "error": "markdown must be non-empty."}, indent=2)


async def _store_index_markdown_payload(
    session: Session,
    url: str,
    payload: Dict[str, Any],
) -> tuple[int, Optional[int]]:
    chunks_stored = await add_documents_to_db(
        session,
        payload["db_urls"],
        payload["db_contents"],
        payload["db_metas"],
        payload["db_chunks"],
        payload["db_fulldocs"],
    )
    if chunks_stored <= 0:
        return chunks_stored, None
    return chunks_stored, _first_chunk_id(session, url)


def _index_markdown_meta(
    *,
    chunk: str,
    chunk_index: int,
    url: str,
    source_type: str,
    effective_run_id: str,
    metadata: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    now_iso = datetime.now(timezone.utc).isoformat()
    meta = extract_section_info(chunk)
    meta.update(
        {
            "chunk_index": chunk_index,
            "url": url,
            "source": urlparse(url).netloc,
            "source_type": source_type,
            "crawl_time": now_iso,
            "crawl_timestamp": now_iso,
            "content_class": "text",
            "is_active": True,
            "content_hash": hashlib.sha256(chunk.encode("utf-8")).hexdigest(),
            "run_id": effective_run_id,
        }
    )
    if metadata:
        meta.update(metadata)
    return meta


def _index_markdown_payload(
    *,
    url: str,
    markdown: str,
    chunks: List[str],
    source_type: str,
    effective_run_id: str,
    metadata: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "db_urls": [],
        "db_chunks": [],
        "db_contents": [],
        "db_metas": [],
        "db_fulldocs": [],
    }
    for i, chunk in enumerate(chunks):
        payload["db_urls"].append(url)
        payload["db_chunks"].append(i)
        payload["db_contents"].append(chunk)
        payload["db_metas"].append(
            _index_markdown_meta(
                chunk=chunk,
                chunk_index=i,
                url=url,
                source_type=source_type,
                effective_run_id=effective_run_id,
                metadata=metadata,
            )
        )
        payload["db_fulldocs"].append(markdown)
    return payload


def _first_chunk_id(session: Session, url: str) -> Optional[int]:
    first_row = session.exec(
        select(CrawledPage).where(CrawledPage.url == url).order_by(cast(Any, CrawledPage.chunk_number))
    ).first()
    if first_row is None:
        return None
    raw_id = first_row.id
    return raw_id if isinstance(raw_id, int) else None


def _search_response_payload(
    query: str,
    fresh_only: bool,
    threshold: float,
    as_of_dt: Optional[datetime],
    recency_weight: float,
    formatted: List[Dict[str, Any]],
) -> str:
    return json.dumps(
        {
            "success": True,
            "query": query,
            "fresh_only": bool(fresh_only),
            "staleness_threshold": threshold,
            "as_of": as_of_dt.isoformat() if as_of_dt else None,
            "recency_bias": recency_weight,
            "results": formatted,
        },
        indent=2,
    )


def _maybe_filter_fresh_results(
    results: List[Dict[str, Any]],
    threshold: float,
    as_of_dt: Optional[datetime],
    fresh_only: bool,
) -> List[Dict[str, Any]]:
    if not (fresh_only or as_of_dt is not None):
        return results
    return _filter_fresh_results(results, threshold, as_of_dt, bool(fresh_only))


def _maybe_rerank_results(query: str, results: List[Dict[str, Any]], match_count: int) -> List[Dict[str, Any]]:
    if not (settings.USE_RERANKING and results):
        return results
    return rerank_results(query, results, top_k=match_count)


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
        validation_error = _index_markdown_validation_error(url, markdown)
        if validation_error is not None:
            return validation_error
        blocks = extract_code_blocks(markdown)
        if not blocks:
            return json.dumps({"success": True, "url": url, "code_examples_indexed": 0}, indent=2)
        effective_run_id = _normalize_run_id(run_id) or _generate_run_id("index-code")
        await _store_code_examples(url, blocks, effective_run_id)
        return json.dumps(
            {
                "success": True,
                "url": url,
                "code_examples_indexed": len(blocks),
                "run_id": effective_run_id,
            },
            indent=2,
        )
    except Exception as exc:
        logger.error(f"index_code_examples {url}: {exc}", exc_info=True)
        return json.dumps({"success": False, "url": url, "error": str(exc)}, indent=2)


async def _store_code_examples(url: str, blocks: List[Dict[str, Any]], effective_run_id: str) -> None:
    with next(get_session()) as session:
        await add_code_examples_to_db(
            session,
            urls=[url] * len(blocks),
            contents=[b["content"] for b in blocks],
            languages=[b["language"] for b in blocks],
            summaries=[None] * len(blocks),
            metadatas=_code_example_metadatas(url, effective_run_id, len(blocks)),
            chunk_numbers=list(range(len(blocks))),
        )


def _storage_policy_maps(source_policy_records: Sequence[Any]) -> tuple[Dict[str, int], Dict[str, Any]]:
    return (
        {sp.source: int(sp.min_active_docs) for sp in source_policy_records},
        {sp.source: sp for sp in source_policy_records},
    )


def _storage_skip_payload(
    db_size_bytes: int,
    usage_ratio: float,
    warn_pct: float,
    high_pct: float,
    hard_pct: float,
) -> str:
    return json.dumps(
        {
            "success": True,
            "db_size_bytes": db_size_bytes,
            "usage_ratio": round(usage_ratio, 4),
            "pressure_level": _pressure_level(usage_ratio, warn_pct, high_pct, hard_pct),
            "actions_taken": [],
        },
        indent=2,
    )


def _storage_result_payload(
    db_size_bytes: int,
    runtime_values: Dict[str, Any],
    usage_ratio: float,
    warn_pct: float,
    high_pct: float,
    hard_pct: float,
    actions_taken: List[str],
    quota_result: Dict[str, Any],
) -> str:
    return json.dumps(
        {
            "success": True,
            "db_size_bytes": db_size_bytes,
            "db_size_gb": round(db_size_bytes / 1024**3, 3),
            "max_db_size_gb": float(runtime_values["max_gb"]),
            "usage_ratio": round(usage_ratio, 4),
            "pressure_level": _pressure_level(usage_ratio, warn_pct, high_pct, hard_pct),
            "actions_taken": actions_taken,
            "source_quota_overrides": quota_result.get("sources_over_quota", []),
        },
        indent=2,
    )


def _code_example_metadatas(url: str, effective_run_id: str, count: int) -> List[Dict[str, Any]]:
    return [
        {
            "source": urlparse(url).netloc,
            "url": url,
            "source_type": _infer_source_type(url),
            "run_id": effective_run_id,
        }
    ] * count


def _search_filter_meta(
    source: Optional[str],
    content_class: Optional[str],
    markdown_variant: Optional[str],
    extraction_strategy: Optional[str],
) -> Dict[str, Any]:
    filter_meta: Dict[str, Any] = {}
    if source:
        filter_meta["source"] = source
    if content_class:
        filter_meta["content_class"] = content_class.strip().lower()
    if markdown_variant:
        filter_meta["markdown_variant"] = markdown_variant.strip().lower()
    if extraction_strategy:
        filter_meta["extraction_strategy"] = extraction_strategy.strip().lower()
    return filter_meta


def _freshness_threshold(staleness_threshold: float) -> float:
    return max(0.0, min(1.0, float(staleness_threshold)))


def _filter_fresh_results(
    results: List[Dict[str, Any]],
    threshold: float,
    as_of_dt: Optional[datetime],
    fresh_only: bool,
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for row in results:
        metadata = cast(Dict[str, Any], row.get("page_metadata")) if isinstance(row.get("page_metadata"), dict) else {}
        if _is_result_fresh(metadata, threshold, as_of_dt, fresh_only):
            filtered.append(row)
    return filtered


def _recency_weight(recency_bias: float) -> float:
    return max(0.0, min(1.0, float(recency_bias)))


def _recency_rescored_results(
    results: List[Dict[str, Any]],
    recency_weight: float,
    match_count: int,
) -> List[Dict[str, Any]]:
    if recency_weight <= 0 or not results:
        return results
    rescored = [_rescored_result_entry(row, recency_weight) for row in results]
    return sorted(rescored, key=lambda r: float(r.get("final_score") or 0.0), reverse=True)[:match_count]


def _rescored_result_entry(row: Dict[str, Any], recency_weight: float) -> Dict[str, Any]:
    metadata = cast(Dict[str, Any], row.get("page_metadata")) if isinstance(row.get("page_metadata"), dict) else {}
    freshness = _compute_freshness_from_metadata(metadata)
    similarity = float(row.get("similarity_score") or 0.0)
    final_score = similarity * (1.0 + recency_weight * freshness)
    row_copy = dict(row)
    row_copy["freshness_score"] = round(freshness, 6)
    row_copy["final_score"] = round(final_score, 6)
    return row_copy


def _format_search_result_entry(result: Dict[str, Any], include_provenance: bool) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "url": result.get("url"),
        "content": result.get("content"),
        "metadata": result.get("page_metadata"),
        "similarity": result.get("similarity_score"),
    }
    if "final_score" in result:
        entry["final_score"] = result.get("final_score")
    if "freshness_score" in result:
        entry["freshness_score"] = result.get("freshness_score")
    if include_provenance:
        entry["provenance"] = _build_requested_provenance(result.get("page_metadata"))
    return entry


def _format_search_results(results: List[Dict[str, Any]], include_provenance: bool) -> List[Dict[str, Any]]:
    return [_format_search_result_entry(result, include_provenance) for result in results]


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
        filter_meta = _search_filter_meta(source, content_class, markdown_variant, extraction_strategy)
        with next(get_session()) as session:
            results = await search_documents(
                session,
                query,
                match_count=match_count,
                filter_metadata=filter_meta or None,
            )

        as_of_dt = _parse_datetime_utc(as_of) if as_of else None
        threshold = _freshness_threshold(staleness_threshold)
        results = _maybe_filter_fresh_results(results, threshold, as_of_dt, bool(fresh_only))
        results = _maybe_rerank_results(query, results, match_count)

        recency_weight = _recency_weight(recency_bias)
        results = _recency_rescored_results(results, recency_weight, match_count)
        formatted = _format_search_results(results, include_provenance)
        return _search_response_payload(query, bool(fresh_only), threshold, as_of_dt, recency_weight, formatted)
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
                select(CrawledPage).where(CrawledPage.url == url).order_by(cast(Any, CrawledPage.chunk_number))
            ).all()

        fit_rows = _rows_matching_variant(list(rows), "fit_markdown")
        if not fit_rows:
            return json.dumps({"success": False, "url": url, "error": "No stored fit_markdown chunks found."}, indent=2)

        return _markdown_rows_payload(
            url=url,
            rows=fit_rows,
            content_key="fit_markdown",
            variant="fit_markdown",
            include_provenance=include_provenance,
        )
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

        return json.dumps(
            {
                "success": True,
                "query": query,
                "results": results,
            },
            indent=2,
        )
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


def _source_priority_map(session: Session) -> Dict[str, float]:
    return {sp.source: sp.priority_weight for sp in session.exec(select(SourcePolicy)).all()}


def _records_for_scoring(session: Session, model_cls: Any, limit: int) -> List[Any]:
    return list(session.exec(select(model_cls).where(cast(Any, model_cls.is_active).is_(True)).limit(limit)).all())


def _record_source(record: Any, metadata_attr: str) -> str:
    metadata = getattr(record, metadata_attr, {})
    meta = metadata if isinstance(metadata, dict) else {}
    return str(meta.get("source") or "")


def _scored_age_days(record: Any, now: datetime) -> float:
    ref_time = getattr(record, "first_seen_at", None) or getattr(record, "crawl_timestamp")
    return max(0.0, (now - ref_time).total_seconds() / 86400)


def _score_record(record: Any, now: datetime, source_priority: float) -> None:
    age_days = _scored_age_days(record, now)
    density = min(1.0, len(getattr(record, "content", "")) / 5000.0)
    record.staleness_score = compute_staleness_score(age_days)
    record.value_score = compute_value_score(
        hit_count=getattr(record, "hit_count", 0),
        content_density=density,
        age_days=age_days,
        source_priority=source_priority,
    )


def _update_model_scores(
    session: Session,
    model_cls: Any,
    metadata_attr: str,
    source: Optional[str],
    limit: int,
    now: datetime,
    source_priority_map: Dict[str, float],
) -> int:
    updated = 0
    for record in _records_for_scoring(session, model_cls, limit):
        src = _record_source(record, metadata_attr)
        if source and src != source:
            continue
        _score_record(record, now, source_priority_map.get(src, 1.0))
        updated += 1
    return updated


async def compute_value_scores(
    ctx: Context,
    source: Optional[str] = None,
    limit: int = 1000,
) -> str:
    """Recompute and persist value_score and staleness_score for indexed records."""
    try:
        now = datetime.now(timezone.utc)
        with next(get_session()) as session:
            priorities = _source_priority_map(session)
            updated_pages = _update_model_scores(session, CrawledPage, "page_metadata", source, limit, now, priorities)
            updated_examples = _update_model_scores(session, CodeExample, "ex_metadata", source, limit, now, priorities)
            session.commit()

        return json.dumps(
            {
                "success": True,
                "updated_crawled_pages": updated_pages,
                "updated_code_examples": updated_examples,
            },
            indent=2,
        )
    except Exception as exc:
        logger.error(f"compute_value_scores failed: {exc}", exc_info=True)
        return json.dumps({"success": False, "error": str(exc)}, indent=2)


def _source_policy_min_docs_map(session: Session) -> Dict[str, int]:
    return {sp.source: int(sp.min_active_docs) for sp in session.exec(select(SourcePolicy)).all()}


def _eviction_records_for_table(session: Session, tbl_model: Any, limit: int) -> List[Any]:
    model_any = cast(Any, tbl_model)
    rows = session.exec(
        select(tbl_model)
        .where(
            cast(Any, model_any.is_active).is_(True),
            cast(Any, model_any.is_pinned).is_(False),
            model_any.tombstoned_at.is_(None),
        )
        .order_by(model_any.value_score.asc())
        .limit(limit)
    ).all()
    return list(rows)


def _candidate_from_record(tbl_name: str, rec_any: Any) -> Optional[Dict[str, Any]]:
    meta_raw = rec_any.page_metadata if hasattr(rec_any, "page_metadata") else getattr(rec_any, "ex_metadata", {})
    meta = meta_raw if isinstance(meta_raw, dict) else {}
    rec_id = getattr(rec_any, "id", None)
    if not isinstance(rec_id, int):
        return None
    return _candidate_payload(tbl_name, rec_any, rec_id, meta)


def _candidate_payload(tbl_name: str, rec_any: Any, rec_id: int, meta: Dict[str, Any]) -> Dict[str, Any]:
    parsed_last_seen = _parse_datetime_utc(getattr(rec_any, "last_seen_at", None))
    return {
        "table": tbl_name,
        "id": rec_id,
        "url": rec_any.url,
        "canonical_key": _canonical_url_key(meta.get("canonical_url") or rec_any.url),
        "canonical_guard": bool(meta.get("canonical_url") or meta.get("markdown_variant")),
        "chunk_number": rec_any.chunk_number,
        "source": meta.get("source", ""),
        "value_score": round(rec_any.value_score, 4),
        "staleness_score": round(rec_any.staleness_score, 4),
        "hit_count": rec_any.hit_count,
        "last_seen_at": parsed_last_seen.isoformat() if parsed_last_seen is not None else None,
        "content_length": len(rec_any.content),
    }


def _candidate_ids_by_table(candidates: List[Dict[str, Any]]) -> tuple[List[int], List[int]]:
    return (
        [c["id"] for c in candidates if c["table"] == "crawled_pages"],
        [c["id"] for c in candidates if c["table"] == "code_examples"],
    )


def _tombstone_candidate_ids(session: Session, page_ids: List[int], code_ids: List[int], reason: str) -> int:
    evicted = 0
    if page_ids:
        evicted += tombstone_records(session, page_ids, "crawled_pages", reason)
    if code_ids:
        evicted += tombstone_records(session, code_ids, "code_examples", reason)
    return evicted


def _collect_eviction_candidates(session: Session, limit: int, source: Optional[str]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for tbl_model, tbl_name in ((CrawledPage, "crawled_pages"), (CodeExample, "code_examples")):
        candidates.extend(_table_eviction_candidates(session, tbl_model, tbl_name, limit, source))
    return candidates


def _table_eviction_candidates(
    session: Session,
    tbl_model: Any,
    tbl_name: str,
    limit: int,
    source: Optional[str],
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for record in _eviction_records_for_table(session, tbl_model, limit):
        candidate = _candidate_from_record(tbl_name, cast(Any, record))
        if candidate is None:
            continue
        if source and candidate.get("source") != source:
            continue
        candidates.append(candidate)
    return candidates


def _evict_selected_candidates(session: Session, candidates: List[Dict[str, Any]]) -> int:
    page_ids, code_ids = _candidate_ids_by_table(candidates)
    return _tombstone_candidate_ids(session, page_ids, code_ids, "preview_eviction")


async def preview_eviction_plan(
    ctx: Context,
    limit: int = 100,
    source: Optional[str] = None,
    dry_run: bool = True,
) -> str:
    """Preview (or execute) which records would be evicted under storage pressure."""
    try:
        with next(get_session()) as session:
            candidates = _collect_eviction_candidates(session, limit, source)
            source_active_counts, canonical_active_counts = _build_active_coverage_maps(session)
            candidates = _apply_eviction_safeguards(
                candidates,
                _source_policy_min_docs_map(session),
                source_active_counts,
                canonical_active_counts,
            )[:limit]
            total_evicted = 0 if dry_run else _evict_selected_candidates(session, candidates)

            return json.dumps(
                {
                    "success": True,
                    "dry_run": dry_run,
                    "candidates_count": len(candidates),
                    "total_evicted": total_evicted,
                    "candidates": candidates[:50],
                },
                indent=2,
            )
    except Exception as exc:
        logger.error(f"preview_eviction_plan failed: {exc}", exc_info=True)
        return json.dumps({"success": False, "error": str(exc)}, indent=2)


def _storage_policy_values(policy: Optional[StoragePolicy]) -> Dict[str, Any]:
    if policy is None:
        return {
            "max_gb": 10.0,
            "warn_pct": 0.80,
            "high_pct": 0.90,
            "hard_pct": 1.00,
            "grace_h": 24,
            "post_evict": 0.75,
            "max_crawled_pages_mb": None,
            "max_code_examples_mb": None,
        }
    return {
        "max_gb": policy.max_db_size_gb,
        "warn_pct": policy.warn_threshold,
        "high_pct": policy.high_threshold,
        "hard_pct": policy.hard_threshold,
        "grace_h": policy.tombstone_grace_hours,
        "post_evict": policy.target_post_evict_ratio,
        "max_crawled_pages_mb": policy.max_crawled_pages_mb,
        "max_code_examples_mb": policy.max_code_examples_mb,
    }


def _pressure_level(usage_ratio: float, warn_pct: float, high_pct: float, hard_pct: float) -> str:
    if usage_ratio < warn_pct:
        return "ok"
    if usage_ratio < high_pct:
        return "warning"
    if usage_ratio < hard_pct:
        return "high"
    return "critical"


def _storage_count(session: Session, sql: str) -> int:
    row = session.execute(_sql_text(sql)).first()
    return int(row[0]) if row else 0


def _storage_table_size(session: Session, table_name: str) -> int:
    row = session.execute(_sql_text(f"SELECT pg_total_relation_size('{table_name}')")).first()
    return int(row[0]) if row else 0


def _table_storage_stats(session: Session, table_name: str) -> Dict[str, Any]:
    total = _storage_count(session, f"SELECT COUNT(*) FROM {table_name}")
    active = _storage_count(
        session,
        f"SELECT COUNT(*) FROM {table_name} WHERE is_active = TRUE AND tombstoned_at IS NULL",
    )
    tombstoned = _storage_count(session, f"SELECT COUNT(*) FROM {table_name} WHERE tombstoned_at IS NOT NULL")
    pinned = _storage_count(session, f"SELECT COUNT(*) FROM {table_name} WHERE is_pinned = TRUE")
    size_bytes = _storage_table_size(session, table_name)
    return {
        "total_rows": total,
        "active_rows": active,
        "tombstoned_rows": tombstoned,
        "pinned_rows": pinned,
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / 1024**2, 2),
    }


def _storage_report_base(session: Session) -> Dict[str, Any]:
    db_size = _get_db_size_bytes(session)
    policy = session.exec(select(StoragePolicy)).first()
    max_gb = policy.max_db_size_gb if policy else 10.0
    max_bytes = int(max_gb * 1024**3)
    usage_ratio = db_size / max_bytes if max_bytes > 0 else 0.0
    return {
        "success": True,
        "db_size_bytes": db_size,
        "db_size_mb": round(db_size / 1024**2, 2),
        "max_db_size_gb": max_gb,
        "usage_ratio": round(usage_ratio, 4),
        "pressure_level": _pressure_level(usage_ratio, 0.80, 0.90, 1.00),
        "tables": {
            "crawled_pages": _table_storage_stats(session, "crawled_pages"),
            "code_examples": _table_storage_stats(session, "code_examples"),
        },
    }


def _source_storage_rows(session: Session) -> List[Dict[str, Any]]:
    src_rows = session.exec(  # type: ignore[call-overload]
        _sql_text(
            "SELECT metadata->>'source' as src, COUNT(*) as total, "
            "SUM(CASE WHEN is_active THEN 1 ELSE 0 END) as active "
            "FROM crawled_pages "
            "GROUP BY metadata->>'source' ORDER BY total DESC LIMIT 50"
        )
    ).all()
    return [{"source": r[0] or "(unknown)", "total": int(r[1]), "active": int(r[2])} for r in src_rows]


def _compact_expired_tombstones(session: Session, grace_cutoff: datetime) -> int:
    compacted = 0
    for tbl_model in (CrawledPage, CodeExample):
        model_any = cast(Any, tbl_model)
        expired = session.exec(
            select(tbl_model).where(
                model_any.tombstoned_at.isnot(None),
                model_any.tombstoned_at <= grace_cutoff,
            )
        ).all()
        for rec in expired:
            session.delete(rec)
            compacted += 1
    if compacted:
        session.commit()
    return compacted


def _stale_record_ids(session: Session, model_cls: Any, stale_threshold: float, limit: int) -> List[int]:
    model_any = cast(Any, model_cls)
    stale_recs = session.exec(
        select(model_cls)
        .where(
            model_any.staleness_score >= stale_threshold,
            cast(Any, model_any.is_pinned).is_(False),
            model_any.tombstoned_at.is_(None),
            cast(Any, model_any.is_active).is_(True),
        )
        .limit(limit)
    ).all()
    ids: List[int] = []
    for row in stale_recs:
        row_id = getattr(row, "id", None)
        if isinstance(row_id, int):
            ids.append(row_id)
    return ids


def _tombstone_stale_records(session: Session, stale_threshold: float = 0.8, limit: int = 500) -> int:
    pruned = 0
    for model_cls in (CrawledPage, CodeExample):
        ids = _stale_record_ids(session, model_cls, stale_threshold=stale_threshold, limit=limit)
        if not ids:
            continue
        pruned += tombstone_records(session, ids, model_cls.__tablename__, "high_pressure_stale_prune")
    return pruned


def _candidate_metadata(record_any: Any) -> Dict[str, Any]:
    if hasattr(record_any, "page_metadata") and isinstance(record_any.page_metadata, dict):
        return record_any.page_metadata
    metadata = getattr(record_any, "ex_metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _value_eviction_candidates(session: Session) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for tbl_model, table_name in ((CrawledPage, "crawled_pages"), (CodeExample, "code_examples")):
        model_any = cast(Any, tbl_model)
        rows = session.exec(
            select(tbl_model)
            .where(
                cast(Any, model_any.is_pinned).is_(False),
                model_any.tombstoned_at.is_(None),
                cast(Any, model_any.is_active).is_(True),
            )
            .limit(200)
        ).all()
        candidates.extend(_rows_to_eviction_candidates(rows, table_name))
    return candidates


def _rows_to_eviction_candidates(rows: Any, table_name: str) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for row in rows:
        row_any = cast(Any, row)
        row_id = getattr(row_any, "id", None)
        if not isinstance(row_id, int):
            continue
        candidates.append(_row_eviction_candidate(table_name, row_any, row_id))
    return candidates


def _row_eviction_candidate(table_name: str, row_any: Any, row_id: int) -> Dict[str, Any]:
    meta = _candidate_metadata(row_any)
    score_fields = _row_eviction_scores(row_any)
    return {
        "table": table_name,
        "id": row_id,
        "url": row_any.url,
        "canonical_key": _canonical_url_key(meta.get("canonical_url") or row_any.url),
        "canonical_guard": bool(meta.get("canonical_url") or meta.get("markdown_variant")),
        "source": str(meta.get("source", "")),
        **score_fields,
    }


def _row_eviction_scores(row_any: Any) -> Dict[str, Any]:
    parsed_last_seen = _parse_datetime_utc(getattr(row_any, "last_seen_at", None))
    return {
        "value_score": float(getattr(row_any, "value_score", 0.0) or 0.0),
        "staleness_score": float(getattr(row_any, "staleness_score", 0.0) or 0.0),
        "hit_count": int(getattr(row_any, "hit_count", 0) or 0),
        "last_seen_at": parsed_last_seen.isoformat() if parsed_last_seen is not None else None,
    }


def _tombstone_selected_candidates(session: Session, selected_candidates: List[Dict[str, Any]]) -> int:
    page_ids, code_ids = _candidate_ids_by_table(selected_candidates)
    return _tombstone_candidate_ids(session, page_ids, code_ids, "hard_pressure_value_evict")


def _enforce_hard_pressure_eviction(
    session: Session,
    db_size_bytes: int,
    max_bytes: int,
    post_evict: float,
    source_policy_map: Dict[str, int],
) -> int:
    target_bytes = int(max_bytes * post_evict)
    if db_size_bytes <= target_bytes:
        return 0
    source_active_counts, canonical_active_counts = _build_active_coverage_maps(session)
    candidates = _value_eviction_candidates(session)
    selected_candidates = _apply_eviction_safeguards(
        candidates,
        source_policy_map,
        source_active_counts,
        canonical_active_counts,
    )
    return _tombstone_selected_candidates(session, selected_candidates[:200])


def _storage_runtime_values(policy_values: Dict[str, Any]) -> Dict[str, Any]:
    max_gb = float(policy_values["max_gb"])
    return {
        "max_gb": max_gb,
        "warn_pct": float(policy_values["warn_pct"]),
        "high_pct": float(policy_values["high_pct"]),
        "hard_pct": float(policy_values["hard_pct"]),
        "grace_h": int(policy_values["grace_h"]),
        "post_evict": float(policy_values["post_evict"]),
        "max_crawled_pages_mb": cast(Optional[int], policy_values["max_crawled_pages_mb"]),
        "max_code_examples_mb": cast(Optional[int], policy_values["max_code_examples_mb"]),
        "max_bytes": int(max_gb * 1024**3),
    }


def _record_ttl_action(actions_taken: List[str], ttl_deleted: Dict[str, int]) -> None:
    ttl_total = ttl_deleted["crawled_pages"] + ttl_deleted["code_examples"]
    if not ttl_total:
        return
    actions_taken.append(
        f"hard-ttl deleted {ttl_total} records "
        f"(pages={ttl_deleted['crawled_pages']}, code={ttl_deleted['code_examples']})"
    )


def _record_table_budget_action(actions_taken: List[str], table_budget_evicted: Dict[str, int]) -> None:
    table_budget_total = table_budget_evicted["crawled_pages"] + table_budget_evicted["code_examples"]
    if not table_budget_total:
        return
    actions_taken.append(
        f"table-budget tombstoned {table_budget_total} records "
        f"(pages={table_budget_evicted['crawled_pages']}, code={table_budget_evicted['code_examples']})"
    )


def _storage_actions(
    *,
    session: Session,
    usage_ratio: float,
    high_pct: float,
    hard_pct: float,
    grace_cutoff: datetime,
    max_bytes: int,
    db_size_bytes: int,
    post_evict: float,
    source_policy_map: Dict[str, int],
    source_policies_by_source: Dict[str, Any],
    max_crawled_pages_mb: Optional[int],
    max_code_examples_mb: Optional[int],
) -> tuple[List[str], Dict[str, Any]]:
    actions_taken: List[str] = []
    ttl_deleted = _apply_hard_ttl_delete(session, source_policies_by_source)
    _record_ttl_action(actions_taken, ttl_deleted)

    quota_result = _enforce_source_quotas(session, source_policies_by_source)
    if quota_result["quota_evicted"]:
        actions_taken.append(f"source-quota tombstoned {quota_result['quota_evicted']} records")

    table_budget_evicted = _enforce_table_budgets(
        session,
        max_crawled_pages_mb=max_crawled_pages_mb,
        max_code_examples_mb=max_code_examples_mb,
    )
    _record_table_budget_action(actions_taken, table_budget_evicted)

    compacted = _compact_expired_tombstones(session, grace_cutoff)
    if compacted:
        actions_taken.append(f"compacted {compacted} expired tombstoned records")

    if usage_ratio >= high_pct:
        pruned = _tombstone_stale_records(session, stale_threshold=0.8, limit=500)
        actions_taken.append(f"tombstoned {pruned} stale records (high pressure)")

    if usage_ratio >= hard_pct:
        evicted = _enforce_hard_pressure_eviction(session, db_size_bytes, max_bytes, post_evict, source_policy_map)
        actions_taken.append(f"value-evicted {evicted} records (hard pressure)")

    return actions_taken, quota_result


async def enforce_storage_budget(
    ctx: Context,
    force: bool = False,
) -> str:
    """Check storage usage and trigger tiered eviction if budget thresholds are exceeded."""
    try:
        with next(get_session()) as session:
            policy = session.exec(select(StoragePolicy)).first()
            source_policy_records = session.exec(select(SourcePolicy)).all()
            source_policy_map, source_policies_by_source = _storage_policy_maps(source_policy_records)
            runtime_values = _storage_runtime_values(_storage_policy_values(policy))

            max_bytes = int(runtime_values["max_bytes"])
            db_size_bytes = _get_db_size_bytes(session)
            usage_ratio = db_size_bytes / max_bytes if max_bytes > 0 else 0.0
            warn_pct = float(runtime_values["warn_pct"])
            high_pct = float(runtime_values["high_pct"])
            hard_pct = float(runtime_values["hard_pct"])

            if usage_ratio < warn_pct and not force:
                return _storage_skip_payload(db_size_bytes, usage_ratio, warn_pct, high_pct, hard_pct)

            grace_cutoff = datetime.now(timezone.utc) - timedelta(hours=int(runtime_values["grace_h"]))
            actions_taken, quota_result = _storage_actions(
                session=session,
                usage_ratio=usage_ratio,
                high_pct=high_pct,
                hard_pct=hard_pct,
                grace_cutoff=grace_cutoff,
                max_bytes=max_bytes,
                db_size_bytes=db_size_bytes,
                post_evict=float(runtime_values["post_evict"]),
                source_policy_map=source_policy_map,
                source_policies_by_source=source_policies_by_source,
                max_crawled_pages_mb=cast(Optional[int], runtime_values["max_crawled_pages_mb"]),
                max_code_examples_mb=cast(Optional[int], runtime_values["max_code_examples_mb"]),
            )
            return _storage_result_payload(
                db_size_bytes,
                runtime_values,
                usage_ratio,
                warn_pct,
                high_pct,
                hard_pct,
                actions_taken,
                quota_result,
            )
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
            model_any = cast(Any, model_cls)
            records = session.exec(select(model_cls).where(model_any.id.in_(record_ids))).all()
            for rec in records:
                rec.is_pinned = True
            session.commit()
            return json.dumps(
                {
                    "success": True,
                    "pinned_count": len(records),
                    "table": table,
                },
                indent=2,
            )
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
            model_any = cast(Any, model_cls)
            records = session.exec(select(model_cls).where(model_any.id.in_(record_ids))).all()
            for rec in records:
                rec.is_pinned = False
            session.commit()
            return json.dumps(
                {
                    "success": True,
                    "unpinned_count": len(records),
                    "table": table,
                },
                indent=2,
            )
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
            report = _storage_report_base(session)

            if group_by == "source":
                report["by_source"] = _source_storage_rows(session)

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
            model_any = cast(Any, model_cls)

            records = session.exec(
                select(model_cls).where(
                    model_any.id.in_(record_ids),
                    model_any.tombstoned_at.isnot(None),
                )
            ).all()

            restored, skipped = _restore_records_within_grace(records, grace_cutoff)

            session.commit()
            return json.dumps(
                {
                    "success": True,
                    "restored_count": restored,
                    "skipped_count": skipped,
                    "table": table,
                },
                indent=2,
            )
    except Exception as exc:
        logger.error(f"restore_tombstoned_records failed: {exc}", exc_info=True)
        return json.dumps({"success": False, "error": str(exc)}, indent=2)


def _restore_records_within_grace(records: Sequence[Any], grace_cutoff: datetime) -> tuple[int, int]:
    restored = 0
    skipped = 0
    for rec in records:
        rec_any = cast(Any, rec)
        ts = _normalized_tombstone_timestamp(rec_any.tombstoned_at)
        if _restorable_tombstone(ts, grace_cutoff):
            _restore_tombstoned_record(rec_any)
            restored += 1
        else:
            skipped += 1
    return restored, skipped


def _normalized_tombstone_timestamp(value: Optional[datetime]) -> Optional[datetime]:
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _restorable_tombstone(ts: Optional[datetime], grace_cutoff: datetime) -> bool:
    return ts is not None and ts >= grace_cutoff


def _restore_tombstoned_record(record: Any) -> None:
    record.tombstoned_at = None
    record.is_active = True


def _policies_for_source(policies: Sequence[SourcePolicy], source: Optional[str]) -> List[SourcePolicy]:
    if not source:
        return list(policies)
    return [policy for policy in policies if policy.source == source]


def _policy_backoff_status(policy: SourcePolicy, now: datetime) -> Optional[str]:
    next_retry_at = _parse_datetime_utc(getattr(policy, "next_retry_at", None))
    if next_retry_at is None or next_retry_at <= now:
        return None
    return next_retry_at.isoformat()


def _latest_source_crawl_time(session: Session, source_name: str) -> Optional[datetime]:
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
        {"source": source_name},
    ).first()
    last_crawled = row[0] if row else None
    if isinstance(last_crawled, datetime) and last_crawled.tzinfo is None:
        return last_crawled.replace(tzinfo=timezone.utc)
    return last_crawled if isinstance(last_crawled, datetime) else None


def _is_source_due(now: datetime, last_crawled: Optional[datetime], interval_h: int) -> bool:
    if last_crawled is None:
        return True
    return (now - last_crawled) >= timedelta(hours=interval_h)


def _due_sources_from_policies(
    session: Session,
    policies: List[SourcePolicy],
    now: datetime,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    due_sources: List[Dict[str, Any]] = []
    backoff_skipped: List[Dict[str, Any]] = []
    for policy in policies:
        next_retry_iso = _policy_backoff_status(policy, now)
        if next_retry_iso is not None:
            backoff_skipped.append({"source": policy.source, "next_retry_at": next_retry_iso})
            continue
        interval_h = max(1, int(getattr(policy, "recrawl_interval_hours", 168) or 168))
        last_crawled = _latest_source_crawl_time(session, policy.source)
        if not _is_source_due(now, last_crawled, interval_h):
            continue
        due_sources.append(_due_source_entry(policy.source, interval_h, last_crawled))
    return due_sources, backoff_skipped


def _due_source_entry(source: str, interval_h: int, last_crawled: Optional[datetime]) -> Dict[str, Any]:
    return {
        "source": source,
        "recrawl_interval_hours": interval_h,
        "last_crawled_at": last_crawled.isoformat() if isinstance(last_crawled, datetime) else None,
    }


def _source_url(source_name: str) -> str:
    if source_name.startswith(("http://", "https://")):
        return source_name
    return f"https://{source_name}"


def _record_recrawl_failure(policy: Optional[SourcePolicy], now: datetime) -> None:
    if policy is None:
        return
    policy.consecutive_failures = int(getattr(policy, "consecutive_failures", 0) or 0) + 1
    backoff_h = _compute_retry_backoff_hours(policy)
    policy.next_retry_at = now + timedelta(hours=backoff_h) if backoff_h > 0 else None


def _record_recrawl_success(policy: Optional[SourcePolicy]) -> None:
    if policy is None:
        return
    policy.consecutive_failures = 0
    policy.next_retry_at = None


def _source_record_ids(session: Session, model_cls: Any, source_name: str) -> List[int]:
    rows = session.exec(
        select(model_cls).where(
            cast(Any, model_cls.is_active).is_(True),
            cast(Any, model_cls.tombstoned_at).is_(None),
        )
    ).all()
    ids: List[int] = []
    for row in rows:
        row_id = getattr(row, "id", None)
        if not isinstance(row_id, int):
            continue
        if str(_record_metadata(row).get("source") or "") != source_name:
            continue
        ids.append(row_id)
    return ids


def _apply_dead_page_tombstones(session: Session, source_name: str) -> int:
    page_ids = _source_record_ids(session, CrawledPage, source_name)
    code_ids = _source_record_ids(session, CodeExample, source_name)
    tombstoned = 0
    if page_ids:
        tombstoned += tombstone_records(session, page_ids, "crawled_pages", "dead_page_policy_404_410")
    if code_ids:
        tombstoned += tombstone_records(session, code_ids, "code_examples", "dead_page_policy_404_410")
    return tombstoned


async def _recrawl_due_source(
    ctx: Context,
    session: Session,
    source_name: str,
    policy: Optional[SourcePolicy],
    now: datetime,
) -> Dict[str, Any]:
    crawl_url = _source_url(source_name)
    duplicate_retired = 0
    superseded_retired = 0
    dead_page_tombstoned = 0
    recrawled_entry: Optional[Dict[str, Any]] = None
    failure_entry: Optional[Dict[str, Any]] = None

    try:
        result = await crawl_to_markdown(
            ctx=ctx,
            url=crawl_url,
            markdown_variant="raw",
            index_result=True,
        )
        parsed = json.loads(result)
        if parsed.get("success"):
            _record_recrawl_success(policy)
            recrawled_entry = {
                "source": source_name,
                "url": crawl_url,
                "pages_crawled": parsed.get("pages_crawled", 1),
            }
            try:
                retire_stats = _retire_source_duplicates_and_superseded(session, source_name)
                duplicate_retired += retire_stats["duplicate_retired"]
                superseded_retired += retire_stats["superseded_retired"]
            except Exception as retire_exc:  # pragma: no cover
                logger.warning(f"retire_source_duplicates_and_superseded failed for {source_name}: {retire_exc}")
        else:
            error_message = parsed.get("error", "recrawl failed")
            failure_entry = {"source": source_name, "url": crawl_url, "error": error_message}
            _record_recrawl_failure(policy, now)
            if _is_dead_page_error(error_message):
                dead_page_tombstoned += _apply_dead_page_tombstones(session, source_name)
    except Exception as exc:  # pragma: no cover
        failure_entry = {"source": source_name, "url": crawl_url, "error": str(exc)}
        _record_recrawl_failure(policy, now)

    return {
        "recrawled": recrawled_entry,
        "failure": failure_entry,
        "dead_page_tombstoned": dead_page_tombstoned,
        "duplicate_retired": duplicate_retired,
        "superseded_retired": superseded_retired,
    }


def _accumulate_recrawl_status(summary: Dict[str, Any], recrawl_status: Dict[str, Any]) -> None:
    if recrawl_status["recrawled"] is not None:
        summary["recrawled"].append(recrawl_status["recrawled"])
    if recrawl_status["failure"] is not None:
        summary["failures"].append(recrawl_status["failure"])
    summary["dead_page_tombstoned"] += int(recrawl_status["dead_page_tombstoned"])
    summary["duplicate_retired"] += int(recrawl_status["duplicate_retired"])
    summary["superseded_retired"] += int(recrawl_status["superseded_retired"])


def _recrawl_due_payload(
    due_sources: List[Dict[str, Any]],
    backoff_skipped: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> str:
    return json.dumps(
        {
            "success": True,
            "due_count": len(due_sources),
            "backoff_skipped_count": len(backoff_skipped),
            "recrawled_count": len(summary["recrawled"]),
            "failed_count": len(summary["failures"]),
            "dead_page_tombstoned": summary["dead_page_tombstoned"],
            "duplicate_retired": summary["duplicate_retired"],
            "superseded_retired": summary["superseded_retired"],
            "recrawled_sources": summary["recrawled"],
            "backoff_skipped": backoff_skipped,
            "failures": summary["failures"],
        },
        indent=2,
    )


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
        source_policy_map: Dict[str, Any] = {}
        summary: Dict[str, Any] = {
            "recrawled": [],
            "failures": [],
            "dead_page_tombstoned": 0,
            "duplicate_retired": 0,
            "superseded_retired": 0,
        }

        with next(get_session()) as session:
            policies = _policies_for_source(session.exec(select(SourcePolicy)).all(), source)
            source_policy_map = {p.source: p for p in policies}
            due_sources, backoff_skipped = _due_sources_from_policies(session, policies, now)

            for item in due_sources:
                src = item["source"]
                policy = source_policy_map.get(src)
                recrawl_status = await _recrawl_due_source(ctx, session, src, policy, now)
                _accumulate_recrawl_status(summary, recrawl_status)

            session.commit()

        return _recrawl_due_payload(due_sources, backoff_skipped, summary)
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
            deleted_by_table = _prune_tombstoned_rows(session, cutoff, force)

            total_deleted = sum(deleted_by_table.values())

            ttl_deleted = _apply_hard_ttl_delete(session, source_policies)
            total_deleted += ttl_deleted["crawled_pages"] + ttl_deleted["code_examples"]
            session.commit()

        return json.dumps(
            {
                "success": True,
                "force": force,
                "grace_hours": grace_hours,
                "hard_deleted_count": total_deleted,
                "deleted_by_table": deleted_by_table,
                "hard_ttl_deleted_by_table": ttl_deleted,
            },
            indent=2,
        )
    except Exception as exc:
        logger.error(f"prune_stale_content failed: {exc}", exc_info=True)
        return json.dumps({"success": False, "error": str(exc)}, indent=2)


def _prune_tombstoned_rows(session: Session, cutoff: datetime, force: bool) -> Dict[str, int]:
    deleted_by_table: Dict[str, int] = {"crawled_pages": 0, "code_examples": 0}
    for model_cls, table_name in ((CrawledPage, "crawled_pages"), (CodeExample, "code_examples")):
        for rec in _tombstoned_rows_for_delete(session, model_cls, cutoff, force):
            session.delete(rec)
            deleted_by_table[table_name] += 1
    return deleted_by_table


def _tombstoned_rows_for_delete(session: Session, model_cls: Any, cutoff: datetime, force: bool) -> List[Any]:
    model_any = cast(Any, model_cls)
    query = select(model_cls).where(model_any.tombstoned_at.isnot(None))
    if not force:
        query = query.where(model_any.tombstoned_at <= cutoff)
    return list(session.exec(query).all())


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
            deleted_by_table = _hard_delete_tombstoned_rows(session, cutoff)

            total_deleted = sum(deleted_by_table.values())
            session.commit()

        return json.dumps(
            {
                "success": True,
                "max_age_hours": max_age_hours,
                "hard_deleted_count": total_deleted,
                "deleted_by_table": deleted_by_table,
            },
            indent=2,
        )
    except Exception as exc:
        logger.error(f"hard_delete_tombstones failed: {exc}", exc_info=True)
        return json.dumps({"success": False, "error": str(exc)}, indent=2)


def _hard_delete_tombstoned_rows(session: Session, cutoff: Optional[datetime]) -> Dict[str, int]:
    deleted_by_table: Dict[str, int] = {"crawled_pages": 0, "code_examples": 0}
    for model_cls, table_name in ((CrawledPage, "crawled_pages"), (CodeExample, "code_examples")):
        for rec in _hard_delete_rows(session, model_cls, cutoff):
            session.delete(rec)
            deleted_by_table[table_name] += 1
    return deleted_by_table


def _hard_delete_rows(session: Session, model_cls: Any, cutoff: Optional[datetime]) -> List[Any]:
    model_any = cast(Any, model_cls)
    query = select(model_cls).where(model_any.tombstoned_at.isnot(None))
    if cutoff is not None:
        query = query.where(model_any.tombstoned_at <= cutoff)
    return list(session.exec(query).all())


def _drift_thresholds(major_threshold: float, minor_threshold: float) -> tuple[float, float]:
    major_cut = max(0.0, min(1.0, float(major_threshold)))
    minor_cut = max(0.0, min(major_cut, float(minor_threshold)))
    return major_cut, minor_cut


def _drift_rows(session: Session, source: Optional[str]) -> tuple[List[Any], List[Any]]:
    active_rows = _drift_query_rows(session, active=True)
    removed_rows = _drift_query_rows(session, active=False)
    active_filtered = _drift_filtered_rows(active_rows, active=True)
    removed_filtered = _drift_filtered_rows(removed_rows, active=False)
    return _filter_rows_by_source(active_filtered, source), _filter_rows_by_source(removed_filtered, source)


def _drift_query_rows(session: Session, active: bool) -> List[Any]:
    if active:
        query = select(CrawledPage).where(
            cast(Any, CrawledPage.is_active).is_(True),
            cast(Any, CrawledPage.tombstoned_at).is_(None),
        )
    else:
        query = select(CrawledPage).where(
            cast(Any, CrawledPage.is_active).is_(False) | cast(Any, CrawledPage.tombstoned_at).isnot(None)
        )
    return list(session.exec(query).all())


def _drift_filtered_rows(rows: List[Any], active: bool) -> List[Any]:
    predicate = _is_active_drift_row if active else _is_removed_drift_row
    return [row for row in rows if predicate(row)]


def _is_active_drift_row(row: Any) -> bool:
    return bool(getattr(row, "is_active", True)) and getattr(row, "tombstoned_at", None) is None


def _is_removed_drift_row(row: Any) -> bool:
    return (not bool(getattr(row, "is_active", True))) or getattr(row, "tombstoned_at", None) is not None


def _selective_reembed_candidates(active_rows: List[Any], major_ids: List[int]) -> List[int]:
    major_set = set(major_ids)
    candidates: List[int] = []
    for row in active_rows:
        row_id = _eligible_reembed_row_id(row, major_set)
        if row_id is None:
            continue
        candidates.append(row_id)
    return candidates


def _eligible_reembed_row_id(row: Any, major_set: set[int]) -> Optional[int]:
    row_id = getattr(row, "id", None)
    if not isinstance(row_id, int):
        return None
    if row_id not in major_set:
        return None
    if int(getattr(row, "hit_count", 0) or 0) <= 0:
        return None
    return row_id


def _drift_payload(
    *,
    source: Optional[str],
    trigger_selective_reembed: bool,
    minor_cut: float,
    major_cut: float,
    active_rows: List[Any],
    removed_rows: List[Any],
    unchanged: List[int],
    minor: List[int],
    major: List[int],
    selective_reembed_candidates: List[int],
    selective_reembedded: int,
) -> Dict[str, Any]:
    return {
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
    }


async def detect_content_drift(
    ctx: Context,
    source: Optional[str] = None,
    major_threshold: float = 0.85,
    minor_threshold: float = 0.60,
    trigger_selective_reembed: bool = False,
) -> str:
    """Classify indexed URLs as unchanged/minor/major drift using lifecycle signals."""
    try:
        major_cut, minor_cut = _drift_thresholds(major_threshold, minor_threshold)
        with next(get_session()) as session:
            active_rows, removed_rows = _drift_rows(session, source)
            unchanged, minor, major = _classify_staleness_row_ids(active_rows, minor_cut=minor_cut, major_cut=major_cut)
            selective_reembed_candidates = _selective_reembed_candidates(active_rows, major)
            selective_reembedded = 0
            if trigger_selective_reembed and selective_reembed_candidates:
                selective_reembedded = await _run_selective_reembed(session, selective_reembed_candidates)

        payload = _drift_payload(
            source=source,
            trigger_selective_reembed=trigger_selective_reembed,
            minor_cut=minor_cut,
            major_cut=major_cut,
            active_rows=active_rows,
            removed_rows=removed_rows,
            unchanged=unchanged,
            minor=minor,
            major=major,
            selective_reembed_candidates=selective_reembed_candidates,
            selective_reembedded=selective_reembedded,
        )
        return json.dumps(payload, indent=2)
    except Exception as exc:
        logger.error(f"detect_content_drift failed: {exc}", exc_info=True)
        return json.dumps({"success": False, "error": str(exc)}, indent=2)
