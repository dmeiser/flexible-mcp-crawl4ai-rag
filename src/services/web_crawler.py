"""Core crawling and chunking logic for web pages."""

import logging
import re
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urldefrag, urlparse
from xml.etree import ElementTree

import httpx
import nltk
from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig, MemoryAdaptiveDispatcher

from src.config import ChunkStrategy, settings

logger = logging.getLogger(__name__)

# Ensure NLTK punkt tokenizer is available
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:  # pragma: no cover
    try:  # pragma: no cover
        nltk.download("punkt_tab", quiet=True)
    except Exception:
        pass


def is_sitemap(url: str) -> bool:
    return url.endswith("sitemap.xml") or "sitemap" in urlparse(url).path


def is_txt(url: str) -> bool:
    return url.endswith(".txt")


async def parse_sitemap(sitemap_url: str) -> List[str]:
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(sitemap_url)
            resp.raise_for_status()
            tree = ElementTree.fromstring(resp.content)
            return [loc.text for loc in tree.findall(".//{*}loc") if loc.text]
    except Exception as exc:
        _log_sitemap_exception(sitemap_url, exc)
    return []


def _log_sitemap_exception(sitemap_url: str, exc: Exception) -> None:
    if isinstance(exc, httpx.HTTPStatusError):
        logger.error(f"HTTP error fetching sitemap {sitemap_url}: {exc.response.status_code}")
        return
    if isinstance(exc, httpx.RequestError):
        logger.error(f"Request error fetching sitemap {sitemap_url}: {exc}")
        return
    if isinstance(exc, ElementTree.ParseError):
        logger.error(f"XML parse error for sitemap {sitemap_url}: {exc}")
        return
    logger.error(f"Unexpected error parsing sitemap {sitemap_url}: {exc}")


# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------


def _fixed_char_chunking(text: str, size: int, overlap: int) -> List[str]:
    if not text:
        return []
    if size <= 0:
        if not text.strip():
            return []
        return [text]
    step = max(1, size - overlap)
    return _chunk_text_by_step(text, size, step)


def _chunk_text_by_step(text: str, size: int, step: int) -> List[str]:
    chunks: List[str] = []
    for start in range(0, len(text), step):
        chunk = text[start : start + size]
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def _paragraph_chunking(text: str, size: int, overlap: int) -> List[str]:
    if not text:
        return []
    chunks = []
    for paragraph in text.split("\n\n"):
        p = paragraph.strip()
        if not p:
            continue
        if len(p) <= size:
            chunks.append(p)
        else:
            chunks.extend(_fixed_char_chunking(p, size, overlap))
    return chunks


def _collect_overlap_sentences(current: List[str], overlap: int) -> List[str]:
    if overlap <= 0:
        return []
    overlap_sents: List[str] = []
    overlap_len = 0
    for sentence in reversed(current):
        if overlap_len + len(sentence) > overlap:
            break
        overlap_sents.insert(0, sentence)
        overlap_len += len(sentence) + 1
    return overlap_sents


def _finalize_sentence_chunk(current: List[str], overlap: int) -> tuple[List[str], int]:
    next_current = _collect_overlap_sentences(current, overlap)
    next_len = sum(len(s) for s in next_current) + max(0, len(next_current) - 1)
    return next_current, next_len


def _tokenize_sentences(text: str, size: int, overlap: int) -> List[str]:
    try:
        return nltk.sent_tokenize(text)
    except Exception as exc:
        logger.warning(f"NLTK sent_tokenize failed ({exc}), falling back to paragraph chunking.")
        return _paragraph_chunking(text, size, overlap)


def _add_sentence_to_chunk(
    sentence: str,
    current: List[str],
    current_len: int,
    size: int,
    overlap: int,
) -> tuple[Optional[str], List[str], int]:
    if _can_append_sentence(current, current_len, sentence, size):
        next_current, next_len = _append_sentence(current, current_len, sentence)
        return None, next_current, next_len

    completed_chunk = " ".join(current)
    next_current, next_len = _finalize_sentence_chunk(current, overlap)
    next_current, next_len = _append_sentence(next_current, next_len, sentence)
    return completed_chunk, next_current, next_len


def _can_append_sentence(current: List[str], current_len: int, sentence: str, size: int) -> bool:
    if not current:
        return True
    sentence_length = len(sentence) + 1
    return current_len + sentence_length <= size


def _append_sentence(current: List[str], current_len: int, sentence: str) -> tuple[List[str], int]:
    current.append(sentence)
    separator_len = 1 if len(current) > 1 else 0
    return current, current_len + len(sentence) + separator_len


def _build_sentence_chunks(sentences: List[str], size: int, overlap: int) -> List[str]:
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for sentence in sentences:
        completed_chunk, current, current_len = _add_sentence_to_chunk(sentence, current, current_len, size, overlap)
        if completed_chunk:
            chunks.append(completed_chunk)
    if current:
        chunks.append(" ".join(current))
    return chunks


def _sentence_chunking(text: str, size: int, overlap: int) -> List[str]:
    if not text:
        return []
    sentences = _tokenize_sentences(text, size, overlap)
    if not sentences:
        return []
    chunks = _build_sentence_chunks(sentences, size, overlap)
    return [c for c in chunks if c.strip()]


def _resolve_chunk_strategy(strategy: Optional[str]) -> ChunkStrategy:
    if not strategy:
        return settings.CHUNK_STRATEGY
    try:
        return ChunkStrategy(strategy.lower())
    except ValueError:
        return settings.CHUNK_STRATEGY


def _chunking_function(strategy: ChunkStrategy) -> Callable[[str, int, int], List[str]]:
    chunking_functions: Dict[ChunkStrategy, Callable[[str, int, int], List[str]]] = {
        ChunkStrategy.FIXED: _fixed_char_chunking,
        ChunkStrategy.SENTENCE: _sentence_chunking,
        ChunkStrategy.PARAGRAPH: _paragraph_chunking,
    }
    return chunking_functions.get(strategy, _paragraph_chunking)


async def chunk_text_according_to_settings(text: str, strategy: Optional[str] = None) -> List[str]:
    """Chunk text using the strategy and parameters from settings.

    Args:
        text: The text to chunk.
        strategy: Optional override for the chunking strategy (``"paragraph"``,
            ``"sentence"``, ``"fixed"``).  Falls back to the server-wide
            ``settings.CHUNK_STRATEGY`` when omitted or invalid.
    """
    if not text:
        return []
    size = settings.CHUNK_SIZE
    overlap = settings.CHUNK_OVERLAP
    effective_strategy = _resolve_chunk_strategy(strategy)
    chunking_function = _chunking_function(effective_strategy)
    return chunking_function(text, size, overlap)


# ---------------------------------------------------------------------------
# Crawl helpers
# ---------------------------------------------------------------------------


async def crawl_markdown_file(crawler: AsyncWebCrawler, url: str) -> List[Dict[str, Any]]:
    """Crawl a single URL and return [{url, markdown}]."""
    run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
    result = await crawler.arun(url=url, config=run_config)
    if result.success and result.markdown:
        return [{"url": url, "markdown": result.markdown}]
    logger.warning(f"Crawl failed or no content for {url}: {getattr(result, 'error_message', 'unknown')}")
    return []


async def crawl_batch(
    crawler: AsyncWebCrawler,
    urls: List[str],
    max_concurrent: int = 10,
) -> List[Dict[str, Any]]:
    """Crawl multiple URLs concurrently and return list of {url, markdown}."""
    if not urls:
        return []
    run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
    dispatcher = MemoryAdaptiveDispatcher(max_session_permit=max_concurrent)
    results = await crawler.arun_many(urls=urls, config=run_config, dispatcher=dispatcher)
    output = []
    for r in results:
        if r.success and r.markdown:
            output.append({"url": r.url, "markdown": r.markdown})
        else:
            logger.warning(f"Batch crawl failed for {r.url}: {getattr(r, 'error_message', 'unknown')}")
    return output


def _pop_current_frontier(queue: List[tuple[str, int]]) -> tuple[List[tuple[str, int]], List[tuple[str, int]]]:
    if not queue:
        return [], []
    depth = queue[0][1]
    current_level: List[tuple[str, int]] = []
    remaining_queue: List[tuple[str, int]] = []
    for url, current_depth in queue:
        if current_depth == depth:
            current_level.append((url, current_depth))
        else:
            remaining_queue.append((url, current_depth))
    return current_level, remaining_queue


def _matches_pattern(url: str, pattern_re: Optional[re.Pattern[str]]) -> bool:
    return pattern_re is None or pattern_re.search(url) is not None


def _collect_urls_to_crawl(
    current_level: List[tuple[str, int]],
    visited: set[str],
    pattern_re: Optional[re.Pattern[str]],
) -> List[str]:
    urls_to_crawl: List[str] = []
    for url, _ in current_level:
        defragged, _ = urldefrag(url)
        if defragged in visited:
            continue
        if not _matches_pattern(defragged, pattern_re):
            continue
        visited.add(defragged)
        urls_to_crawl.append(defragged)
    return urls_to_crawl


def _internal_hrefs(result: Any) -> List[str]:
    links = getattr(result, "links", None)
    if not links:
        return []
    internal = links.get("internal", [])
    hrefs: List[str] = []
    for link_info in internal:
        href = link_info.get("href", "")
        if href:
            hrefs.append(href)
    return hrefs


def _collect_next_depth_urls(detail_results: List[Any], visited: set[str], next_depth: int) -> List[tuple[str, int]]:
    next_urls: List[tuple[str, int]] = []
    for result in detail_results:
        next_urls.extend(_result_next_depth_urls(result, visited, next_depth))
    return next_urls


def _result_next_depth_urls(result: Any, visited: set[str], next_depth: int) -> List[tuple[str, int]]:
    if not getattr(result, "success", False):
        return []
    base_domain = urlparse(getattr(result, "url", "")).netloc
    return _hrefs_to_next_depth_urls(_internal_hrefs(result), base_domain, visited, next_depth)


def _hrefs_to_next_depth_urls(
    hrefs: List[str],
    base_domain: str,
    visited: set[str],
    next_depth: int,
) -> List[tuple[str, int]]:
    next_urls: List[tuple[str, int]] = []
    for href in hrefs:
        defragged, _ = urldefrag(href)
        if urlparse(defragged).netloc != base_domain:
            continue
        if defragged in visited:
            continue
        next_urls.append((defragged, next_depth))
    return next_urls


async def _crawl_detail_results(
    crawler: AsyncWebCrawler,
    urls_to_crawl: List[str],
    max_concurrent: int,
) -> List[Any]:
    run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
    dispatcher = MemoryAdaptiveDispatcher(max_session_permit=max_concurrent)
    return await crawler.arun_many(urls=urls_to_crawl, config=run_config, dispatcher=dispatcher)


async def _process_crawl_frontier(
    crawler: AsyncWebCrawler,
    current_level: List[tuple[str, int]],
    queue: List[tuple[str, int]],
    visited: set[str],
    pattern_re: Optional[re.Pattern[str]],
    max_depth: int,
    max_concurrent: int,
) -> tuple[List[tuple[str, int]], List[Dict[str, Any]]]:
    current_depth = current_level[0][1]
    urls_to_crawl = _collect_urls_to_crawl(current_level, visited, pattern_re)
    if not urls_to_crawl:
        return queue, []

    batch_results = await crawl_batch(crawler, urls_to_crawl, max_concurrent)
    if current_depth >= max_depth:
        return queue, batch_results

    detail_results = await _crawl_detail_results(crawler, urls_to_crawl, max_concurrent)
    next_queue = queue + _collect_next_depth_urls(detail_results, visited, current_depth + 1)
    return next_queue, batch_results


async def crawl_recursive_internal_links(
    crawler: AsyncWebCrawler,
    start_urls: List[str],
    max_depth: int = 3,
    max_concurrent: int = 10,
    url_pattern: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Recursively crawl internal links up to max_depth levels.
    Uses BFS to avoid revisiting pages.
    """
    visited: set[str] = set()
    results: List[Dict[str, Any]] = []
    queue = [(url, 0) for url in start_urls]

    pattern_re = re.compile(url_pattern) if url_pattern else None

    while queue:
        current_level, queue = _pop_current_frontier(queue)
        queue, batch_results = await _process_crawl_frontier(
            crawler,
            current_level,
            queue,
            visited,
            pattern_re,
            max_depth,
            max_concurrent,
        )
        results.extend(batch_results)

    return results
