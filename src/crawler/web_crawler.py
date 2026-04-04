"""Core crawling and chunking logic for web pages."""
import logging
import re
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, urldefrag
from xml.etree import ElementTree

import httpx
import nltk
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode, MemoryAdaptiveDispatcher

from ..utils import settings, ChunkStrategy

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
    except httpx.HTTPStatusError as exc:
        logger.error(f"HTTP error fetching sitemap {sitemap_url}: {exc.response.status_code}")
    except httpx.RequestError as exc:
        logger.error(f"Request error fetching sitemap {sitemap_url}: {exc}")
    except ElementTree.ParseError as exc:
        logger.error(f"XML parse error for sitemap {sitemap_url}: {exc}")
    except Exception as exc:
        logger.error(f"Unexpected error parsing sitemap {sitemap_url}: {exc}")
    return []


# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------

def _fixed_char_chunking(text: str, size: int, overlap: int) -> List[str]:
    if not text:
        return []
    if size <= 0:
        return [text] if text.strip() else []
    chunks = []
    start = 0
    step = max(1, size - overlap)
    while start < len(text):
        chunk = text[start : start + size]
        if chunk.strip():
            chunks.append(chunk)
        start += step
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


def _sentence_chunking(text: str, size: int, overlap: int) -> List[str]:
    if not text:
        return []
    try:
        sentences = nltk.sent_tokenize(text)
    except Exception as exc:
        logger.warning(f"NLTK sent_tokenize failed ({exc}), falling back to paragraph chunking.")
        return _paragraph_chunking(text, size, overlap)

    if not sentences:
        return []

    # Build chunks greedily
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent) + (1 if current else 0)
        if current_len + sent_len > size and current:
            chunks.append(" ".join(current))
            # Back-track for overlap
            if overlap > 0:
                overlap_sents: List[str] = []
                overlap_len = 0
                for s in reversed(current):
                    if overlap_len + len(s) > overlap:
                        break
                    overlap_sents.insert(0, s)
                    overlap_len += len(s) + 1
                current = overlap_sents
                current_len = sum(len(s) for s in current) + max(0, len(current) - 1)
            else:
                current = []
                current_len = 0
        current.append(sent)
        current_len += len(sent) + (1 if len(current) > 1 else 0)

    if current:
        chunks.append(" ".join(current))

    return [c for c in chunks if c.strip()]


async def chunk_text_according_to_settings(text: str) -> List[str]:
    """Chunk text using the strategy and parameters from settings."""
    if not text:
        return []
    size = settings.CHUNK_SIZE
    overlap = settings.CHUNK_OVERLAP
    strategy = settings.CHUNK_STRATEGY

    if strategy == ChunkStrategy.FIXED:
        return _fixed_char_chunking(text, size, overlap)
    elif strategy == ChunkStrategy.SENTENCE:
        return _sentence_chunking(text, size, overlap)
    elif strategy == ChunkStrategy.PARAGRAPH:
        return _paragraph_chunking(text, size, overlap)
    # Default fallback
    return _paragraph_chunking(text, size, overlap)


# ---------------------------------------------------------------------------
# Crawl helpers
# ---------------------------------------------------------------------------

async def crawl_markdown_file(
    crawler: AsyncWebCrawler, url: str
) -> List[Dict[str, Any]]:
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
    results = await crawler.arun_many(
        urls=urls, config=run_config, dispatcher=dispatcher
    )
    output = []
    for r in results:
        if r.success and r.markdown:
            output.append({"url": r.url, "markdown": r.markdown})
        else:
            logger.warning(f"Batch crawl failed for {r.url}: {getattr(r, 'error_message', 'unknown')}")
    return output


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
    visited: set = set()
    results: List[Dict[str, Any]] = []
    queue = [(url, 0) for url in start_urls]

    pattern_re = re.compile(url_pattern) if url_pattern else None

    while queue:
        # Collect all URLs at the current frontier (same depth level)
        current_level = [(u, d) for u, d in queue if d == queue[0][1]]
        queue = [(u, d) for u, d in queue if d != queue[0][1]]

        current_depth = current_level[0][1] if current_level else 0
        urls_to_crawl = []
        for u, _ in current_level:
            defragged, _ = urldefrag(u)
            if defragged not in visited:
                if pattern_re is None or pattern_re.search(defragged):
                    visited.add(defragged)
                    urls_to_crawl.append(defragged)

        if not urls_to_crawl:
            continue

        batch_results = await crawl_batch(crawler, urls_to_crawl, max_concurrent)
        results.extend(batch_results)

        if current_depth < max_depth:
            # Collect new internal links
            run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
            dispatcher = MemoryAdaptiveDispatcher(max_session_permit=max_concurrent)
            detail_results = await crawler.arun_many(
                urls=urls_to_crawl, config=run_config, dispatcher=dispatcher
            )
            for r in detail_results:
                if r.success and r.links:
                    base_domain = urlparse(r.url).netloc
                    for link_info in r.links.get("internal", []):
                        href = link_info.get("href", "")
                        if href:
                            defragged, _ = urldefrag(href)
                            link_domain = urlparse(defragged).netloc
                            if link_domain == base_domain and defragged not in visited:
                                queue.append((defragged, current_depth + 1))

    return results
