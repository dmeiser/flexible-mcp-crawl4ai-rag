"""Functions for extracting metadata from crawled content."""

import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse


def _markdown_links(markdown: str) -> List[tuple[str, str]]:
    """Return (text, href) markdown link tuples from content."""
    return re.findall(r"\[(?P<text>[^\]]*)\]\((?P<url>[^)]+)\)", markdown)


def _is_external_link(href: str, base_host: str) -> bool:
    """Return True if href points to a different host than base_host."""
    host = urlparse(href).netloc
    return bool(host and base_host and host != base_host)


def _empty_link_graph() -> Dict[str, Any]:
    return {
        "total_links": 0,
        "internal_links": 0,
        "external_links": 0,
        "links": [],
    }


def _resolve_base_host(base_url: Optional[str]) -> str:
    if not isinstance(base_url, str) or not base_url:
        return ""
    return urlparse(base_url).netloc


def _build_link_entries(markdown: str, base_host: str) -> List[Dict[str, Any]]:
    links: List[Dict[str, Any]] = []
    for text, href in _markdown_links(markdown):
        clean_href = href.strip()
        if not clean_href:
            continue
        links.append(
            {
                "url": clean_href,
                "anchor_text": text.strip(),
                "is_external": _is_external_link(clean_href, base_host),
            }
        )
    return links


def _count_external_links(links: List[Dict[str, Any]]) -> int:
    return sum(1 for link in links if link.get("is_external"))


def extract_section_info(chunk: str) -> Dict[str, Any]:
    """
    Extracts headers and stats from a chunk.

    Args:
        chunk: Markdown chunk

    Returns:
        Dictionary with headers and stats
    """
    headers = re.findall(r"^(#+)\s+(.+)$", chunk, re.MULTILINE)
    header_str = "; ".join([f"{h[0]} {h[1]}" for h in headers]) if headers else ""

    return {"headers": header_str, "char_count": len(chunk), "word_count": len(chunk.split())}


def extract_link_graph(markdown: str, base_url: Optional[str] = None) -> Dict[str, Any]:
    """Extract lightweight link graph information from markdown text."""
    if not markdown:
        return _empty_link_graph()

    base_host = _resolve_base_host(base_url)
    links = _build_link_entries(markdown, base_host)
    external = _count_external_links(links)
    internal = len(links) - external

    return {
        "total_links": len(links),
        "internal_links": internal,
        "external_links": external,
        "links": links,
    }


def _empty_media_metadata() -> Dict[str, Any]:
    return {
        "images": [],
        "media_links": [],
        "image_count": 0,
        "media_link_count": 0,
    }


def _extract_images(markdown: str) -> List[Dict[str, str]]:
    images: List[Dict[str, str]] = []
    image_matches = re.findall(r"!\[(?P<alt>[^\]]*)\]\((?P<url>[^)]+)\)", markdown)
    for alt, url in image_matches:
        clean_url = url.strip()
        if not clean_url:
            continue
        images.append({"url": clean_url, "alt_text": alt.strip()})
    return images


def _extract_media_links(markdown: str, media_suffixes: tuple[str, ...]) -> List[Dict[str, str]]:
    media_links: List[Dict[str, str]] = []
    for text, url in _markdown_links(markdown):
        clean_url = url.strip()
        if not clean_url:
            continue
        if not clean_url.lower().endswith(media_suffixes):
            continue
        media_links.append({"url": clean_url, "text": text.strip()})
    return media_links


def extract_media_metadata(markdown: str) -> Dict[str, Any]:
    """Extract image/video/audio style references from markdown text."""
    if not markdown:
        return _empty_media_metadata()

    images = _extract_images(markdown)

    media_suffixes = (
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".svg",
        ".mp4",
        ".webm",
        ".mp3",
        ".wav",
        ".pdf",
    )
    media_links = _extract_media_links(markdown, media_suffixes)

    return {
        "images": images,
        "media_links": media_links,
        "image_count": len(images),
        "media_link_count": len(media_links),
    }
